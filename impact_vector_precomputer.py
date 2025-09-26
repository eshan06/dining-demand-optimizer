"""
ImpactVectorPrecomputer module for dining demand optimization system.

This module precomputes demand impact vectors for instant "what-if" analysis,
enabling real-time UI updates when users adjust schedule parameters.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import time

from demand_service import DemandService
from time_grid_service import TimeGridService, MealPeriod, TimeBin
from sections_catalog import SectionsCatalog
from availability_service import AvailabilityService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ImpactVector:
    """Represents a demand impact vector for a section-slot combination."""
    section_id: str
    slot_minutes: int
    bin_index: int
    demand_delta: float
    
    def __str__(self) -> str:
        return f"ImpactVector(section={self.section_id}, slot={self.slot_minutes}, bin={self.bin_index}, delta={self.demand_delta:.3f})"


@dataclass
class PrecomputeConfig:
    """Configuration for impact vector precomputation."""
    max_shift_minutes: int = 120  # Maximum shift allowed in UI (2 hours)
    shift_step_minutes: int = 15  # Step size for slot generation
    min_shift_minutes: int = -120  # Minimum shift allowed
    cache_invalidation_threshold: float = 0.01  # Threshold for cache invalidation
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_shift_minutes <= self.min_shift_minutes:
            raise ValueError("max_shift_minutes must be greater than min_shift_minutes")
        if self.shift_step_minutes <= 0:
            raise ValueError("shift_step_minutes must be positive")


class ImpactVectorPrecomputer:
    """
    Precomputes demand impact vectors for instant what-if analysis.
    
    For each section and each allowed slot, computes the demand delta
    that would result from moving the section to that time slot.
    """
    
    def __init__(self, 
                 demand_service: DemandService,
                 time_grid_service: TimeGridService,
                 sections_catalog: SectionsCatalog,
                 availability_service: AvailabilityService,
                 meal_propensity_service: 'MealPropensityService',
                 config: Optional[PrecomputeConfig] = None):
        """
        Initialize ImpactVectorPrecomputer.
        
        Args:
            demand_service: DemandService instance for demand calculations
            time_grid_service: TimeGridService instance for time bin management
            sections_catalog: SectionsCatalog instance for section data
            availability_service: AvailabilityService instance for availability tracking
            config: Configuration for precomputation
        """
        self.demand_service = demand_service
        self.time_grid = time_grid_service
        self.sections_catalog = sections_catalog
        self.availability_service = availability_service
        self.meal_propensity_service = meal_propensity_service
        self.config = config or PrecomputeConfig()
        
        # Storage for impact vectors
        self.impact_vectors = {}  # (section_id, slot_minutes) -> List[ImpactVector]
        self.baseline_demand = {}  # bin_index -> baseline_demand
        self.is_precomputed = False
        self.last_precompute_time = None
        
        # Cache invalidation tracking
        self.last_bin_size = None
        self.last_meal_windows = None
        
        logger.info("Initialized ImpactVectorPrecomputer")
    
    def precompute_impact_vectors(self, student_ids: List[str]) -> Dict[str, int]:
        """
        Precompute impact vectors for all sections and allowed slots.
        
        Args:
            student_ids: List of student IDs to include in calculations
            
        Returns:
            Dictionary with precomputation statistics
        """
        logger.info("Starting impact vector precomputation...")
        start_time = time.time()
        
        # Check if we need to invalidate cache
        if self._should_invalidate_cache():
            self._invalidate_cache()
        
        # Generate allowed slots
        allowed_slots = self._generate_allowed_slots()
        logger.info(f"Generated {len(allowed_slots)} allowed slots")
        
        # Compute baseline demand
        self._compute_baseline_demand(student_ids)
        
        # Precompute impact vectors for each section
        total_vectors = 0
        sections_processed = 0
        
        for section_id in self.sections_catalog.sections_dict.keys():
            try:
                section_vectors = self._precompute_section_impact_vectors(
                    section_id, allowed_slots, student_ids
                )
                self.impact_vectors[section_id] = section_vectors
                total_vectors += len(section_vectors)
                sections_processed += 1
                
                if sections_processed % 10 == 0:
                    logger.info(f"Processed {sections_processed} sections...")
                    
            except Exception as e:
                logger.error(f"Error processing section {section_id}: {e}")
                continue
        
        # Update cache invalidation tracking
        self._update_cache_tracking()
        
        self.is_precomputed = True
        self.last_precompute_time = time.time()
        
        elapsed_time = time.time() - start_time
        
        stats = {
            'sections_processed': sections_processed,
            'total_vectors': total_vectors,
            'allowed_slots': len(allowed_slots),
            'elapsed_time': elapsed_time,
            'vectors_per_second': total_vectors / elapsed_time if elapsed_time > 0 else 0
        }
        
        logger.info(f"Precomputation completed in {elapsed_time:.2f}s")
        logger.info(f"Generated {total_vectors} impact vectors for {sections_processed} sections")
        
        return stats
    
    def get_impact_vector(self, section_id: str, slot_minutes: int) -> List[ImpactVector]:
        """
        Get impact vector for a specific section-slot combination.
        
        Args:
            section_id: Section identifier
            slot_minutes: Time slot in minutes from midnight
            
        Returns:
            List of ImpactVector objects for this section-slot combination
        """
        if not self.is_precomputed:
            raise ValueError("Impact vectors not precomputed. Call precompute_impact_vectors() first.")
        
        key = (section_id, slot_minutes)
        return self.impact_vectors.get(section_id, {}).get(slot_minutes, [])
    
    def get_what_if_demand(self, section_id: str, slot_minutes: int) -> Dict[int, float]:
        """
        Get what-if demand by adding impact vectors to baseline demand.
        
        Args:
            section_id: Section identifier
            slot_minutes: Time slot in minutes from midnight
            
        Returns:
            Dictionary mapping bin_index to what-if demand
        """
        if not self.is_precomputed:
            raise ValueError("Impact vectors not precomputed. Call precompute_impact_vectors() first.")
        
        # Start with baseline demand (convert BinDemand objects to floats)
        what_if_demand = {bin_idx: bin_demand.expected_demand for bin_idx, bin_demand in self.baseline_demand.items()}
        
        # Add impact vectors
        impact_vectors = self.get_impact_vector(section_id, slot_minutes)
        for vector in impact_vectors:
            what_if_demand[vector.bin_index] += vector.demand_delta
        
        return what_if_demand
    
    def get_impact_summary(self, section_id: str, slot_minutes: int) -> Dict[str, float]:
        """
        Get summary statistics for a section-slot impact.
        
        Args:
            section_id: Section identifier
            slot_minutes: Time slot in minutes from midnight
            
        Returns:
            Dictionary with impact summary statistics
        """
        impact_vectors = self.get_impact_vector(section_id, slot_minutes)
        
        if not impact_vectors:
            return {
                'total_impact': 0.0,
                'max_positive_impact': 0.0,
                'max_negative_impact': 0.0,
                'affected_bins': 0,
                'peak_impact_bin': None
            }
        
        deltas = [vector.demand_delta for vector in impact_vectors]
        bin_indices = [vector.bin_index for vector in impact_vectors]
        
        # Find peak impact bin
        peak_impact_idx = np.argmax(np.abs(deltas))
        peak_impact_bin = bin_indices[peak_impact_idx]
        
        return {
            'total_impact': sum(deltas),
            'max_positive_impact': max(deltas) if deltas else 0.0,
            'max_negative_impact': min(deltas) if deltas else 0.0,
            'affected_bins': len([d for d in deltas if abs(d) > 1e-6]),
            'peak_impact_bin': peak_impact_bin,
            'peak_impact_delta': deltas[peak_impact_idx]
        }
    
    def invalidate_cache(self) -> None:
        """Invalidate the precomputed cache."""
        logger.info("Invalidating impact vector cache...")
        self._invalidate_cache()
    
    def _generate_allowed_slots(self) -> List[int]:
        """Generate list of allowed time slots based on configuration."""
        slots = []
        current_slot = self.config.min_shift_minutes
        
        while current_slot <= self.config.max_shift_minutes:
            slots.append(current_slot)
            current_slot += self.config.shift_step_minutes
        
        return slots
    
    def _compute_baseline_demand(self, student_ids: List[str]) -> None:
        """Compute baseline demand for all time bins."""
        logger.info("Computing baseline demand...")
        
        # Get baseline demand from demand service
        baseline_demands, _ = self.demand_service.compute_demand_for_schedule(student_ids)
        
        # Store baseline demand by bin index
        self.baseline_demand = {}
        for bin_idx, demand in enumerate(baseline_demands):
            self.baseline_demand[bin_idx] = demand
        
        logger.info(f"Computed baseline demand for {len(self.baseline_demand)} bins")
    
    def _precompute_section_impact_vectors(self, 
                                         section_id: str, 
                                         allowed_slots: List[int],
                                         student_ids: List[str]) -> Dict[int, List[ImpactVector]]:
        """
        Precompute impact vectors for a specific section.
        
        Args:
            section_id: Section identifier
            allowed_slots: List of allowed time slots
            student_ids: List of student IDs
            
        Returns:
            Dictionary mapping slot_minutes to list of ImpactVector objects
        """
        section_vectors = {}
        
        # Get original section data
        if section_id not in self.sections_catalog.sections_dict:
            logger.warning(f"Section {section_id} not found in catalog")
            return section_vectors
        
        original_section = self.sections_catalog.sections_dict[section_id]
        original_start_min = original_section.start_minutes
        original_end_min = original_section.end_minutes
        section_duration = original_end_min - original_start_min
        
        # Get students enrolled in this section
        enrolled_students = self.sections_catalog.get_students_in_section(section_id)
        if not enrolled_students:
            logger.warning(f"No students enrolled in section {section_id}")
            return section_vectors
        
        for slot_minutes in allowed_slots:
            try:
                # Calculate new section times
                new_start_min = original_start_min + slot_minutes
                new_end_min = new_start_min + section_duration
                
                # Skip if new times are outside valid range
                if (new_start_min < 0 or 
                    new_end_min > self.time_grid.day_end_minutes or
                    new_start_min >= new_end_min):
                    continue
                
                # Compute impact vectors for this slot
                slot_vectors = self._compute_slot_impact_vectors(
                    section_id, slot_minutes, new_start_min, new_end_min,
                    enrolled_students, student_ids
                )
                
                if slot_vectors:
                    section_vectors[slot_minutes] = slot_vectors
                    
            except Exception as e:
                logger.error(f"Error computing impact for section {section_id}, slot {slot_minutes}: {e}")
                continue
        
        return section_vectors
    
    def _compute_slot_impact_vectors(self, 
                                   section_id: str,
                                   slot_minutes: int,
                                   new_start_min: int,
                                   new_end_min: int,
                                   enrolled_students: List[str],
                                   all_student_ids: List[str]) -> List[ImpactVector]:
        """
        Compute impact vectors for a specific section-slot combination.
        
        Args:
            section_id: Section identifier
            slot_minutes: Time slot offset
            new_start_min: New start time in minutes
            new_end_min: New end time in minutes
            enrolled_students: Students enrolled in this section
            all_student_ids: All student IDs for demand calculation
            
        Returns:
            List of ImpactVector objects
        """
        impact_vectors = []
        
        # Get time bins that would be covered by the new section time
        new_time_bins = self.time_grid.get_bins_for_time_range(new_start_min, new_end_min)
        new_bin_indices = [bin_obj.index for bin_obj in new_time_bins]
        
        # Get original time bins
        original_section = self.sections_catalog.sections_dict[section_id]
        original_bin_indices = original_section.time_bins
        
        # Compute demand delta for each affected bin
        affected_bins = set(original_bin_indices) | set(new_bin_indices)
        
        for bin_idx in affected_bins:
            # Compute demand with section in original position
            original_demand = self._compute_bin_demand_with_section_position(
                bin_idx, section_id, original_bin_indices, enrolled_students, all_student_ids
            )
            
            # Compute demand with section in new position
            new_demand = self._compute_bin_demand_with_section_position(
                bin_idx, section_id, new_bin_indices, enrolled_students, all_student_ids
            )
            
            # Calculate delta
            demand_delta = new_demand - original_demand
            
            # Only store non-zero deltas to save space
            if abs(demand_delta) > 1e-6:
                impact_vector = ImpactVector(
                    section_id=section_id,
                    slot_minutes=slot_minutes,
                    bin_index=bin_idx,
                    demand_delta=demand_delta
                )
                impact_vectors.append(impact_vector)
        
        return impact_vectors
    
    def _compute_bin_demand_with_section_position(self, 
                                                bin_idx: int,
                                                section_id: str,
                                                section_bin_indices: List[int],
                                                enrolled_students: List[str],
                                                all_student_ids: List[str]) -> float:
        """
        Compute demand for a specific bin with section in a specific position.
        
        Args:
            bin_idx: Time bin index
            section_id: Section identifier
            section_bin_indices: List of bin indices covered by section
            enrolled_students: Students enrolled in this section
            all_student_ids: All student IDs for demand calculation
            
        Returns:
            Expected demand for the bin
        """
        # This is a simplified version - in practice, you'd want to use
        # the actual demand service logic here
        bin_obj = self.time_grid.get_bin(bin_idx)
        if not bin_obj:
            return 0.0
        
        total_demand = 0.0
        
        for student_id in all_student_ids:
            # Check if student is enrolled in this section
            is_enrolled = student_id in enrolled_students
            
            # Determine if student has class in this bin
            has_class_in_bin = False
            if is_enrolled and bin_idx in section_bin_indices:
                has_class_in_bin = True
            else:
                # Check other sections for this student
                student_availability = self.availability_service.get_student_availability(student_id)
                if student_availability and bin_idx in student_availability.class_bins:
                    has_class_in_bin = True
            
            # If student has class, demand is 0
            if has_class_in_bin:
                continue
            
            # Get student's meal propensity for this time period
            meal_propensity = self.meal_propensity_service.get_student_propensity(student_id)
            
            # Determine meal period
            if bin_obj.meal_period == MealPeriod.BREAKFAST:
                propensity = meal_propensity.p_breakfast
            elif bin_obj.meal_period == MealPeriod.LUNCH:
                propensity = meal_propensity.p_lunch
            elif bin_obj.meal_period == MealPeriod.DINNER:
                propensity = meal_propensity.p_dinner
            else:
                propensity = 0.0
            
            # Apply meal plan constraints (simplified)
            # In practice, you'd want to use the full meal plan logic
            total_demand += propensity
        
        return total_demand
    
    def _should_invalidate_cache(self) -> bool:
        """Check if cache should be invalidated based on configuration changes."""
        current_bin_size = self.time_grid.bin_size_minutes
        current_meal_windows = {
            'breakfast': self.time_grid.breakfast_window,
            'lunch': self.time_grid.lunch_window,
            'dinner': self.time_grid.dinner_window
        }
        
        if (self.last_bin_size != current_bin_size or 
            self.last_meal_windows != current_meal_windows):
            return True
        
        return False
    
    def _invalidate_cache(self) -> None:
        """Invalidate the precomputed cache."""
        self.impact_vectors.clear()
        self.baseline_demand.clear()
        self.is_precomputed = False
        self.last_precompute_time = None
    
    def _update_cache_tracking(self) -> None:
        """Update cache invalidation tracking variables."""
        self.last_bin_size = self.time_grid.bin_size_minutes
        self.last_meal_windows = {
            'breakfast': self.time_grid.breakfast_window,
            'lunch': self.time_grid.lunch_window,
            'dinner': self.time_grid.dinner_window
        }
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_vectors = sum(
            len(slot_vectors) 
            for section_vectors in self.impact_vectors.values() 
            for slot_vectors in section_vectors.values()
        )
        
        return {
            'is_precomputed': self.is_precomputed,
            'last_precompute_time': self.last_precompute_time,
            'total_sections': len(self.impact_vectors),
            'total_vectors': total_vectors,
            'baseline_bins': len(self.baseline_demand),
            'cache_size_mb': self._estimate_cache_size_mb()
        }
    
    def _estimate_cache_size_mb(self) -> float:
        """Estimate cache size in MB."""
        # Rough estimation: each ImpactVector is about 32 bytes
        total_vectors = sum(
            len(slot_vectors) 
            for section_vectors in self.impact_vectors.values() 
            for slot_vectors in section_vectors.values()
        )
        
        bytes_per_vector = 32
        total_bytes = total_vectors * bytes_per_vector
        return total_bytes / (1024 * 1024)


def main():
    """Test the ImpactVectorPrecomputer with sample data."""
    from dataloader import DataLoader
    from meal_plan_normalizer import MealPlanNormalizer
    from meal_propensity_service import MealPropensityService
    from feature_builder import FeatureBuilder
    from ml_model_trainer import MLModelTrainer
    
    # Initialize all services
    logger.info("Initializing services...")
    dl = DataLoader()
    dl.load_all_data()
    
    tg = TimeGridService()
    sc = SectionsCatalog(tg)
    sc.build_catalog(dl.class_enrollments)
    
    av = AvailabilityService(tg, sc)
    av.compute_availability(list(dl.students_data['student_id']))
    
    mp = MealPropensityService()
    mp.compute_meal_propensities(dl.swipes_data, list(dl.students_data['student_id']))
    
    fb = FeatureBuilder(tg, mp, av, sc)
    training_df = fb.build_training_matrix(list(dl.students_data['student_id'])[:50])
    
    mt = MLModelTrainer(fb, tg)
    mt.train_model(training_df)
    
    ds = DemandService(mt, tg, av, sc)
    
    # Initialize ImpactVectorPrecomputer
    config = PrecomputeConfig(
        max_shift_minutes=60,  # 1 hour max shift
        shift_step_minutes=15,  # 15-minute steps
        min_shift_minutes=-60   # 1 hour min shift
    )
    
    ivp = ImpactVectorPrecomputer(ds, tg, sc, av, mp, config)
    
    # Test with a subset of students
    test_students = list(dl.students_data['student_id'])[:100]
    
    # Precompute impact vectors
    stats = ivp.precompute_impact_vectors(test_students)
    print(f"Precomputation stats: {stats}")
    
    # Test impact vector retrieval
    section_id = list(sc.sections_dict.keys())[0]
    slot_minutes = 30  # 30-minute shift
    
    impact_vectors = ivp.get_impact_vector(section_id, slot_minutes)
    print(f"Impact vectors for section {section_id}, slot {slot_minutes}: {len(impact_vectors)}")
    
    # Test what-if demand
    what_if_demand = ivp.get_what_if_demand(section_id, slot_minutes)
    print(f"What-if demand for section {section_id}, slot {slot_minutes}: {len(what_if_demand)} bins")
    
    # Test impact summary
    impact_summary = ivp.get_impact_summary(section_id, slot_minutes)
    print(f"Impact summary: {impact_summary}")
    
    # Test cache stats
    cache_stats = ivp.get_cache_stats()
    print(f"Cache stats: {cache_stats}")


if __name__ == "__main__":
    main()
