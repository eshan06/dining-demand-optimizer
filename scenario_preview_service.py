"""
ScenarioPreviewService module for dining demand optimization system.

This module provides live preview functionality for manual schedule edits,
enabling "drag to try" feedback without running the full optimizer.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EditValidationResult:
    """Result of validating a manual edit."""
    is_valid: bool
    section_id: str
    original_start_min: int
    proposed_start_min: int
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


@dataclass
class PreviewResult:
    """Result of a scenario preview."""
    success: bool
    demand_series: Dict[int, float]  # bin_index -> demand
    metrics: Dict[str, float]
    applied_edits: Dict[str, int]  # section_id -> new_start_min
    validation_results: List[EditValidationResult]
    computation_time: float
    method_used: str  # 'impact_vectors' or 'full_recompute'
    message: str


class ScenarioPreviewService:
    """
    Service for providing live preview of manual schedule edits.
    
    This service enables instant feedback when users manually adjust
    section schedules in the UI, without running the full optimizer.
    """
    
    def __init__(self, 
                 impact_vector_precomputer,
                 demand_service,
                 time_grid_service,
                 sections_catalog,
                 max_impact_vector_edits: int = 10):
        """
        Initialize the ScenarioPreviewService.
        
        Args:
            impact_vector_precomputer: Precomputed impact vectors for fast updates
            demand_service: Full demand service for complex scenarios
            time_grid_service: Time grid service for bin information
            sections_catalog: Sections catalog for section information
            max_impact_vector_edits: Maximum edits to use impact vectors (vs full recompute)
        """
        self.ivp = impact_vector_precomputer
        self.demand_service = demand_service
        self.time_grid = time_grid_service
        self.sections_catalog = sections_catalog
        self.max_impact_vector_edits = max_impact_vector_edits
        
        # Check if impact vectors are precomputed
        if not self.ivp.is_precomputed:
            raise ValueError("Impact vectors must be precomputed before using preview service")
    
    def preview_scenario(self, 
                        manual_edits: Dict[str, int],
                        student_ids: List[str]) -> PreviewResult:
        """
        Preview the impact of manual schedule edits.
        
        Args:
            manual_edits: Dictionary mapping section_id to new start_minutes
            student_ids: List of student IDs for demand calculation
            
        Returns:
            PreviewResult with updated demand and metrics
        """
        start_time = time.time()
        
        try:
            # Validate all edits
            validation_results = self._validate_edits(manual_edits)
            valid_edits = {edit.section_id: edit.proposed_start_min 
                          for edit in validation_results if edit.is_valid}
            
            if not valid_edits:
                return PreviewResult(
                    success=False,
                    demand_series={},
                    metrics={},
                    applied_edits={},
                    validation_results=validation_results,
                    computation_time=time.time() - start_time,
                    method_used='none',
                    message="No valid edits to apply"
                )
            
            # Choose computation method based on number of edits
            if len(valid_edits) <= self.max_impact_vector_edits:
                demand_series, method_used = self._compute_demand_with_impact_vectors(valid_edits)
            else:
                demand_series, method_used = self._compute_demand_with_full_recompute(valid_edits, student_ids)
            
            # Calculate metrics
            metrics = self._calculate_preview_metrics(demand_series, valid_edits)
            
            computation_time = time.time() - start_time
            
            return PreviewResult(
                success=True,
                demand_series=demand_series,
                metrics=metrics,
                applied_edits=valid_edits,
                validation_results=validation_results,
                computation_time=computation_time,
                method_used=method_used,
                message=f"Preview completed using {method_used}"
            )
            
        except Exception as e:
            logger.error(f"Preview failed: {e}")
            return PreviewResult(
                success=False,
                demand_series={},
                metrics={},
                applied_edits={},
                validation_results=[],
                computation_time=time.time() - start_time,
                method_used='error',
                message=f"Preview failed: {e}"
            )
    
    def _validate_edits(self, manual_edits: Dict[str, int]) -> List[EditValidationResult]:
        """
        Validate manual edits against constraints and allowed slots.
        
        Args:
            manual_edits: Dictionary mapping section_id to new start_minutes
            
        Returns:
            List of EditValidationResult objects
        """
        validation_results = []
        
        for section_id, proposed_start_min in manual_edits.items():
            # Check if section exists
            if section_id not in self.sections_catalog.sections_dict:
                validation_results.append(EditValidationResult(
                    is_valid=False,
                    section_id=section_id,
                    original_start_min=0,
                    proposed_start_min=proposed_start_min,
                    error_message=f"Section {section_id} not found"
                ))
                continue
            
            section = self.sections_catalog.sections_dict[section_id]
            original_start_min = section.start_minutes
            
            # Check if the proposed time is within allowed range
            if not self._is_time_within_allowed_range(proposed_start_min):
                validation_results.append(EditValidationResult(
                    is_valid=False,
                    section_id=section_id,
                    original_start_min=original_start_min,
                    proposed_start_min=proposed_start_min,
                    error_message=f"Proposed time {proposed_start_min} is outside allowed range"
                ))
                continue
            
            # Check if the proposed time is an allowed slot for this section
            if not self._is_allowed_slot(section_id, proposed_start_min):
                validation_results.append(EditValidationResult(
                    is_valid=False,
                    section_id=section_id,
                    original_start_min=original_start_min,
                    proposed_start_min=proposed_start_min,
                    error_message=f"Proposed time {proposed_start_min} is not an allowed slot for section {section_id}"
                ))
                continue
            
            # Check for conflicts with other sections (optional - could be expensive)
            conflict_warning = self._check_for_conflicts(section_id, proposed_start_min)
            
            validation_results.append(EditValidationResult(
                is_valid=True,
                section_id=section_id,
                original_start_min=original_start_min,
                proposed_start_min=proposed_start_min,
                warning_message=conflict_warning
            ))
        
        return validation_results
    
    def _is_time_within_allowed_range(self, start_minutes: int) -> bool:
        """Check if the proposed time is within the allowed day range."""
        return (self.time_grid.day_start_minutes <= start_minutes <= 
                self.time_grid.day_end_minutes - 60)  # Leave room for class duration
    
    def _is_allowed_slot(self, section_id: str, start_minutes: int) -> bool:
        """Check if the proposed time is an allowed slot for this section."""
        if section_id not in self.ivp.impact_vectors:
            return False
        
        return start_minutes in self.ivp.impact_vectors[section_id]
    
    def _check_for_conflicts(self, section_id: str, proposed_start_min: int) -> Optional[str]:
        """
        Check for potential conflicts with other sections.
        This is a simplified check - in practice, you might want more sophisticated conflict detection.
        """
        # Get the section duration
        section = self.sections_catalog.sections_dict[section_id]
        duration = section.end_minutes - section.start_minutes
        proposed_end_min = proposed_start_min + duration
        
        # Check for overlapping sections (simplified)
        overlapping_sections = []
        for other_section_id, other_section in self.sections_catalog.sections_dict.items():
            if other_section_id == section_id:
                continue
            
            if (other_section.start_minutes < proposed_end_min and 
                other_section.end_minutes > proposed_start_min):
                overlapping_sections.append(other_section_id)
        
        if overlapping_sections:
            return f"Potential conflict with sections: {overlapping_sections[:3]}{'...' if len(overlapping_sections) > 3 else ''}"
        
        return None
    
    def _compute_demand_with_impact_vectors(self, valid_edits: Dict[str, int]) -> Tuple[Dict[int, float], str]:
        """
        Compute demand using precomputed impact vectors (fast method).
        
        Args:
            valid_edits: Dictionary of valid section edits
            
        Returns:
            Tuple of (demand_series, method_used)
        """
        # Start with baseline demand
        demand_series = {bin_idx: bin_demand.expected_demand 
                        for bin_idx, bin_demand in self.ivp.baseline_demand.items()}
        
        # Apply impact vectors for each edit
        for section_id, new_start_min in valid_edits.items():
            impact_vectors = self.ivp.get_impact_vector(section_id, new_start_min)
            for vector in impact_vectors:
                demand_series[vector.bin_index] += vector.demand_delta
        
        return demand_series, 'impact_vectors'
    
    def _compute_demand_with_full_recompute(self, valid_edits: Dict[str, int], student_ids: List[str]) -> Tuple[Dict[int, float], str]:
        """
        Compute demand using full recomputation (slower but more accurate for many edits).
        
        Args:
            valid_edits: Dictionary of valid section edits
            student_ids: List of student IDs for demand calculation
            
        Returns:
            Tuple of (demand_series, method_used)
        """
        # Apply edits to the sections catalog temporarily
        original_sections = {}
        for section_id, new_start_min in valid_edits.items():
            section = self.sections_catalog.sections_dict[section_id]
            original_sections[section_id] = {
                'start_minutes': section.start_minutes,
                'end_minutes': section.end_minutes,
                'time_bins': section.time_bins.copy()
            }
            
            # Update section times
            duration = section.end_minutes - section.start_minutes
            section.start_minutes = new_start_min
            section.end_minutes = new_start_min + duration
            
            # Recompute time bins
            new_time_bins = self.time_grid.get_bins_for_time_range(section.start_minutes, section.end_minutes)
            section.time_bins = [tb.index for tb in new_time_bins]
        
        try:
            # Recompute demand with updated sections
            demand_series, _ = self.demand_service.compute_demand_for_schedule(student_ids)
            
            # Convert to simple dictionary format
            demand_dict = {bin_idx: bin_demand.expected_demand 
                          for bin_idx, bin_demand in demand_series.items()}
            
            return demand_dict, 'full_recompute'
            
        finally:
            # Restore original section times
            for section_id, original_data in original_sections.items():
                section = self.sections_catalog.sections_dict[section_id]
                section.start_minutes = original_data['start_minutes']
                section.end_minutes = original_data['end_minutes']
                section.time_bins = original_data['time_bins']
    
    def _calculate_preview_metrics(self, demand_series: Dict[int, float], applied_edits: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate metrics for the preview scenario.
        
        Args:
            demand_series: Dictionary mapping bin_index to demand
            applied_edits: Dictionary of applied edits
            
        Returns:
            Dictionary of metrics
        """
        if not demand_series:
            return {}
        
        demand_values = list(demand_series.values())
        
        # Basic demand metrics
        peak_demand = max(demand_values)
        total_demand = sum(demand_values)
        mean_demand = total_demand / len(demand_values)
        variance = sum((d - mean_demand)**2 for d in demand_values) / len(demand_values)
        
        # P95 demand (95th percentile)
        sorted_demands = sorted(demand_values)
        p95_index = int(0.95 * len(sorted_demands))
        p95_demand = sorted_demands[p95_index] if p95_index < len(sorted_demands) else sorted_demands[-1]
        
        # Calculate shift metrics
        total_shift_minutes = 0
        sections_moved = 0
        
        for section_id, new_start_min in applied_edits.items():
            section = self.sections_catalog.sections_dict[section_id]
            original_start_min = section.start_minutes
            if new_start_min != original_start_min:
                sections_moved += 1
                total_shift_minutes += abs(new_start_min - original_start_min)
        
        avg_shift_minutes = total_shift_minutes / sections_moved if sections_moved > 0 else 0
        percent_moved = (sections_moved / len(applied_edits)) * 100 if applied_edits else 0
        
        return {
            'peak_demand': peak_demand,
            'total_demand': total_demand,
            'mean_demand': mean_demand,
            'variance': variance,
            'p95_demand': p95_demand,
            'sections_moved': sections_moved,
            'percent_moved': percent_moved,
            'total_shift_minutes': total_shift_minutes,
            'avg_shift_minutes': avg_shift_minutes,
            'edits_applied': len(applied_edits)
        }
    
    def get_preview_summary(self, result: PreviewResult) -> str:
        """
        Get a human-readable summary of preview results.
        
        Args:
            result: PreviewResult object
            
        Returns:
            Formatted summary string
        """
        if not result.success:
            return f"Preview failed: {result.message}"
        
        summary = f"""
Scenario Preview Results:
========================
Success: {result.success}
Method: {result.method_used}
Computation Time: {result.computation_time:.3f}s
Edits Applied: {result.metrics.get('edits_applied', 0)}

Demand Metrics:
- Peak Demand: {result.metrics.get('peak_demand', 0):.2f}
- P95 Demand: {result.metrics.get('p95_demand', 0):.2f}
- Mean Demand: {result.metrics.get('mean_demand', 0):.2f}
- Variance: {result.metrics.get('variance', 0):.2f}

Schedule Changes:
- Sections Moved: {result.metrics.get('sections_moved', 0)}
- Percent Moved: {result.metrics.get('percent_moved', 0):.1f}%
- Total Shift Minutes: {result.metrics.get('total_shift_minutes', 0)}
- Average Shift: {result.metrics.get('avg_shift_minutes', 0):.1f} minutes

Validation Results:
"""
        
        for validation in result.validation_results:
            status = "✓" if validation.is_valid else "✗"
            summary += f"- {status} Section {validation.section_id}: {validation.original_start_min} → {validation.proposed_start_min}"
            if validation.error_message:
                summary += f" (Error: {validation.error_message})"
            elif validation.warning_message:
                summary += f" (Warning: {validation.warning_message})"
            summary += "\n"
        
        summary += f"\nMessage: {result.message}"
        return summary
    
    def get_available_slots_for_section(self, section_id: str) -> List[int]:
        """
        Get available time slots for a specific section.
        
        Args:
            section_id: Section identifier
            
        Returns:
            List of available start_minutes
        """
        if section_id not in self.ivp.impact_vectors:
            return []
        
        return list(self.ivp.impact_vectors[section_id].keys())
    
    def get_section_info(self, section_id: str) -> Optional[Dict]:
        """
        Get information about a specific section.
        
        Args:
            section_id: Section identifier
            
        Returns:
            Dictionary with section information or None if not found
        """
        if section_id not in self.sections_catalog.sections_dict:
            return None
        
        section = self.sections_catalog.sections_dict[section_id]
        return {
            'section_id': section_id,
            'current_start_minutes': section.start_minutes,
            'current_end_minutes': section.end_minutes,
            'student_count': section.student_count,
            'available_slots': self.get_available_slots_for_section(section_id)
        }


def main():
    """Test the ScenarioPreviewService with actual data."""
    logger.info("Testing ScenarioPreviewService...")
    
    # Import required modules
    from dataloader import DataLoader
    from time_grid_service import TimeGridService
    from sections_catalog import SectionsCatalog
    from availability_service import AvailabilityService
    from meal_propensity_service import MealPropensityService
    from feature_builder import FeatureBuilder
    from ml_model_trainer import MLModelTrainer
    from demand_service import DemandService
    from impact_vector_precomputer import ImpactVectorPrecomputer
    
    try:
        # Load data
        logger.info("Loading data...")
        dataloader = DataLoader()
        dataloader.load_all_data()
        
        # Initialize services
        time_grid = TimeGridService()
        sections_catalog = SectionsCatalog(time_grid)
        sections_catalog.build_catalog(dataloader.class_enrollments)
        
        availability_service = AvailabilityService(time_grid, sections_catalog)
        all_student_ids = dataloader.students_data['student_id'].unique().tolist()
        availability_service.compute_availability(all_student_ids)
        
        meal_propensity_service = MealPropensityService()
        meal_propensity_service.compute_meal_propensities(dataloader.swipes_data, all_student_ids)
        
        feature_builder = FeatureBuilder(time_grid, meal_propensity_service, availability_service, sections_catalog)
        ml_trainer = MLModelTrainer(feature_builder, time_grid)
        
        # Train model
        logger.info("Training ML model...")
        training_df = ml_trainer.prepare_training_data(all_student_ids)
        ml_trainer.train_model(training_df)
        
        # Initialize demand service
        demand_service = DemandService(ml_trainer, time_grid, availability_service, sections_catalog)
        
        # Initialize impact vector precomputer
        ivp = ImpactVectorPrecomputer(demand_service, time_grid, sections_catalog, availability_service, meal_propensity_service)
        ivp.precompute_impact_vectors(all_student_ids)
        
        # Initialize preview service
        preview_service = ScenarioPreviewService(ivp, demand_service, time_grid, sections_catalog)
        
        # Test 1: Small number of edits (should use impact vectors)
        logger.info("Testing with small number of edits...")
        small_edits = {}
        section_ids = list(sections_catalog.sections_dict.keys())[:3]
        for i, section_id in enumerate(section_ids):
            available_slots = preview_service.get_available_slots_for_section(section_id)
            if available_slots:
                small_edits[section_id] = available_slots[0]  # Use first available slot
        
        if small_edits:
            result1 = preview_service.preview_scenario(small_edits, all_student_ids)
            print(preview_service.get_preview_summary(result1))
        
        # Test 2: Large number of edits (should use full recompute)
        logger.info("Testing with large number of edits...")
        large_edits = {}
        section_ids = list(sections_catalog.sections_dict.keys())[:15]
        for i, section_id in enumerate(section_ids):
            available_slots = preview_service.get_available_slots_for_section(section_id)
            if available_slots:
                large_edits[section_id] = available_slots[0]  # Use first available slot
        
        if large_edits:
            result2 = preview_service.preview_scenario(large_edits, all_student_ids)
            print(preview_service.get_preview_summary(result2))
        
        # Test 3: Invalid edits
        logger.info("Testing with invalid edits...")
        invalid_edits = {
            'nonexistent_section': 500,
            'invalid_time': 1000
        }
        result3 = preview_service.preview_scenario(invalid_edits, all_student_ids)
        print(preview_service.get_preview_summary(result3))
        
        # Test 4: Get section info
        logger.info("Testing section info retrieval...")
        if section_ids:
            section_info = preview_service.get_section_info(section_ids[0])
            if section_info:
                print(f"Section info for {section_ids[0]}:")
                for key, value in section_info.items():
                    print(f"  {key}: {value}")
        
        logger.info("ScenarioPreviewService test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
