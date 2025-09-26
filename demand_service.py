"""
DemandService module for dining demand optimization system.

This module computes demand curves and metrics for any schedule scenario,
using the trained ML model to predict swipe probabilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

from ml_model_trainer import MLModelTrainer
from time_grid_service import TimeGridService, MealPeriod
from availability_service import AvailabilityService
from sections_catalog import SectionsCatalog

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DemandMetrics:
    """Represents demand metrics for a time period."""
    total_demand: float
    peak_demand: float
    p95_demand: float
    variance: float
    mean_demand: float
    std_demand: float
    
    # Per-meal breakdown
    breakfast_demand: float
    lunch_demand: float
    dinner_demand: float
    none_demand: float
    
    # Peak information
    peak_bin: int
    peak_time: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_demand': self.total_demand,
            'peak_demand': self.peak_demand,
            'p95_demand': self.p95_demand,
            'variance': self.variance,
            'mean_demand': self.mean_demand,
            'std_demand': self.std_demand,
            'breakfast_demand': self.breakfast_demand,
            'lunch_demand': self.lunch_demand,
            'dinner_demand': self.dinner_demand,
            'none_demand': self.none_demand,
            'peak_bin': self.peak_bin,
            'peak_time': self.peak_time
        }


@dataclass
class BinDemand:
    """Represents demand for a single time bin."""
    bin_index: int
    time_range: str
    meal_period: str
    expected_demand: float
    student_count: int
    free_students: int
    in_class_students: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'bin_index': self.bin_index,
            'time_range': self.time_range,
            'meal_period': self.meal_period,
            'expected_demand': self.expected_demand,
            'student_count': self.student_count,
            'free_students': self.free_students,
            'in_class_students': self.in_class_students
        }


class DemandService:
    """Service for computing demand curves and metrics for schedule scenarios."""
    
    def __init__(self, 
                 ml_trainer: MLModelTrainer,
                 time_grid_service: TimeGridService,
                 availability_service: AvailabilityService,
                 sections_catalog: SectionsCatalog):
        """
        Initialize DemandService.
        
        Args:
            ml_trainer: Trained MLModelTrainer instance
            time_grid_service: TimeGridService instance
            availability_service: AvailabilityService instance
            sections_catalog: SectionsCatalog instance
        """
        self.ml_trainer = ml_trainer
        self.time_grid = time_grid_service
        self.availability = availability_service
        self.sections_catalog = sections_catalog
        
        # Cache for demand calculations
        self._demand_cache = {}
        self._current_schedule_hash = None
        
        logger.info("Initialized DemandService")
    
    def compute_demand_for_schedule(self, 
                                  student_ids: List[str],
                                  schedule_changes: Optional[Dict] = None) -> Tuple[List[BinDemand], DemandMetrics]:
        """
        Compute demand curve for a given schedule scenario.
        
        Args:
            student_ids: List of student IDs to include
            schedule_changes: Optional dict of schedule changes to apply
            
        Returns:
            Tuple of (bin_demands, overall_metrics)
        """
        logger.info(f"Computing demand for {len(student_ids)} students...")
        
        # Apply schedule changes if provided
        if schedule_changes:
            self._apply_schedule_changes(schedule_changes)
        
        # Compute demand for each bin
        bin_demands = []
        total_demand = 0.0
        
        for bin_idx in range(len(self.time_grid.time_bins)):
            bin_obj = self.time_grid.get_bin(bin_idx)
            bin_demand = self._compute_bin_demand(student_ids, bin_idx, bin_obj)
            bin_demands.append(bin_demand)
            total_demand += bin_demand.expected_demand
        
        # Compute overall metrics
        metrics = self._compute_demand_metrics(bin_demands, total_demand)
        
        logger.info(f"Computed demand - Total: {total_demand:.1f}, Peak: {metrics.peak_demand:.1f}")
        
        return bin_demands, metrics
    
    def _apply_schedule_changes(self, schedule_changes: Dict) -> None:
        """
        Apply schedule changes to the availability service.
        
        Args:
            schedule_changes: Dict with section_id -> new_time_info
        """
        logger.info(f"Applying {len(schedule_changes)} schedule changes...")
        
        # Update sections catalog with new times
        for section_id, new_time_info in schedule_changes.items():
            # Find the section in the DataFrame
            section_mask = self.sections_catalog.sections_df['section_id'] == section_id
            if section_mask.any():
                # Update section times in DataFrame (only start_min and end_min exist)
                self.sections_catalog.sections_df.loc[section_mask, 'start_min'] = new_time_info.get('start_minutes', self.sections_catalog.sections_df.loc[section_mask, 'start_min'].iloc[0])
                self.sections_catalog.sections_df.loc[section_mask, 'end_min'] = new_time_info.get('end_minutes', self.sections_catalog.sections_df.loc[section_mask, 'end_min'].iloc[0])
                
                # Update the Section object in sections_dict
                if section_id in self.sections_catalog.sections_dict:
                    section = self.sections_catalog.sections_dict[section_id]
                    section.start_minutes = new_time_info.get('start_minutes', section.start_minutes)
                    section.end_minutes = new_time_info.get('end_minutes', section.end_minutes)
                    
                    # Recompute time bins covered by this section
                    new_time_bins = self.time_grid.get_bins_for_time_range(section.start_minutes, section.end_minutes)
                    section.time_bins = [tb.index for tb in new_time_bins]
        
        # Recompute availability for affected students
        affected_students = set()
        for section_id in schedule_changes.keys():
            students_in_section = self.sections_catalog.get_students_in_section(section_id)
            affected_students.update(students_in_section)
        
        if affected_students:
            logger.info(f"Recomputing availability for {len(affected_students)} affected students...")
            self.availability.compute_availability(list(affected_students))
    
    def _compute_bin_demand(self, 
                           student_ids: List[str], 
                           bin_idx: int, 
                           bin_obj) -> BinDemand:
        """Compute demand for a single time bin."""
        expected_demand = 0.0
        free_students = 0
        in_class_students = 0
        
        for student_id in student_ids:
            # Check if student is free in this bin
            is_free = self.availability.is_free_in_bin(student_id, bin_idx)
            
            if is_free:
                free_students += 1
                # Get probability from ML model
                prob = self.ml_trainer.predict_swipe_probability(student_id, bin_idx)
                expected_demand += prob
            else:
                in_class_students += 1
        
        return BinDemand(
            bin_index=bin_idx,
            time_range=f"{self._minutes_to_time(bin_obj.start_minutes)}-{self._minutes_to_time(bin_obj.end_minutes)}",
            meal_period=bin_obj.meal_period.value,
            expected_demand=expected_demand,
            student_count=len(student_ids),
            free_students=free_students,
            in_class_students=in_class_students
        )
    
    def _compute_demand_metrics(self, 
                               bin_demands: List[BinDemand], 
                               total_demand: float) -> DemandMetrics:
        """Compute overall demand metrics."""
        demands = [bd.expected_demand for bd in bin_demands]
        
        # Basic statistics
        mean_demand = np.mean(demands)
        std_demand = np.std(demands)
        peak_demand = np.max(demands)
        peak_bin = np.argmax(demands)
        peak_time = bin_demands[peak_bin].time_range
        
        # P95 demand
        p95_demand = np.percentile(demands, 95)
        
        # Variance
        variance = np.var(demands)
        
        # Per-meal breakdown
        meal_demands = defaultdict(float)
        for bd in bin_demands:
            meal_demands[bd.meal_period] += bd.expected_demand
        
        return DemandMetrics(
            total_demand=total_demand,
            peak_demand=peak_demand,
            p95_demand=p95_demand,
            variance=variance,
            mean_demand=mean_demand,
            std_demand=std_demand,
            breakfast_demand=meal_demands['Breakfast'],
            lunch_demand=meal_demands['Lunch'],
            dinner_demand=meal_demands['Dinner'],
            none_demand=meal_demands['None'],
            peak_bin=peak_bin,
            peak_time=peak_time
        )
    
    def _minutes_to_time(self, minutes: int) -> str:
        """Convert minutes since midnight to HH:MM format."""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def compare_schedules(self, 
                         student_ids: List[str],
                         baseline_schedule: Optional[Dict] = None,
                         proposed_schedule: Optional[Dict] = None) -> Dict:
        """
        Compare demand between two schedule scenarios.
        
        Args:
            student_ids: List of student IDs
            baseline_schedule: Optional baseline schedule changes
            proposed_schedule: Optional proposed schedule changes
            
        Returns:
            Dictionary with comparison metrics
        """
        logger.info("Comparing schedule scenarios...")
        
        # Compute baseline demand
        baseline_demands, baseline_metrics = self.compute_demand_for_schedule(
            student_ids, baseline_schedule
        )
        
        # Compute proposed demand
        proposed_demands, proposed_metrics = self.compute_demand_for_schedule(
            student_ids, proposed_schedule
        )
        
        # Compute differences
        demand_diff = proposed_metrics.total_demand - baseline_metrics.total_demand
        peak_diff = proposed_metrics.peak_demand - baseline_metrics.peak_demand
        p95_diff = proposed_metrics.p95_demand - baseline_metrics.p95_demand
        
        # Per-meal differences
        meal_diffs = {
            'breakfast': proposed_metrics.breakfast_demand - baseline_metrics.breakfast_demand,
            'lunch': proposed_metrics.lunch_demand - baseline_metrics.lunch_demand,
            'dinner': proposed_metrics.dinner_demand - baseline_metrics.dinner_demand,
            'none': proposed_metrics.none_demand - baseline_metrics.none_demand
        }
        
        # Peak load reduction
        peak_reduction = baseline_metrics.peak_demand - proposed_metrics.peak_demand
        peak_reduction_pct = (peak_reduction / baseline_metrics.peak_demand) * 100 if baseline_metrics.peak_demand > 0 else 0
        
        comparison = {
            'baseline_metrics': baseline_metrics.to_dict(),
            'proposed_metrics': proposed_metrics.to_dict(),
            'total_demand_change': demand_diff,
            'peak_demand_change': peak_diff,
            'p95_demand_change': p95_diff,
            'peak_reduction': peak_reduction,
            'peak_reduction_pct': peak_reduction_pct,
            'meal_demand_changes': meal_diffs,
            'improvement': peak_reduction > 0
        }
        
        logger.info(f"Schedule comparison - Peak reduction: {peak_reduction:.1f} ({peak_reduction_pct:.1f}%)")
        
        return comparison
    
    def get_demand_summary(self, bin_demands: List[BinDemand]) -> pd.DataFrame:
        """
        Get demand summary as DataFrame.
        
        Args:
            bin_demands: List of BinDemand objects
            
        Returns:
            DataFrame with demand summary
        """
        data = []
        for bd in bin_demands:
            data.append({
                'bin_index': bd.bin_index,
                'time_range': bd.time_range,
                'meal_period': bd.meal_period,
                'expected_demand': bd.expected_demand,
                'free_students': bd.free_students,
                'in_class_students': bd.in_class_students,
                'utilization_pct': (bd.in_class_students / bd.student_count) * 100 if bd.student_count > 0 else 0
            })
        
        return pd.DataFrame(data)
    
    def get_peak_analysis(self, bin_demands: List[BinDemand], top_n: int = 5) -> Dict:
        """
        Analyze peak demand periods.
        
        Args:
            bin_demands: List of BinDemand objects
            top_n: Number of top peaks to return
            
        Returns:
            Dictionary with peak analysis
        """
        # Sort by demand (descending)
        sorted_demands = sorted(bin_demands, key=lambda x: x.expected_demand, reverse=True)
        
        peaks = []
        for i, bd in enumerate(sorted_demands[:top_n]):
            peaks.append({
                'rank': i + 1,
                'bin_index': bd.bin_index,
                'time_range': bd.time_range,
                'meal_period': bd.meal_period,
                'expected_demand': bd.expected_demand,
                'free_students': bd.free_students,
                'in_class_students': bd.in_class_students
            })
        
        return {
            'top_peaks': peaks,
            'total_peaks': len([bd for bd in bin_demands if bd.expected_demand > 0])
        }
    
    def export_demand_data(self, 
                          bin_demands: List[BinDemand], 
                          metrics: DemandMetrics,
                          filepath: str) -> None:
        """
        Export demand data to CSV.
        
        Args:
            bin_demands: List of BinDemand objects
            metrics: DemandMetrics object
            filepath: Output file path
        """
        # Create summary DataFrame
        summary_df = self.get_demand_summary(bin_demands)
        
        # Add metrics as additional rows
        metrics_data = {
            'bin_index': 'METRICS',
            'time_range': 'OVERALL',
            'meal_period': 'ALL',
            'expected_demand': metrics.total_demand,
            'free_students': 'N/A',
            'in_class_students': 'N/A',
            'utilization_pct': 'N/A'
        }
        
        # Append metrics
        summary_df = pd.concat([summary_df, pd.DataFrame([metrics_data])], ignore_index=True)
        
        # Export
        summary_df.to_csv(filepath, index=False)
        logger.info(f"Exported demand data to {filepath}")


def main():
    """Example usage of DemandService."""
    from dataloader import DataLoader
    from time_grid_service import TimeGridService
    from meal_propensity_service import MealPropensityService
    from availability_service import AvailabilityService
    from sections_catalog import SectionsCatalog
    from feature_builder import FeatureBuilder
    from ml_model_trainer import MLModelTrainer, ModelConfig
    
    # Load all data
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Create all services
    time_grid = TimeGridService()
    sections_catalog = SectionsCatalog(time_grid)
    sections_catalog.build_catalog(data['class_enrollments'])
    
    # Get student IDs
    student_ids = list(set(data['section_enrollments']['student_id'].tolist()))
    
    # Create services
    meal_propensity = MealPropensityService()
    meal_propensity.compute_meal_propensities(data['swipes_data'], student_ids)
    
    availability = AvailabilityService(time_grid, sections_catalog)
    availability.compute_availability(student_ids)
    
    feature_builder = FeatureBuilder(time_grid, meal_propensity, availability, sections_catalog)
    
    # Train ML model
    config = ModelConfig(model_type='gradient_boosting', n_estimators=50)
    ml_trainer = MLModelTrainer(feature_builder, time_grid, config)
    training_df = ml_trainer.prepare_training_data(student_ids)
    ml_trainer.train_model(training_df)
    
    # Create demand service
    demand_service = DemandService(ml_trainer, time_grid, availability, sections_catalog)
    
    # Compute demand for current schedule
    bin_demands, metrics = demand_service.compute_demand_for_schedule(student_ids)
    
    print("=== Demand Service Demo ===")
    print(f"Total Expected Demand: {metrics.total_demand:.1f}")
    print(f"Peak Demand: {metrics.peak_demand:.1f} at {metrics.peak_time}")
    print(f"P95 Demand: {metrics.p95_demand:.1f}")
    print(f"Mean Demand: {metrics.mean_demand:.1f} ± {metrics.std_demand:.1f}")
    print()
    
    print("Per-Meal Breakdown:")
    print(f"  Breakfast: {metrics.breakfast_demand:.1f}")
    print(f"  Lunch: {metrics.lunch_demand:.1f}")
    print(f"  Dinner: {metrics.dinner_demand:.1f}")
    print(f"  None: {metrics.none_demand:.1f}")
    print()
    
    # Show top 5 demand peaks
    peak_analysis = demand_service.get_peak_analysis(bin_demands, top_n=5)
    print("Top 5 Demand Peaks:")
    for peak in peak_analysis['top_peaks']:
        print(f"  {peak['rank']}. {peak['time_range']} ({peak['meal_period']}): {peak['expected_demand']:.1f}")
    print()
    
    # Show demand summary
    summary_df = demand_service.get_demand_summary(bin_demands)
    print("Sample Demand Summary (first 10 bins):")
    print(summary_df.head(10))
    print()
    
    # Test schedule comparison (simulate moving a class)
    print("Testing Schedule Comparison...")
    
    # Get a section to modify
    section_id = sections_catalog.sections_df.iloc[0]['section_id']
    original_section = sections_catalog.sections_df.iloc[0]
    
    # Create a schedule change (move class 1 hour later)
    # Convert minutes to time strings for display
    start_minutes = original_section['start_min']
    end_minutes = original_section['end_min']
    
    start_hour = start_minutes // 60
    start_min = start_minutes % 60
    new_start_time = f"{(start_hour + 1) % 24:02d}:{start_min:02d}"
    
    end_hour = end_minutes // 60
    end_min = end_minutes % 60
    new_end_time = f"{(end_hour + 1) % 24:02d}:{end_min:02d}"
    
    schedule_changes = {
        section_id: {
            'start_time': new_start_time,
            'end_time': new_end_time,
            'start_minutes': start_minutes + 60,
            'end_minutes': end_minutes + 60
        }
    }
    
    # Compare schedules
    comparison = demand_service.compare_schedules(
        student_ids, 
        baseline_schedule=None,  # Current schedule
        proposed_schedule=schedule_changes
    )
    
    print(f"Schedule Comparison Results:")
    print(f"  Peak Reduction: {comparison['peak_reduction']:.1f} ({comparison['peak_reduction_pct']:.1f}%)")
    print(f"  Total Demand Change: {comparison['total_demand_change']:.1f}")
    print(f"  Improvement: {comparison['improvement']}")


if __name__ == "__main__":
    main()
