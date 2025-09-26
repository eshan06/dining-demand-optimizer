"""
AvailabilityService module for dining demand optimization system.

This module efficiently tracks student availability across time bins,
providing fast lookups for demand modeling and optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from time_grid_service import TimeGridService
from sections_catalog import SectionsCatalog

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StudentAvailability:
    """Represents availability data for a single student."""
    student_id: str
    total_bins: int
    class_bins: Set[int]  # Set of bin indices where student has class
    free_bins: Set[int]   # Set of bin indices where student is free
    
    def has_class_in_bin(self, bin_index: int) -> bool:
        """Check if student has class in a specific bin."""
        return bin_index in self.class_bins
    
    def is_free_in_bin(self, bin_index: int) -> bool:
        """Check if student is free in a specific bin."""
        return bin_index in self.free_bins
    
    def get_availability_percentage(self) -> float:
        """Get percentage of time bins where student is free."""
        if self.total_bins == 0:
            return 0.0
        return len(self.free_bins) / self.total_bins * 100


class AvailabilityService:
    """Service for tracking and querying student availability across time bins."""
    
    def __init__(self, time_grid_service: TimeGridService, sections_catalog: SectionsCatalog):
        """
        Initialize AvailabilityService.
        
        Args:
            time_grid_service: TimeGridService instance
            sections_catalog: SectionsCatalog instance
        """
        self.time_grid = time_grid_service
        self.sections_catalog = sections_catalog
        self.availability_data = {}  # student_id -> StudentAvailability
        self.bin_utilization = defaultdict(set)  # bin_index -> set of student_ids with class
        self.free_students_by_bin = defaultdict(set)  # bin_index -> set of free student_ids
        
        logger.info("Initialized AvailabilityService")
    
    def compute_availability(self, student_ids: List[str]) -> None:
        """
        Compute availability for all students.
        
        Args:
            student_ids: List of all student IDs to process
        """
        logger.info(f"Computing availability for {len(student_ids)} students...")
        
        # Clear existing data
        self.availability_data.clear()
        self.bin_utilization.clear()
        self.free_students_by_bin.clear()
        
        total_bins = len(self.time_grid.time_bins)
        
        for student_id in student_ids:
            # Get all sections this student is enrolled in
            student_sections = self.sections_catalog.get_sections_for_student(student_id)
            
            # Find all bins where student has class
            class_bins = set()
            for section in student_sections:
                class_bins.update(section.time_bins)
            
            # Calculate free bins (all bins minus class bins)
            free_bins = set(range(total_bins)) - class_bins
            
            # Create availability record
            availability = StudentAvailability(
                student_id=student_id,
                total_bins=total_bins,
                class_bins=class_bins,
                free_bins=free_bins
            )
            
            self.availability_data[student_id] = availability
            
            # Update bin utilization tracking
            for bin_idx in class_bins:
                self.bin_utilization[bin_idx].add(student_id)
            
            for bin_idx in free_bins:
                self.free_students_by_bin[bin_idx].add(student_id)
        
        logger.info(f"Computed availability for {len(self.availability_data)} students")
        self._log_availability_summary()
    
    def has_class_in_bin(self, student_id: str, bin_index: int) -> bool:
        """
        Check if a student has class in a specific time bin.
        
        Args:
            student_id: Student identifier
            bin_index: Time bin index
            
        Returns:
            True if student has class in this bin
        """
        if student_id not in self.availability_data:
            return False
        
        return self.availability_data[student_id].has_class_in_bin(bin_index)
    
    def is_free_in_bin(self, student_id: str, bin_index: int) -> bool:
        """
        Check if a student is free in a specific time bin.
        
        Args:
            student_id: Student identifier
            bin_index: Time bin index
            
        Returns:
            True if student is free in this bin
        """
        if student_id not in self.availability_data:
            return True  # Assume free if not in system
        
        return self.availability_data[student_id].is_free_in_bin(bin_index)
    
    def get_students_in_class_at_bin(self, bin_index: int) -> Set[str]:
        """
        Get all students who have class in a specific time bin.
        
        Args:
            bin_index: Time bin index
            
        Returns:
            Set of student IDs with class in this bin
        """
        return self.bin_utilization.get(bin_index, set())
    
    def get_free_students_at_bin(self, bin_index: int) -> Set[str]:
        """
        Get all students who are free in a specific time bin.
        
        Args:
            bin_index: Time bin index
            
        Returns:
            Set of student IDs free in this bin
        """
        return self.free_students_by_bin.get(bin_index, set())
    
    def get_student_availability(self, student_id: str) -> Optional[StudentAvailability]:
        """
        Get full availability data for a student.
        
        Args:
            student_id: Student identifier
            
        Returns:
            StudentAvailability object or None if not found
        """
        return self.availability_data.get(student_id)
    
    def get_availability_in_time_range(self, student_id: str, start_minutes: int, end_minutes: int) -> List[bool]:
        """
        Get availability for a student across a time range.
        
        Args:
            student_id: Student identifier
            start_minutes: Start of time range
            end_minutes: End of time range
            
        Returns:
            List of boolean values (True = free, False = in class) for each bin in range
        """
        if student_id not in self.availability_data:
            return []
        
        # Find bins that overlap with the time range
        overlapping_bins = self.time_grid.get_bins_for_time_range(start_minutes, end_minutes)
        bin_indices = [bin_obj.index for bin_obj in overlapping_bins]
        
        # Get availability for each bin
        availability = []
        for bin_idx in bin_indices:
            is_free = self.is_free_in_bin(student_id, bin_idx)
            availability.append(is_free)
        
        return availability
    
    def get_bin_utilization_stats(self) -> Dict[int, Dict[str, int]]:
        """
        Get utilization statistics for each time bin.
        
        Returns:
            Dictionary mapping bin_index to stats (total_students, in_class, free)
        """
        stats = {}
        total_students = len(self.availability_data)
        
        for bin_idx in range(len(self.time_grid.time_bins)):
            in_class = len(self.get_students_in_class_at_bin(bin_idx))
            free = len(self.get_free_students_at_bin(bin_idx))
            
            stats[bin_idx] = {
                'total_students': total_students,
                'in_class': in_class,
                'free': free,
                'utilization_percentage': (in_class / total_students * 100) if total_students > 0 else 0
            }
        
        return stats
    
    def get_meal_period_availability(self) -> Dict[str, Dict[str, int]]:
        """
        Get availability statistics by meal period.
        
        Returns:
            Dictionary with meal period statistics
        """
        from time_grid_service import MealPeriod
        
        meal_stats = {}
        
        for period in MealPeriod:
            period_bins = self.time_grid.get_bins_in_meal_period(period)
            bin_indices = [bin_obj.index for bin_obj in period_bins]
            
            total_in_class = 0
            total_free = 0
            
            for bin_idx in bin_indices:
                in_class = len(self.get_students_in_class_at_bin(bin_idx))
                free = len(self.get_free_students_at_bin(bin_idx))
                total_in_class += in_class
                total_free += free
            
            meal_stats[period.value] = {
                'bins_count': len(bin_indices),
                'total_in_class': total_in_class,
                'total_free': total_free,
                'avg_in_class_per_bin': total_in_class / len(bin_indices) if bin_indices else 0,
                'avg_free_per_bin': total_free / len(bin_indices) if bin_indices else 0
            }
        
        return meal_stats
    
    def get_availability_summary(self) -> Dict:
        """
        Get comprehensive availability summary.
        
        Returns:
            Dictionary with availability statistics
        """
        if not self.availability_data:
            return {}
        
        total_students = len(self.availability_data)
        total_bins = len(self.time_grid.time_bins)
        
        # Calculate average availability across all students
        total_free_bins = sum(len(avail.free_bins) for avail in self.availability_data.values())
        total_class_bins = sum(len(avail.class_bins) for avail in self.availability_data.values())
        
        # Find peak utilization bins
        bin_utilization = self.get_bin_utilization_stats()
        max_utilization = max(stats['in_class'] for stats in bin_utilization.values())
        peak_bins = [bin_idx for bin_idx, stats in bin_utilization.items() 
                    if stats['in_class'] == max_utilization]
        
        return {
            'total_students': total_students,
            'total_bins': total_bins,
            'total_free_bins': total_free_bins,
            'total_class_bins': total_class_bins,
            'avg_availability_percentage': (total_free_bins / (total_students * total_bins) * 100) if total_students > 0 else 0,
            'max_utilization': max_utilization,
            'peak_utilization_bins': peak_bins,
            'bins_with_classes': len([bin_idx for bin_idx, stats in bin_utilization.items() if stats['in_class'] > 0])
        }
    
    def _log_availability_summary(self) -> None:
        """Log summary of computed availability data."""
        if not self.availability_data:
            return
        
        summary = self.get_availability_summary()
        
        logger.info("=== Availability Summary ===")
        logger.info(f"Total students: {summary['total_students']}")
        logger.info(f"Total time bins: {summary['total_bins']}")
        logger.info(f"Average availability: {summary['avg_availability_percentage']:.1f}%")
        logger.info(f"Peak utilization: {summary['max_utilization']} students in class")
        logger.info(f"Bins with classes: {summary['bins_with_classes']}")
        
        # Show meal period breakdown
        meal_stats = self.get_meal_period_availability()
        logger.info("Meal period availability:")
        for period, stats in meal_stats.items():
            logger.info(f"  {period}: {stats['avg_in_class_per_bin']:.1f} avg in class per bin")
    
    def export_availability_table(self) -> pd.DataFrame:
        """
        Export availability data as a DataFrame.
        
        Returns:
            DataFrame with columns: student_id, bin_index, has_class_in_bin
        """
        data = []
        
        for student_id, availability in self.availability_data.items():
            for bin_idx in range(len(self.time_grid.time_bins)):
                has_class = availability.has_class_in_bin(bin_idx)
                data.append({
                    'student_id': student_id,
                    'bin_index': bin_idx,
                    'has_class_in_bin': has_class
                })
        
        return pd.DataFrame(data)
    
    def __str__(self) -> str:
        """String representation of the service."""
        if not self.availability_data:
            return "AvailabilityService: No data loaded"
        
        summary = self.get_availability_summary()
        return (f"AvailabilityService: {summary['total_students']} students, "
                f"{summary['avg_availability_percentage']:.1f}% avg availability")


def main():
    """Example usage of AvailabilityService."""
    from dataloader import DataLoader
    
    # Load data
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Create services
    time_grid = TimeGridService()
    sections_catalog = SectionsCatalog(time_grid)
    sections_catalog.build_catalog(data['class_enrollments'])
    
    # Get all student IDs from the data
    all_student_ids = list(set(data['section_enrollments']['student_id'].tolist()))
    
    # Create availability service
    availability_service = AvailabilityService(time_grid, sections_catalog)
    
    # Compute availability
    availability_service.compute_availability(all_student_ids)
    
    print("=== Availability Service Demo ===")
    print(f"Service: {availability_service}")
    print()
    
    # Show summary
    summary = availability_service.get_availability_summary()
    print("Availability Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    
    # Test individual queries
    print("Individual Query Tests:")
    sample_student = all_student_ids[0]
    print(f"Student {sample_student}:")
    
    # Check availability in different bins
    test_bins = [0, 10, 20, 30, 40, 50]  # Various time bins
    for bin_idx in test_bins:
        has_class = availability_service.has_class_in_bin(sample_student, bin_idx)
        bin_obj = time_grid.get_bin(bin_idx)
        print(f"  Bin {bin_idx} ({bin_obj}): {'In Class' if has_class else 'Free'}")
    print()
    
    # Show bin utilization
    print("Bin Utilization (first 10 bins with classes):")
    bin_stats = availability_service.get_bin_utilization_stats()
    bins_with_classes = [(bin_idx, stats) for bin_idx, stats in bin_stats.items() if stats['in_class'] > 0]
    bins_with_classes.sort(key=lambda x: x[1]['in_class'], reverse=True)
    
    for bin_idx, stats in bins_with_classes[:10]:
        bin_obj = time_grid.get_bin(bin_idx)
        print(f"  Bin {bin_idx} ({bin_obj}): {stats['in_class']} students in class")
    print()
    
    # Show meal period statistics
    print("Meal Period Availability:")
    meal_stats = availability_service.get_meal_period_availability()
    for period, stats in meal_stats.items():
        print(f"  {period}: {stats['avg_in_class_per_bin']:.1f} avg students in class per bin")
    print()
    
    # Test time range queries
    print("Time Range Query Test:")
    lunch_availability = availability_service.get_availability_in_time_range(
        sample_student, 720, 780  # 12:00-13:00
    )
    print(f"Student {sample_student} availability during lunch (12:00-13:00):")
    print(f"  Free bins: {sum(lunch_availability)} out of {len(lunch_availability)}")
    
    # Show some free students at a specific bin
    print(f"\nFree Students at Bin 20:")
    free_students = availability_service.get_free_students_at_bin(20)
    print(f"  {len(free_students)} students free: {list(free_students)[:5]}...")


if __name__ == "__main__":
    main()
