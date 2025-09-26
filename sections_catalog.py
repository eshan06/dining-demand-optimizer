"""
SectionsCatalog module for dining demand optimization system.

This module processes class enrollment data into a clean, queryable format
with time bin alignment for optimization and analysis.
"""

import pandas as pd
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from time_grid_service import TimeGridService, TimeBin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a class section with its time and enrollment information."""
    section_id: str
    start_minutes: int
    end_minutes: int
    student_count: int
    time_bins: List[int]  # List of bin indices this section covers
    
    def __str__(self) -> str:
        start_hour = self.start_minutes // 60
        start_min = self.start_minutes % 60
        end_hour = self.end_minutes // 60
        end_min = self.end_minutes % 60
        return (f"Section {self.section_id}: {start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d} "
                f"({self.student_count} students, {len(self.time_bins)} bins)")


class SectionsCatalog:
    """Catalog of class sections with enrollment and time bin information."""
    
    def __init__(self, time_grid_service: TimeGridService):
        """
        Initialize SectionsCatalog.
        
        Args:
            time_grid_service: TimeGridService instance for time bin calculations
        """
        self.time_grid = time_grid_service
        self.sections_df = None
        self.section_enrollment_df = None
        self.sections_dict = {}  # section_id -> Section object
        
        logger.info("Initialized SectionsCatalog")
    
    def build_catalog(self, class_enrollments_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Build the sections catalog from class enrollment data.
        
        Args:
            class_enrollments_df: DataFrame with class enrollment data
            
        Returns:
            Dictionary containing sections and section_enrollment DataFrames
        """
        logger.info("Building sections catalog...")
        
        # Process sections table
        self.sections_df = self._build_sections_table(class_enrollments_df)
        
        # Process section enrollment table
        self.section_enrollment_df = self._build_section_enrollment_table(class_enrollments_df)
        
        # Build section objects with time bin information
        self._build_sections_dict()
        
        logger.info(f"Built catalog with {len(self.sections_df)} sections and {len(self.section_enrollment_df)} enrollments")
        
        return {
            'sections': self.sections_df,
            'section_enrollment': self.section_enrollment_df
        }
    
    def _build_sections_table(self, class_enrollments_df: pd.DataFrame) -> pd.DataFrame:
        """Build the sections table with section_id, start_min, end_min."""
        sections_data = []
        
        for idx, row in class_enrollments_df.iterrows():
            # Generate or use existing section_id
            section_id = self._generate_section_id(row, idx)
            
            # Convert times to minutes
            start_minutes = self._time_to_minutes(row['start_time'])
            end_minutes = self._time_to_minutes(row['end_time'])
            
            # Count students in this section
            student_count = self._count_students_in_section(row['enrollment'])
            
            sections_data.append({
                'section_id': section_id,
                'start_min': start_minutes,
                'end_min': end_minutes,
                'student_count': student_count,
                'course_code': row.get('course_code', ''),
                'course_name': row.get('course_name', ''),
                'days_of_week': row.get('days_of_week', '')
            })
        
        return pd.DataFrame(sections_data)
    
    def _build_section_enrollment_table(self, class_enrollments_df: pd.DataFrame) -> pd.DataFrame:
        """Build the section enrollment table by parsing and exploding enrollment strings."""
        enrollment_data = []
        
        for idx, row in class_enrollments_df.iterrows():
            section_id = self._generate_section_id(row, idx)
            enrollment_str = row['enrollment']
            
            # Parse enrollment string
            student_list = self._parse_enrollment_string(enrollment_str)
            
            # Create enrollment records
            for student_id in student_list:
                enrollment_data.append({
                    'section_id': section_id,
                    'student_id': student_id
                })
        
        return pd.DataFrame(enrollment_data)
    
    def _generate_section_id(self, row: pd.Series, row_index: int) -> str:
        """
        Generate a stable section_id.
        
        Args:
            row: Row from class_enrollments_df
            row_index: Index of the row
            
        Returns:
            Stable section_id string
        """
        # If section_id already exists, use it
        if 'section_id' in row and pd.notna(row['section_id']):
            return str(row['section_id'])
        
        # Generate stable ID based on time + roster
        time_info = f"{row['start_time']}_{row['end_time']}"
        roster_info = str(row['enrollment'])[:50]  # First 50 chars of enrollment
        
        # Create hash for stability
        hash_input = f"{time_info}_{roster_info}".encode('utf-8')
        hash_obj = hashlib.md5(hash_input)
        hash_suffix = hash_obj.hexdigest()[:8]
        
        return f"SECT_{row_index:03d}_{hash_suffix}"
    
    def _parse_enrollment_string(self, enrollment_str: str) -> List[str]:
        """
        Parse enrollment string into list of student IDs.
        
        Args:
            enrollment_str: String like "[STU001,STU002,STU003]"
            
        Returns:
            List of student IDs
        """
        try:
            # Remove brackets and split by comma
            clean_str = enrollment_str.strip('[]')
            student_list = [student.strip() for student in clean_str.split(',')]
            return [s for s in student_list if s]  # Remove empty strings
        except (AttributeError, TypeError):
            logger.warning(f"Failed to parse enrollment string: {enrollment_str}")
            return []
    
    def _count_students_in_section(self, enrollment_str: str) -> int:
        """Count number of students in a section."""
        student_list = self._parse_enrollment_string(enrollment_str)
        return len(student_list)
    
    def _time_to_minutes(self, time_str: str) -> int:
        """Convert time string (HH:MM) to minutes since midnight."""
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to parse time '{time_str}': {e}")
            return 0
    
    def _build_sections_dict(self) -> None:
        """Build dictionary of Section objects with time bin information."""
        if self.sections_df is None:
            return
        
        self.sections_dict = {}
        
        for _, row in self.sections_df.iterrows():
            section_id = row['section_id']
            start_minutes = row['start_min']
            end_minutes = row['end_min']
            student_count = row['student_count']
            
            # Find time bins this section covers
            time_bins = self._get_time_bins_for_section(start_minutes, end_minutes)
            
            section = Section(
                section_id=section_id,
                start_minutes=start_minutes,
                end_minutes=end_minutes,
                student_count=student_count,
                time_bins=time_bins
            )
            
            self.sections_dict[section_id] = section
    
    def _get_time_bins_for_section(self, start_minutes: int, end_minutes: int) -> List[int]:
        """
        Get time bin indices that a section covers.
        
        Args:
            start_minutes: Section start time
            end_minutes: Section end time
            
        Returns:
            List of bin indices
        """
        overlapping_bins = self.time_grid.get_bins_for_time_range(start_minutes, end_minutes)
        return [bin_obj.index for bin_obj in overlapping_bins]
    
    def get_section(self, section_id: str) -> Optional[Section]:
        """
        Get Section object by ID.
        
        Args:
            section_id: Section identifier
            
        Returns:
            Section object or None if not found
        """
        return self.sections_dict.get(section_id)
    
    def get_sections_in_time_range(self, start_minutes: int, end_minutes: int) -> List[Section]:
        """
        Get all sections that overlap with a time range.
        
        Args:
            start_minutes: Start of time range
            end_minutes: End of time range
            
        Returns:
            List of overlapping Section objects
        """
        overlapping_sections = []
        
        for section in self.sections_dict.values():
            # Check if section overlaps with time range
            if (section.start_minutes < end_minutes and section.end_minutes > start_minutes):
                overlapping_sections.append(section)
        
        return overlapping_sections
    
    def get_sections_in_bin(self, bin_index: int) -> List[Section]:
        """
        Get all sections that cover a specific time bin.
        
        Args:
            bin_index: Time bin index
            
        Returns:
            List of Section objects that cover this bin
        """
        sections_in_bin = []
        
        for section in self.sections_dict.values():
            if bin_index in section.time_bins:
                sections_in_bin.append(section)
        
        return sections_in_bin
    
    def get_students_in_section(self, section_id: str) -> List[str]:
        """
        Get list of student IDs enrolled in a section.
        
        Args:
            section_id: Section identifier
            
        Returns:
            List of student IDs
        """
        if self.section_enrollment_df is None:
            return []
        
        section_enrollments = self.section_enrollment_df[
            self.section_enrollment_df['section_id'] == section_id
        ]
        
        return section_enrollments['student_id'].tolist()
    
    def get_sections_for_student(self, student_id: str) -> List[Section]:
        """
        Get all sections a student is enrolled in.
        
        Args:
            student_id: Student identifier
            
        Returns:
            List of Section objects
        """
        if self.section_enrollment_df is None:
            return []
        
        student_sections = self.section_enrollment_df[
            self.section_enrollment_df['student_id'] == student_id
        ]['section_id'].tolist()
        
        return [self.sections_dict[sid] for sid in student_sections if sid in self.sections_dict]
    
    def get_catalog_summary(self) -> Dict:
        """
        Get summary statistics of the catalog.
        
        Returns:
            Dictionary with catalog statistics
        """
        if self.sections_df is None or self.section_enrollment_df is None:
            return {}
        
        # Time range statistics
        time_ranges = self.sections_df[['start_min', 'end_min']].values
        all_times = time_ranges.flatten()
        
        # Bin coverage statistics
        bin_coverage = {}
        for section in self.sections_dict.values():
            for bin_idx in section.time_bins:
                bin_coverage[bin_idx] = bin_coverage.get(bin_idx, 0) + 1
        
        return {
            'total_sections': len(self.sections_df),
            'total_enrollments': len(self.section_enrollment_df),
            'unique_students': self.section_enrollment_df['student_id'].nunique(),
            'avg_students_per_section': self.sections_df['student_count'].mean(),
            'time_range': {
                'earliest_start': int(all_times.min()),
                'latest_end': int(all_times.max())
            },
            'bins_with_sections': len(bin_coverage),
            'max_sections_per_bin': max(bin_coverage.values()) if bin_coverage else 0,
            'avg_sections_per_bin': sum(bin_coverage.values()) / len(bin_coverage) if bin_coverage else 0
        }
    
    def get_time_bin_utilization(self) -> Dict[int, int]:
        """
        Get utilization count for each time bin.
        
        Returns:
            Dictionary mapping bin index to number of sections
        """
        bin_utilization = {}
        
        for section in self.sections_dict.values():
            for bin_idx in section.time_bins:
                bin_utilization[bin_idx] = bin_utilization.get(bin_idx, 0) + 1
        
        return bin_utilization
    
    def __str__(self) -> str:
        """String representation of the catalog."""
        summary = self.get_catalog_summary()
        if not summary:
            return "SectionsCatalog: No data loaded"
        
        return (f"SectionsCatalog: {summary['total_sections']} sections, "
                f"{summary['total_enrollments']} enrollments, "
                f"{summary['unique_students']} unique students")


def main():
    """Example usage of SectionsCatalog."""
    from dataloader import DataLoader
    from time_grid_service import TimeGridService
    
    # Load data
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Create time grid
    time_grid = TimeGridService()
    
    # Create sections catalog
    catalog = SectionsCatalog(time_grid)
    
    # Build catalog
    catalog_data = catalog.build_catalog(data['class_enrollments'])
    
    print("=== Sections Catalog Demo ===")
    print(f"Catalog: {catalog}")
    print()
    
    # Show summary
    summary = catalog.get_catalog_summary()
    print("Catalog Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    
    # Show sample sections
    print("Sample Sections (first 5):")
    sections_df = catalog_data['sections']
    for idx, row in sections_df.head().iterrows():
        section = catalog.get_section(row['section_id'])
        print(f"  {section}")
    print()
    
    # Show sample enrollments
    print("Sample Enrollments (first 10):")
    enrollments_df = catalog_data['section_enrollment']
    print(enrollments_df.head(10))
    print()
    
    # Test time-based queries
    print("Time-based Queries:")
    
    # Sections in lunch time (12:00-13:00)
    lunch_sections = catalog.get_sections_in_time_range(720, 780)
    print(f"Sections during lunch (12:00-13:00): {len(lunch_sections)}")
    
    # Sections in a specific bin
    bin_20 = catalog.get_sections_in_bin(20)  # Should be around 12:00
    print(f"Sections in bin 20: {len(bin_20)}")
    
    # Show time bin utilization
    print("\nTime Bin Utilization (first 10 bins with sections):")
    utilization = catalog.get_time_bin_utilization()
    for bin_idx in sorted(utilization.keys())[:10]:
        bin_obj = time_grid.get_bin(bin_idx)
        print(f"  Bin {bin_idx} ({bin_obj}): {utilization[bin_idx]} sections")
    
    # Test student queries
    print(f"\nStudent Queries:")
    sample_student = enrollments_df['student_id'].iloc[0]
    student_sections = catalog.get_sections_for_student(sample_student)
    print(f"Student {sample_student} is in {len(student_sections)} sections:")
    for section in student_sections[:3]:  # Show first 3
        print(f"  {section}")


if __name__ == "__main__":
    main()
