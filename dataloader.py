"""
DataLoader module for dining demand optimization system.

This module handles CSV data ingestion, validation, and preprocessing for:
- Student dining swipe data
- Class enrollment schedules
- Student meal plan information
"""

import pandas as pd
import logging
import ast
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of CSV data for dining demand optimization."""
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.swipes_data = None
        self.class_enrollments = None
        self.students_data = None
        self.section_enrollments = None
        
        # Valid meal periods
        self.valid_meal_periods = {"Breakfast", "Lunch", "Dinner"}
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and process all CSV files.
        
        Returns:
            Dictionary containing processed DataFrames
        """
        logger.info("Starting data loading process...")
        
        # Load individual datasets
        self.swipes_data = self._load_swipes_data()
        self.class_enrollments = self._load_class_enrollments()
        self.students_data = self._load_students_data()
        
        # Process enrollments into section-student pairs
        self.section_enrollments = self._process_enrollments()
        
        logger.info("Data loading completed successfully!")
        
        return {
            'swipes_data': self.swipes_data,
            'class_enrollments': self.class_enrollments,
            'students_data': self.students_data,
            'section_enrollments': self.section_enrollments
        }
    
    def _load_swipes_data(self) -> pd.DataFrame:
        """Load and validate swipes data."""
        logger.info("Loading swipes data...")
        
        file_path = self.data_dir / "swipes_data.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Swipes data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate meal_period values
        self._validate_meal_periods(df)
        
        logger.info(f"Loaded {len(df)} swipe records")
        return df
    
    def _load_class_enrollments(self) -> pd.DataFrame:
        """Load and process class enrollment data."""
        logger.info("Loading class enrollment data...")
        
        file_path = self.data_dir / "class_enrollments.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Class enrollments file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Convert time strings to minutes since midnight
        df['start_time_minutes'] = df['start_time'].apply(self._time_to_minutes)
        df['end_time_minutes'] = df['end_time'].apply(self._time_to_minutes)
        
        # Add section_id if not present (using row index)
        if 'section_id' not in df.columns:
            df['section_id'] = df.index
        
        logger.info(f"Loaded {len(df)} class sections")
        return df
    
    def _load_students_data(self) -> pd.DataFrame:
        """Load and concatenate all students CSV files."""
        logger.info("Loading students data...")
        
        students_files = [
            "students_1.csv",
            "students_2.csv", 
            "students_3.csv"
        ]
        
        dfs = []
        for filename in students_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} students from {filename}")
            else:
                logger.warning(f"Students file not found: {file_path}")
        
        if not dfs:
            raise FileNotFoundError("No students CSV files found")
        
        # Concatenate all student dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Deduplicate on student_id
        original_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['student_id'], keep='first')
        deduplicated_count = len(combined_df)
        
        if original_count != deduplicated_count:
            logger.info(f"Removed {original_count - deduplicated_count} duplicate student records")
        
        logger.info(f"Total unique students: {len(combined_df)}")
        return combined_df
    
    def _process_enrollments(self) -> pd.DataFrame:
        """Process enrollment strings into section-student pairs."""
        logger.info("Processing class enrollments...")
        
        if self.class_enrollments is None:
            raise ValueError("Class enrollments must be loaded first")
        
        enrollment_records = []
        
        for idx, row in self.class_enrollments.iterrows():
            section_id = row['section_id']
            enrollment_str = row['enrollment']
            
            try:
                # Parse the enrollment string - remove brackets and split by comma
                # The format is "[STU001,STU002,STU003]" so we need to clean it
                clean_str = enrollment_str.strip('[]')
                student_list = [student.strip() for student in clean_str.split(',')]
                
                # Create records for each student in this section
                for student_id in student_list:
                    enrollment_records.append({
                        'section_id': section_id,
                        'student_id': student_id,
                        'start_time': row['start_time'],
                        'end_time': row['end_time'],
                        'start_time_minutes': row['start_time_minutes'],
                        'end_time_minutes': row['end_time_minutes'],
                        'course_code': row.get('course_code', ''),
                        'course_name': row.get('course_name', ''),
                        'days_of_week': row.get('days_of_week', '')
                    })
                    
            except (ValueError, AttributeError) as e:
                logger.error(f"Failed to parse enrollment for section {section_id}: {e}")
                continue
        
        df = pd.DataFrame(enrollment_records)
        logger.info(f"Created {len(df)} section-student enrollment records")
        return df
    
    def _validate_meal_periods(self, df: pd.DataFrame) -> None:
        """Validate meal_period values and log any out-of-domain entries."""
        if 'meal_period' not in df.columns:
            logger.warning("No meal_period column found in swipes data")
            return
        
        invalid_periods = df[~df['meal_period'].isin(self.valid_meal_periods)]
        
        if len(invalid_periods) > 0:
            logger.warning(f"Found {len(invalid_periods)} invalid meal_period entries:")
            invalid_counts = invalid_periods['meal_period'].value_counts()
            for period, count in invalid_counts.items():
                logger.warning(f"  - '{period}': {count} occurrences")
        else:
            logger.info("All meal_period values are valid")
    
    def _time_to_minutes(self, time_str: str) -> int:
        """
        Convert time string (HH:MM) to minutes since midnight.
        
        Args:
            time_str: Time in format "HH:MM"
            
        Returns:
            Minutes since midnight as integer
        """
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to parse time '{time_str}': {e}")
            return 0
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary statistics of loaded data."""
        summary = {}
        
        if self.swipes_data is not None:
            summary['swipes'] = {
                'total_records': len(self.swipes_data),
                'unique_students': self.swipes_data['student_id'].nunique(),
                'meal_period_counts': self.swipes_data['meal_period'].value_counts().to_dict()
            }
        
        if self.class_enrollments is not None:
            summary['classes'] = {
                'total_sections': len(self.class_enrollments),
                'time_range': {
                    'earliest_start': min(self.class_enrollments['start_time_minutes']),
                    'latest_end': max(self.class_enrollments['end_time_minutes'])
                }
            }
        
        if self.students_data is not None:
            summary['students'] = {
                'total_students': len(self.students_data),
                'meal_plan_counts': self.students_data['meal_plan'].value_counts().to_dict()
            }
        
        if self.section_enrollments is not None:
            summary['enrollments'] = {
                'total_enrollments': len(self.section_enrollments),
                'avg_students_per_section': len(self.section_enrollments) / len(self.class_enrollments) if self.class_enrollments is not None else 0
            }
        
        return summary


def main():
    """Example usage of DataLoader."""
    loader = DataLoader()
    
    try:
        data = loader.load_all_data()
        
        # Print summary
        summary = loader.get_data_summary()
        print("\n=== Data Summary ===")
        for dataset, stats in summary.items():
            print(f"\n{dataset.upper()}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        # Show sample data
        print("\n=== Sample Data ===")
        print("\nSwipes Data (first 5 rows):")
        print(data['swipes_data'].head())
        
        print("\nClass Enrollments (first 3 rows):")
        print(data['class_enrollments'][['course_code', 'start_time', 'end_time', 'start_time_minutes', 'end_time_minutes']].head(3))
        
        print("\nSection Enrollments (first 5 rows):")
        print(data['section_enrollments'][['section_id', 'student_id', 'start_time_minutes', 'end_time_minutes']].head())
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


if __name__ == "__main__":
    main()
