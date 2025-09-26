"""
MealPropensityService module for dining demand optimization system.

This module computes baseline meal propensity labels for each student based on
historical dining patterns, providing the foundation for demand modeling.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StudentMealPropensity:
    """Represents meal propensity data for a single student."""
    student_id: str
    p_breakfast: float
    p_lunch: float
    p_dinner: float
    total_swipes: int
    has_historical_data: bool
    
    def get_meal_period_propensity(self, meal_period: str) -> float:
        """Get propensity for a specific meal period."""
        meal_period = meal_period.lower()
        if meal_period == 'breakfast':
            return self.p_breakfast
        elif meal_period == 'lunch':
            return self.p_lunch
        elif meal_period == 'dinner':
            return self.p_dinner
        else:
            return 0.0
    
    def get_primary_meal_period(self) -> str:
        """Get the meal period with highest propensity."""
        propensities = {
            'breakfast': self.p_breakfast,
            'lunch': self.p_lunch,
            'dinner': self.p_dinner
        }
        return max(propensities, key=propensities.get)
    
    def __str__(self) -> str:
        return (f"Student {self.student_id}: "
                f"Breakfast={self.p_breakfast:.3f}, "
                f"Lunch={self.p_lunch:.3f}, "
                f"Dinner={self.p_dinner:.3f} "
                f"({self.total_swipes} swipes, "
                f"{'historical' if self.has_historical_data else 'estimated'})")


class MealPropensityService:
    """Service for computing and managing student meal propensity data."""
    
    def __init__(self):
        """Initialize MealPropensityService."""
        self.student_meal_props = {}  # student_id -> StudentMealPropensity
        self.population_averages = {}  # meal_period -> average propensity
        self.meal_periods = ['breakfast', 'lunch', 'dinner']
        
        logger.info("Initialized MealPropensityService")
    
    def compute_meal_propensities(self, swipes_data: pd.DataFrame, all_student_ids: List[str]) -> Dict[str, StudentMealPropensity]:
        """
        Compute meal propensities for all students.
        
        Args:
            swipes_data: DataFrame with swipe data (student_id, meal_period)
            all_student_ids: List of all student IDs to include
            
        Returns:
            Dictionary mapping student_id to StudentMealPropensity
        """
        logger.info(f"Computing meal propensities for {len(all_student_ids)} students...")
        
        # Clear existing data
        self.student_meal_props.clear()
        
        # Compute propensities for students with historical data
        self._compute_historical_propensities(swipes_data)
        
        # Compute population averages
        self._compute_population_averages(swipes_data)
        
        # Backfill missing students with population averages
        self._backfill_missing_students(all_student_ids)
        
        logger.info(f"Computed propensities for {len(self.student_meal_props)} students")
        self._log_propensity_summary()
        
        return self.student_meal_props.copy()
    
    def _compute_historical_propensities(self, swipes_data: pd.DataFrame) -> None:
        """Compute propensities for students with historical swipe data."""
        if swipes_data.empty:
            return
        
        # Group by student and meal period
        meal_counts = swipes_data.groupby(['student_id', 'meal_period']).size().reset_index(name='count')
        
        # Pivot to get counts by meal period
        pivot_counts = meal_counts.pivot(index='student_id', columns='meal_period', values='count').fillna(0)
        
        # Ensure all meal periods are present
        for period in self.meal_periods:
            if period.capitalize() not in pivot_counts.columns:
                pivot_counts[period.capitalize()] = 0
        
        # Convert to proportions
        for student_id in pivot_counts.index:
            total_swipes = pivot_counts.loc[student_id].sum()
            
            if total_swipes > 0:
                propensities = StudentMealPropensity(
                    student_id=student_id,
                    p_breakfast=pivot_counts.loc[student_id, 'Breakfast'] / total_swipes,
                    p_lunch=pivot_counts.loc[student_id, 'Lunch'] / total_swipes,
                    p_dinner=pivot_counts.loc[student_id, 'Dinner'] / total_swipes,
                    total_swipes=int(total_swipes),
                    has_historical_data=True
                )
                
                self.student_meal_props[student_id] = propensities
    
    def _compute_population_averages(self, swipes_data: pd.DataFrame) -> None:
        """Compute population average propensities."""
        if swipes_data.empty:
            # Default to equal distribution if no data
            self.population_averages = {
                'breakfast': 1.0 / 3,
                'lunch': 1.0 / 3,
                'dinner': 1.0 / 3
            }
            return
        
        # Count total swipes by meal period
        meal_counts = swipes_data['meal_period'].value_counts()
        total_swipes = meal_counts.sum()
        
        # Calculate proportions
        self.population_averages = {
            'breakfast': meal_counts.get('Breakfast', 0) / total_swipes,
            'lunch': meal_counts.get('Lunch', 0) / total_swipes,
            'dinner': meal_counts.get('Dinner', 0) / total_swipes
        }
        
        logger.info(f"Population averages: {self.population_averages}")
    
    def _backfill_missing_students(self, all_student_ids: List[str]) -> None:
        """Backfill missing students with population averages."""
        missing_students = set(all_student_ids) - set(self.student_meal_props.keys())
        
        for student_id in missing_students:
            propensities = StudentMealPropensity(
                student_id=student_id,
                p_breakfast=self.population_averages['breakfast'],
                p_lunch=self.population_averages['lunch'],
                p_dinner=self.population_averages['dinner'],
                total_swipes=0,
                has_historical_data=False
            )
            
            self.student_meal_props[student_id] = propensities
        
        if missing_students:
            logger.info(f"Backfilled {len(missing_students)} students with population averages")
    
    def get_student_propensity(self, student_id: str) -> Optional[StudentMealPropensity]:
        """
        Get meal propensity data for a specific student.
        
        Args:
            student_id: Student identifier
            
        Returns:
            StudentMealPropensity object or None if not found
        """
        return self.student_meal_props.get(student_id)
    
    def get_meal_period_propensity(self, student_id: str, meal_period: str) -> float:
        """
        Get propensity for a specific student and meal period.
        
        Args:
            student_id: Student identifier
            meal_period: Meal period ('breakfast', 'lunch', 'dinner')
            
        Returns:
            Propensity value (0.0 to 1.0)
        """
        propensity = self.get_student_propensity(student_id)
        if propensity is None:
            return 0.0
        
        return propensity.get_meal_period_propensity(meal_period)
    
    def get_students_by_meal_preference(self, meal_period: str, min_propensity: float = 0.0) -> List[str]:
        """
        Get students with high propensity for a specific meal period.
        
        Args:
            meal_period: Meal period to filter by
            min_propensity: Minimum propensity threshold
            
        Returns:
            List of student IDs
        """
        matching_students = []
        
        for student_id, propensity in self.student_meal_props.items():
            if propensity.get_meal_period_propensity(meal_period) >= min_propensity:
                matching_students.append(student_id)
        
        return matching_students
    
    def get_propensity_distribution(self, meal_period: str) -> Dict[str, int]:
        """
        Get distribution of propensities for a meal period.
        
        Args:
            meal_period: Meal period to analyze
            
        Returns:
            Dictionary with propensity ranges and counts
        """
        propensities = []
        for propensity in self.student_meal_props.values():
            propensities.append(propensity.get_meal_period_propensity(meal_period))
        
        # Create bins
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        
        distribution = {label: 0 for label in bin_labels}
        
        for prop in propensities:
            for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
                if low <= prop < high or (i == len(bins) - 2 and prop == 1.0):
                    distribution[bin_labels[i]] += 1
                    break
        
        return distribution
    
    def get_propensity_summary(self) -> Dict:
        """
        Get comprehensive summary of propensity data.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.student_meal_props:
            return {}
        
        total_students = len(self.student_meal_props)
        historical_students = sum(1 for p in self.student_meal_props.values() if p.has_historical_data)
        estimated_students = total_students - historical_students
        
        # Calculate average propensities
        avg_propensities = {}
        for period in self.meal_periods:
            propensities = [p.get_meal_period_propensity(period) for p in self.student_meal_props.values()]
            avg_propensities[period] = np.mean(propensities)
        
        # Find students with strong preferences
        strong_preferences = {}
        for period in self.meal_periods:
            strong_prefs = self.get_students_by_meal_preference(period, 0.6)
            strong_preferences[period] = len(strong_prefs)
        
        return {
            'total_students': total_students,
            'historical_students': historical_students,
            'estimated_students': estimated_students,
            'avg_propensities': avg_propensities,
            'population_averages': self.population_averages,
            'strong_preferences': strong_preferences
        }
    
    def export_student_meal_props_table(self) -> pd.DataFrame:
        """
        Export student meal propensities as a DataFrame.
        
        Returns:
            DataFrame with columns: student_id, p_breakfast, p_lunch, p_dinner, total_swipes, has_historical_data
        """
        data = []
        
        for propensity in self.student_meal_props.values():
            data.append({
                'student_id': propensity.student_id,
                'p_breakfast': propensity.p_breakfast,
                'p_lunch': propensity.p_lunch,
                'p_dinner': propensity.p_dinner,
                'total_swipes': propensity.total_swipes,
                'has_historical_data': propensity.has_historical_data
            })
        
        return pd.DataFrame(data)
    
    def _log_propensity_summary(self) -> None:
        """Log summary of computed propensities."""
        if not self.student_meal_props:
            return
        
        summary = self.get_propensity_summary()
        
        logger.info("=== Meal Propensity Summary ===")
        logger.info(f"Total students: {summary['total_students']}")
        logger.info(f"Historical data: {summary['historical_students']}")
        logger.info(f"Estimated (population avg): {summary['estimated_students']}")
        logger.info("Average propensities:")
        for period, avg in summary['avg_propensities'].items():
            logger.info(f"  {period.capitalize()}: {avg:.3f}")
        logger.info("Strong preferences (≥60%):")
        for period, count in summary['strong_preferences'].items():
            logger.info(f"  {period.capitalize()}: {count} students")
    
    def __str__(self) -> str:
        """String representation of the service."""
        if not self.student_meal_props:
            return "MealPropensityService: No data loaded"
        
        summary = self.get_propensity_summary()
        return (f"MealPropensityService: {summary['total_students']} students, "
                f"{summary['historical_students']} with historical data")


def main():
    """Example usage of MealPropensityService."""
    from dataloader import DataLoader
    
    # Load data
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Get all student IDs from the system
    all_student_ids = list(set(data['students_data']['student_id'].tolist()))
    
    # Create propensity service
    propensity_service = MealPropensityService()
    
    # Compute meal propensities
    propensities = propensity_service.compute_meal_propensities(
        data['swipes_data'], 
        all_student_ids
    )
    
    print("=== Meal Propensity Service Demo ===")
    print(f"Service: {propensity_service}")
    print()
    
    # Show summary
    summary = propensity_service.get_propensity_summary()
    print("Propensity Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Show sample propensities
    print("Sample Student Propensities (first 10):")
    sample_students = list(propensities.keys())[:10]
    for student_id in sample_students:
        propensity = propensities[student_id]
        print(f"  {propensity}")
    print()
    
    # Test individual queries
    print("Individual Query Tests:")
    sample_student = sample_students[0]
    print(f"Student {sample_student}:")
    print(f"  Breakfast propensity: {propensity_service.get_meal_period_propensity(sample_student, 'breakfast'):.3f}")
    print(f"  Lunch propensity: {propensity_service.get_meal_period_propensity(sample_student, 'lunch'):.3f}")
    print(f"  Dinner propensity: {propensity_service.get_meal_period_propensity(sample_student, 'dinner'):.3f}")
    print(f"  Primary meal period: {propensities[sample_student].get_primary_meal_period()}")
    print()
    
    # Show propensity distributions
    print("Propensity Distributions:")
    for period in ['breakfast', 'lunch', 'dinner']:
        distribution = propensity_service.get_propensity_distribution(period)
        print(f"  {period.capitalize()}:")
        for range_label, count in distribution.items():
            print(f"    {range_label}: {count} students")
    print()
    
    # Show students with strong preferences
    print("Students with Strong Preferences (≥60%):")
    for period in ['breakfast', 'lunch', 'dinner']:
        strong_prefs = propensity_service.get_students_by_meal_preference(period, 0.6)
        print(f"  {period.capitalize()}: {len(strong_prefs)} students")
        if strong_prefs:
            print(f"    Examples: {strong_prefs[:5]}")
    print()
    
    # Export table
    print("Export Test (first 5 rows):")
    df = propensity_service.export_student_meal_props_table()
    print(df.head())


if __name__ == "__main__":
    main()
