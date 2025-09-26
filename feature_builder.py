"""
FeatureBuilder module for dining demand optimization system.

This module constructs a comprehensive training matrix for machine learning,
combining meal propensities, meal plans, time bin features, and schedule context.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from time_grid_service import TimeGridService, MealPeriod
from meal_propensity_service import MealPropensityService
from availability_service import AvailabilityService
from sections_catalog import SectionsCatalog

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingFeatures:
    """Represents a single training example."""
    student_id: str
    bin_index: int
    meal_period_bin: str
    minutes_from_meal_center: int
    bin_index_normalized: float
    
    # Meal propensity features
    p_breakfast: float
    p_lunch: float
    p_dinner: float
    
    # Meal plan features
    weekly_allowance: float
    unlimited: bool
    is_flex: bool
    
    # Schedule features
    has_class_in_bin: bool
    free_gap_before_min: int
    free_gap_after_min: int
    is_large_gap: bool
    
    # Target label
    target_probability: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'student_id': self.student_id,
            'bin_index': self.bin_index,
            'meal_period_bin': self.meal_period_bin,
            'minutes_from_meal_center': self.minutes_from_meal_center,
            'bin_index_normalized': self.bin_index_normalized,
            'p_breakfast': self.p_breakfast,
            'p_lunch': self.p_lunch,
            'p_dinner': self.p_dinner,
            'weekly_allowance': self.weekly_allowance,
            'unlimited': self.unlimited,
            'is_flex': self.is_flex,
            'has_class_in_bin': self.has_class_in_bin,
            'free_gap_before_min': self.free_gap_before_min,
            'free_gap_after_min': self.free_gap_after_min,
            'is_large_gap': self.is_large_gap,
            'target_probability': self.target_probability
        }


class FeatureBuilder:
    """Builder for ML training features from dining demand data."""
    
    def __init__(self, 
                 time_grid_service: TimeGridService,
                 meal_propensity_service: MealPropensityService,
                 availability_service: AvailabilityService,
                 sections_catalog: SectionsCatalog):
        """
        Initialize FeatureBuilder.
        
        Args:
            time_grid_service: TimeGridService instance
            meal_propensity_service: MealPropensityService instance
            availability_service: AvailabilityService instance
            sections_catalog: SectionsCatalog instance
        """
        self.time_grid = time_grid_service
        self.meal_propensity = meal_propensity_service
        self.availability = availability_service
        self.sections_catalog = sections_catalog
        
        # Meal period centers (in minutes since midnight)
        self.meal_centers = {
            'Breakfast': 8 * 60 + 45,  # 08:45 (middle of 07:00-10:30)
            'Lunch': 12 * 60 + 45,     # 12:45 (middle of 11:00-14:30)
            'Dinner': 18 * 60 + 45     # 18:45 (middle of 17:00-20:30)
        }
        
        logger.info("Initialized FeatureBuilder")
    
    def build_training_matrix(self, student_ids: List[str]) -> pd.DataFrame:
        """
        Build comprehensive training matrix for all students and time bins.
        
        Args:
            student_ids: List of student IDs to include
            
        Returns:
            DataFrame with training features
        """
        logger.info(f"Building training matrix for {len(student_ids)} students...")
        
        training_examples = []
        total_bins = len(self.time_grid.time_bins)
        
        for student_id in student_ids:
            # Get student's meal propensity data
            propensity = self.meal_propensity.get_student_propensity(student_id)
            if propensity is None:
                logger.warning(f"No propensity data for student {student_id}, skipping")
                continue
            
            # Get student's meal plan data (assuming it's available from meal_propensity_service)
            # For now, we'll use default values - in practice, this would come from MealPlanNormalizer
            meal_plan_features = self._get_meal_plan_features(student_id)
            
            # Build features for each time bin
            for bin_idx in range(total_bins):
                bin_obj = self.time_grid.get_bin(bin_idx)
                
                # Build features for this (student_id, bin_index) pair
                features = self._build_single_features(
                    student_id, bin_idx, bin_obj, propensity, meal_plan_features
                )
                
                training_examples.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame([example.to_dict() for example in training_examples])
        
        logger.info(f"Built training matrix with {len(df)} examples")
        self._log_feature_summary(df)
        
        return df
    
    def _get_meal_plan_features(self, student_id: str) -> Dict:
        """
        Get meal plan features for a student.
        
        Args:
            student_id: Student identifier
            
        Returns:
            Dictionary with meal plan features
        """
        # TODO: In practice, this would come from MealPlanNormalizer
        # For now, return default values
        return {
            'weekly_allowance': 14.0,  # Default to 14 meals
            'unlimited': False,
            'is_flex': False
        }
    
    def _build_single_features(self, 
                              student_id: str, 
                              bin_idx: int, 
                              bin_obj, 
                              propensity, 
                              meal_plan_features: Dict) -> TrainingFeatures:
        """Build features for a single (student_id, bin_index) pair."""
        
        # Bin features
        meal_period_bin = bin_obj.meal_period.value
        minutes_from_meal_center = self._calculate_minutes_from_meal_center(
            bin_obj.start_minutes, meal_period_bin
        )
        bin_index_normalized = bin_idx / len(self.time_grid.time_bins)
        
        # Schedule features
        has_class_in_bin = self.availability.has_class_in_bin(student_id, bin_idx)
        free_gap_before = self._calculate_free_gap_before(student_id, bin_idx)
        free_gap_after = self._calculate_free_gap_after(student_id, bin_idx)
        is_large_gap = (free_gap_before >= 60) and (free_gap_after >= 60)  # 1+ hour gaps
        
        # Target probability (weak label)
        target_probability = self._calculate_target_probability(
            student_id, bin_idx, meal_period_bin, free_gap_before, free_gap_after
        )
        
        return TrainingFeatures(
            student_id=student_id,
            bin_index=bin_idx,
            meal_period_bin=meal_period_bin,
            minutes_from_meal_center=minutes_from_meal_center,
            bin_index_normalized=bin_index_normalized,
            p_breakfast=propensity.p_breakfast,
            p_lunch=propensity.p_lunch,
            p_dinner=propensity.p_dinner,
            weekly_allowance=meal_plan_features['weekly_allowance'],
            unlimited=meal_plan_features['unlimited'],
            is_flex=meal_plan_features['is_flex'],
            has_class_in_bin=has_class_in_bin,
            free_gap_before_min=free_gap_before,
            free_gap_after_min=free_gap_after,
            is_large_gap=is_large_gap,
            target_probability=target_probability
        )
    
    def _calculate_minutes_from_meal_center(self, bin_start_minutes: int, meal_period: str) -> int:
        """Calculate minutes from the center of the meal period."""
        if meal_period in self.meal_centers:
            center_minutes = self.meal_centers[meal_period]
            return abs(bin_start_minutes - center_minutes)
        else:
            return 0  # For 'None' meal periods
    
    def _calculate_free_gap_before(self, student_id: str, bin_idx: int) -> int:
        """Calculate free time gap before this bin (in minutes)."""
        if bin_idx == 0:
            return 0
        
        # Look backwards for consecutive free bins
        gap_minutes = 0
        bin_size = self.time_grid.bin_size_minutes
        
        for i in range(bin_idx - 1, -1, -1):
            if self.availability.is_free_in_bin(student_id, i):
                gap_minutes += bin_size
            else:
                break
        
        return gap_minutes
    
    def _calculate_free_gap_after(self, student_id: str, bin_idx: int) -> int:
        """Calculate free time gap after this bin (in minutes)."""
        total_bins = len(self.time_grid.time_bins)
        if bin_idx >= total_bins - 1:
            return 0
        
        # Look forwards for consecutive free bins
        gap_minutes = 0
        bin_size = self.time_grid.bin_size_minutes
        
        for i in range(bin_idx + 1, total_bins):
            if self.availability.is_free_in_bin(student_id, i):
                gap_minutes += bin_size
            else:
                break
        
        return gap_minutes
    
    def _calculate_target_probability(self, 
                                    student_id: str, 
                                    bin_idx: int, 
                                    meal_period: str, 
                                    free_gap_before: int, 
                                    free_gap_after: int) -> float:
        """
        Calculate target probability for weak labeling.
        
        Distributes meal propensity into bins of the meal window,
        weighting bins with larger free gaps higher.
        """
        # Get base propensity for this meal period
        if meal_period == 'Breakfast':
            base_propensity = self.meal_propensity.get_meal_period_propensity(student_id, 'breakfast')
        elif meal_period == 'Lunch':
            base_propensity = self.meal_propensity.get_meal_period_propensity(student_id, 'lunch')
        elif meal_period == 'Dinner':
            base_propensity = self.meal_propensity.get_meal_period_propensity(student_id, 'dinner')
        else:
            return 0.0  # No meal propensity for 'None' periods
        
        # If student has class in this bin, probability is 0
        if self.availability.has_class_in_bin(student_id, bin_idx):
            return 0.0
        
        # Calculate gap weight (larger gaps = higher weight)
        total_gap = free_gap_before + free_gap_after
        gap_weight = min(1.0, total_gap / 120.0)  # Normalize to 0-1, max at 2 hours
        
        # Calculate distance weight (closer to meal center = higher weight)
        bin_obj = self.time_grid.get_bin(bin_idx)
        minutes_from_center = self._calculate_minutes_from_meal_center(bin_obj.start_minutes, meal_period)
        distance_weight = max(0.1, 1.0 - (minutes_from_center / 180.0))  # Decay over 3 hours
        
        # Combine weights
        combined_weight = gap_weight * distance_weight
        
        # Calculate final probability
        target_probability = base_propensity * combined_weight
        
        return min(1.0, target_probability)  # Cap at 1.0
    
    def _log_feature_summary(self, df: pd.DataFrame) -> None:
        """Log summary of built features."""
        logger.info("=== Feature Summary ===")
        logger.info(f"Total examples: {len(df)}")
        logger.info(f"Unique students: {df['student_id'].nunique()}")
        logger.info(f"Unique bins: {df['bin_index'].nunique()}")
        
        # Meal period distribution
        meal_dist = df['meal_period_bin'].value_counts()
        logger.info("Meal period distribution:")
        for period, count in meal_dist.items():
            logger.info(f"  {period}: {count}")
        
        # Target probability statistics
        target_stats = df['target_probability'].describe()
        logger.info("Target probability statistics:")
        logger.info(f"  Mean: {target_stats['mean']:.3f}")
        logger.info(f"  Std: {target_stats['std']:.3f}")
        logger.info(f"  Min: {target_stats['min']:.3f}")
        logger.info(f"  Max: {target_stats['max']:.3f}")
        
        # Class distribution
        class_dist = df['has_class_in_bin'].value_counts()
        logger.info("Class distribution:")
        logger.info(f"  In class: {class_dist.get(True, 0)}")
        logger.info(f"  Free: {class_dist.get(False, 0)}")
    
    def get_feature_importance_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze feature importance and correlations.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dictionary with feature analysis
        """
        # Calculate correlations with target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['target_probability'].abs().sort_values(ascending=False)
        
        # Analyze meal period patterns
        meal_patterns = {}
        for period in ['Breakfast', 'Lunch', 'Dinner', 'None']:
            period_data = df[df['meal_period_bin'] == period]
            if len(period_data) > 0:
                meal_patterns[period] = {
                    'count': len(period_data),
                    'avg_target': period_data['target_probability'].mean(),
                    'avg_gap_before': period_data['free_gap_before_min'].mean(),
                    'avg_gap_after': period_data['free_gap_after_min'].mean()
                }
        
        return {
            'feature_correlations': correlations.to_dict(),
            'meal_patterns': meal_patterns,
            'total_examples': len(df),
            'feature_columns': list(df.columns)
        }
    
    def export_training_data(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Export training data to CSV.
        
        Args:
            df: Training DataFrame
            filepath: Output file path
        """
        df.to_csv(filepath, index=False)
        logger.info(f"Exported training data to {filepath}")


def main():
    """Example usage of FeatureBuilder."""
    from dataloader import DataLoader
    
    # Load all data
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Create all services
    time_grid = TimeGridService()
    sections_catalog = SectionsCatalog(time_grid)
    sections_catalog.build_catalog(data['class_enrollments'])
    
    # Get student IDs from section enrollments
    student_ids = list(set(data['section_enrollments']['student_id'].tolist()))
    
    # Create meal propensity service
    meal_propensity = MealPropensityService()
    meal_propensity.compute_meal_propensities(data['swipes_data'], student_ids)
    
    # Create availability service
    availability = AvailabilityService(time_grid, sections_catalog)
    availability.compute_availability(student_ids)
    
    # Create feature builder
    feature_builder = FeatureBuilder(time_grid, meal_propensity, availability, sections_catalog)
    
    # Build training matrix
    training_df = feature_builder.build_training_matrix(student_ids)
    
    print("=== Feature Builder Demo ===")
    print(f"Training matrix shape: {training_df.shape}")
    print()
    
    # Show sample data
    print("Sample Training Examples (first 5):")
    print(training_df.head())
    print()
    
    # Show feature analysis
    print("Feature Analysis:")
    analysis = feature_builder.get_feature_importance_analysis(training_df)
    
    print("Top Feature Correlations with Target:")
    correlations = analysis['feature_correlations']
    for feature, corr in list(correlations.items())[:10]:
        if feature != 'target_probability':
            print(f"  {feature}: {corr:.3f}")
    print()
    
    print("Meal Period Patterns:")
    for period, patterns in analysis['meal_patterns'].items():
        print(f"  {period}:")
        print(f"    Examples: {patterns['count']}")
        print(f"    Avg target: {patterns['avg_target']:.3f}")
        print(f"    Avg gap before: {patterns['avg_gap_before']:.1f} min")
        print(f"    Avg gap after: {patterns['avg_gap_after']:.1f} min")
    print()
    
    # Show target distribution
    print("Target Probability Distribution:")
    target_stats = training_df['target_probability'].describe()
    print(f"  Mean: {target_stats['mean']:.3f}")
    print(f"  Median: {target_stats['50%']:.3f}")
    print(f"  Std: {target_stats['std']:.3f}")
    print(f"  Min: {target_stats['min']:.3f}")
    print(f"  Max: {target_stats['max']:.3f}")
    
    # Show examples with high target probability
    high_target = training_df[training_df['target_probability'] > 0.1].head()
    if len(high_target) > 0:
        print(f"\nExamples with High Target Probability (>0.1):")
        print(high_target[['student_id', 'bin_index', 'meal_period_bin', 'target_probability', 'free_gap_before_min', 'free_gap_after_min']])


if __name__ == "__main__":
    main()
