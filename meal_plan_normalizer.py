"""
MealPlanNormalizer module for dining demand optimization system.

This module normalizes meal plan data from the students table into a standardized format
that can be used by downstream ML and optimization components.
"""

import pandas as pd
import logging
import re
from typing import Dict, Tuple, Optional
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MealPlanNormalizer:
    """Normalizes meal plan data into standardized format for optimization."""
    
    def __init__(self):
        """Initialize MealPlanNormalizer."""
        self.students_clean = None
        
        # Define meal plan patterns and their normalized values
        self.plan_patterns = {
            # Unlimited plans
            'unlimited': {
                'patterns': [r'unlimited', r'unltd', r'unlim'],
                'weekly_allowance': math.inf,
                'unlimited': True,
                'is_flex': False
            },
            'unlimited_flex': {
                'patterns': [r'unlimited\s+flex', r'unltd\s+flex', r'unlim\s+flex'],
                'weekly_allowance': math.inf,
                'unlimited': True,
                'is_flex': True
            },
            # Limited meal plans
            'flex_plan': {
                'patterns': [r'flex\s+plan', r'flex\s+meals'],
                'weekly_allowance': math.inf,  # Flex plans typically have unlimited swipes but limited dining hall access
                'unlimited': False,
                'is_flex': True
            },
            # Numeric meal plans
            'numeric_meals': {
                'patterns': [r'(\d+)\s*meals?', r'(\d+)\s*swipes?'],
                'weekly_allowance': None,  # Will be extracted from pattern
                'unlimited': False,
                'is_flex': False
            }
        }
    
    def normalize_meal_plans(self, students_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize meal plan data from students table.
        
        Args:
            students_data: DataFrame with student_id and meal_plan columns
            
        Returns:
            DataFrame with normalized meal plan data
        """
        logger.info("Starting meal plan normalization...")
        
        if 'student_id' not in students_data.columns or 'meal_plan' not in students_data.columns:
            raise ValueError("students_data must contain 'student_id' and 'meal_plan' columns")
        
        # Create working copy
        df = students_data[['student_id', 'meal_plan']].copy()
        
        # Initialize normalized columns
        df['weekly_allowance'] = None
        df['unlimited'] = False
        df['is_flex'] = False
        df['plan_raw'] = df['meal_plan']
        
        # Process each student's meal plan
        for idx, row in df.iterrows():
            meal_plan = str(row['meal_plan']).strip()
            normalized = self._normalize_single_plan(meal_plan)
            
            df.at[idx, 'weekly_allowance'] = normalized['weekly_allowance']
            df.at[idx, 'unlimited'] = normalized['unlimited']
            df.at[idx, 'is_flex'] = normalized['is_flex']
        
        # Create clean table with only required columns
        self.students_clean = df[['student_id', 'weekly_allowance', 'unlimited', 'is_flex', 'plan_raw']].copy()
        
        # Log summary statistics
        self._log_normalization_summary()
        
        logger.info(f"Normalized {len(self.students_clean)} student meal plans")
        return self.students_clean
    
    def _normalize_single_plan(self, meal_plan: str) -> Dict:
        """
        Normalize a single meal plan string.
        
        Args:
            meal_plan: Raw meal plan string
            
        Returns:
            Dictionary with normalized values
        """
        meal_plan_lower = meal_plan.lower()
        
        # Check for unlimited flex plans first (more specific)
        if self._matches_patterns(meal_plan_lower, self.plan_patterns['unlimited_flex']['patterns']):
            return {
                'weekly_allowance': self.plan_patterns['unlimited_flex']['weekly_allowance'],
                'unlimited': self.plan_patterns['unlimited_flex']['unlimited'],
                'is_flex': self.plan_patterns['unlimited_flex']['is_flex']
            }
        
        # Check for regular unlimited plans
        elif self._matches_patterns(meal_plan_lower, self.plan_patterns['unlimited']['patterns']):
            return {
                'weekly_allowance': self.plan_patterns['unlimited']['weekly_allowance'],
                'unlimited': self.plan_patterns['unlimited']['unlimited'],
                'is_flex': self.plan_patterns['unlimited']['is_flex']
            }
        
        # Check for flex plans
        elif self._matches_patterns(meal_plan_lower, self.plan_patterns['flex_plan']['patterns']):
            return {
                'weekly_allowance': self.plan_patterns['flex_plan']['weekly_allowance'],
                'unlimited': self.plan_patterns['flex_plan']['unlimited'],
                'is_flex': self.plan_patterns['flex_plan']['is_flex']
            }
        
        # Check for numeric meal plans
        elif self._matches_patterns(meal_plan_lower, self.plan_patterns['numeric_meals']['patterns']):
            # Extract the number from the pattern
            match = re.search(r'(\d+)', meal_plan)
            if match:
                weekly_allowance = int(match.group(1))
                return {
                    'weekly_allowance': weekly_allowance,
                    'unlimited': False,
                    'is_flex': False
                }
        
        # Default case - unknown plan type
        logger.warning(f"Unknown meal plan format: '{meal_plan}' - treating as limited plan")
        return {
            'weekly_allowance': 0,  # Conservative default
            'unlimited': False,
            'is_flex': False
        }
    
    def _matches_patterns(self, text: str, patterns: list) -> bool:
        """
        Check if text matches any of the given patterns.
        
        Args:
            text: Text to check
            patterns: List of regex patterns
            
        Returns:
            True if any pattern matches
        """
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _log_normalization_summary(self) -> None:
        """Log summary statistics of normalized meal plans."""
        if self.students_clean is None:
            return
        
        logger.info("=== Meal Plan Normalization Summary ===")
        
        # Count by unlimited status
        unlimited_count = self.students_clean['unlimited'].sum()
        limited_count = len(self.students_clean) - unlimited_count
        
        logger.info(f"Unlimited plans: {unlimited_count}")
        logger.info(f"Limited plans: {limited_count}")
        
        # Count by flex status
        flex_count = self.students_clean['is_flex'].sum()
        non_flex_count = len(self.students_clean) - flex_count
        
        logger.info(f"Flex plans: {flex_count}")
        logger.info(f"Non-flex plans: {non_flex_count}")
        
        # Count by weekly allowance ranges
        finite_allowances = self.students_clean[self.students_clean['weekly_allowance'] != math.inf]
        if len(finite_allowances) > 0:
            allowance_counts = finite_allowances['weekly_allowance'].value_counts().sort_index()
            logger.info("Weekly allowance distribution:")
            for allowance, count in allowance_counts.items():
                logger.info(f"  {allowance} meals: {count} students")
        
        # Show unique plan types
        unique_plans = self.students_clean['plan_raw'].value_counts()
        logger.info(f"Unique plan types found: {len(unique_plans)}")
        for plan, count in unique_plans.head(10).items():
            logger.info(f"  '{plan}': {count} students")
    
    def get_clean_students_table(self) -> pd.DataFrame:
        """
        Get the cleaned students table.
        
        Returns:
            DataFrame with normalized meal plan data
        """
        if self.students_clean is None:
            raise ValueError("No normalized data available. Call normalize_meal_plans() first.")
        
        return self.students_clean.copy()
    
    def get_plan_type_summary(self) -> Dict:
        """
        Get summary of plan types and their characteristics.
        
        Returns:
            Dictionary with plan type summaries
        """
        if self.students_clean is None:
            raise ValueError("No normalized data available. Call normalize_meal_plans() first.")
        
        summary = {}
        
        # Group by plan characteristics
        plan_groups = self.students_clean.groupby(['unlimited', 'is_flex']).size()
        
        for (unlimited, is_flex), count in plan_groups.items():
            plan_type = []
            if unlimited:
                plan_type.append("Unlimited")
            else:
                plan_type.append("Limited")
            
            if is_flex:
                plan_type.append("Flex")
            else:
                plan_type.append("Fixed")
            
            plan_name = " ".join(plan_type)
            summary[plan_name] = {
                'count': count,
                'unlimited': unlimited,
                'is_flex': is_flex
            }
        
        return summary


def main():
    """Example usage of MealPlanNormalizer."""
    from dataloader import DataLoader
    
    # Load data using DataLoader
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Initialize normalizer
    normalizer = MealPlanNormalizer()
    
    # Normalize meal plans
    students_clean = normalizer.normalize_meal_plans(data['students_data'])
    
    # Display results
    print("\n=== Normalized Students Table (first 10 rows) ===")
    print(students_clean.head(10))
    
    print("\n=== Plan Type Summary ===")
    summary = normalizer.get_plan_type_summary()
    for plan_type, details in summary.items():
        print(f"{plan_type}: {details['count']} students")
    
    print("\n=== Sample of Each Plan Type ===")
    for plan_type in students_clean['plan_raw'].unique()[:5]:
        sample = students_clean[students_clean['plan_raw'] == plan_type].iloc[0]
        print(f"'{sample['plan_raw']}' -> weekly_allowance: {sample['weekly_allowance']}, "
              f"unlimited: {sample['unlimited']}, is_flex: {sample['is_flex']}")


if __name__ == "__main__":
    main()
