"""
MLModelTrainer module for dining demand optimization system.

This module trains a machine learning model to predict per-bin swipe probabilities,
with proper calibration for meal plan budgets and class conflicts.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib

from feature_builder import FeatureBuilder
from time_grid_service import TimeGridService, MealPeriod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML model training."""
    model_type: str = 'gradient_boosting'  # 'logistic' or 'gradient_boosting'
    test_size: float = 0.2
    random_state: int = 42
    calibration_method: str = 'isotonic'  # 'isotonic' or 'sigmoid'
    
    # Gradient boosting parameters
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    
    # Logistic regression parameters
    C: float = 1.0
    max_iter: int = 1000


class MLModelTrainer:
    """Trainer for ML model to predict per-bin swipe probabilities."""
    
    def __init__(self, 
                 feature_builder: FeatureBuilder,
                 time_grid_service: TimeGridService,
                 config: Optional[ModelConfig] = None):
        """
        Initialize MLModelTrainer.
        
        Args:
            feature_builder: FeatureBuilder instance
            time_grid_service: TimeGridService instance
            config: Model configuration
        """
        self.feature_builder = feature_builder
        self.time_grid = time_grid_service
        self.config = config or ModelConfig()
        
        self.model = None
        self.calibrated_model = None
        self.feature_columns = None
        self.training_data = None
        self.meal_plan_constraints = {}
        
        logger.info(f"Initialized MLModelTrainer with {self.config.model_type} model")
    
    def prepare_training_data(self, student_ids: List[str]) -> pd.DataFrame:
        """
        Prepare training data with proper weak labeling.
        
        Args:
            student_ids: List of student IDs to include
            
        Returns:
            Prepared training DataFrame
        """
        logger.info(f"Preparing training data for {len(student_ids)} students...")
        
        # Get base training matrix from FeatureBuilder
        base_df = self.feature_builder.build_training_matrix(student_ids)
        
        # Create enhanced weak labels
        enhanced_df = self._create_enhanced_weak_labels(base_df)
        
        # Store feature columns for later use
        self.feature_columns = [col for col in enhanced_df.columns 
                               if col not in ['student_id', 'bin_index', 'target_probability']]
        
        self.training_data = enhanced_df
        
        logger.info(f"Prepared training data with {len(enhanced_df)} examples")
        self._log_training_data_summary(enhanced_df)
        
        return enhanced_df
    
    def _create_enhanced_weak_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced weak labels with proper gap weighting and constraints.
        
        Args:
            df: Base training DataFrame
            
        Returns:
            DataFrame with enhanced weak labels
        """
        df = df.copy()
        
        # Initialize enhanced target column
        df['enhanced_target'] = 0.0
        
        # Process each student separately
        for student_id in df['student_id'].unique():
            student_mask = df['student_id'] == student_id
            student_data = df[student_mask].copy()
            
            # Get student's meal propensity
            propensity = self.feature_builder.meal_propensity.get_student_propensity(student_id)
            if propensity is None:
                continue
            
            # Process each meal period
            for meal_period in ['Breakfast', 'Lunch', 'Dinner']:
                meal_mask = student_data['meal_period_bin'] == meal_period
                meal_data = student_data[meal_mask].copy()
                
                if len(meal_data) == 0:
                    continue
                
                # Get base propensity for this meal period
                if meal_period == 'Breakfast':
                    base_propensity = propensity.p_breakfast
                elif meal_period == 'Lunch':
                    base_propensity = propensity.p_lunch
                else:  # Dinner
                    base_propensity = propensity.p_dinner
                
                # Calculate weights based on free gaps
                meal_data['gap_weight'] = self._calculate_gap_weight(
                    meal_data['free_gap_before_min'], 
                    meal_data['free_gap_after_min']
                )
                
                # Calculate distance weights (closer to meal center = higher weight)
                meal_data['distance_weight'] = self._calculate_distance_weight(
                    meal_data['minutes_from_meal_center']
                )
                
                # Calculate combined weights
                meal_data['combined_weight'] = (
                    meal_data['gap_weight'] * meal_data['distance_weight']
                )
                
                # Normalize weights so they sum to 1
                total_weight = meal_data['combined_weight'].sum()
                if total_weight > 0:
                    meal_data['normalized_weight'] = meal_data['combined_weight'] / total_weight
                else:
                    meal_data['normalized_weight'] = 1.0 / len(meal_data)
                
                # Calculate enhanced target probabilities
                meal_data['enhanced_target'] = (
                    base_propensity * meal_data['normalized_weight']
                )
                
                # Apply class conflict constraint (0 if in class)
                meal_data.loc[meal_data['has_class_in_bin'], 'enhanced_target'] = 0.0
                
                # Update the main dataframe
                df.loc[student_mask & meal_mask, 'enhanced_target'] = meal_data['enhanced_target']
        
        # Replace original target with enhanced target
        df['target_probability'] = df['enhanced_target']
        df = df.drop('enhanced_target', axis=1)
        
        return df
    
    def _calculate_gap_weight(self, gap_before: pd.Series, gap_after: pd.Series) -> pd.Series:
        """Calculate weight based on free time gaps."""
        total_gap = gap_before + gap_after
        
        # Normalize to 0-1, with diminishing returns after 2 hours
        weight = np.minimum(1.0, total_gap / 120.0)
        
        # Add bonus for very large gaps
        weight = np.where(total_gap >= 180, weight * 1.2, weight)
        
        return weight
    
    def _calculate_distance_weight(self, minutes_from_center: pd.Series) -> pd.Series:
        """Calculate weight based on distance from meal center."""
        # Decay function: higher weight closer to center
        weight = np.maximum(0.1, 1.0 - (minutes_from_center / 180.0))
        
        return weight
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for ML model."""
        X_encoded = X.copy()
        
        # Encode meal_period_bin
        if 'meal_period_bin' in X_encoded.columns:
            meal_period_mapping = {
                'Breakfast': 1,
                'Lunch': 2, 
                'Dinner': 3,
                'None': 0
            }
            X_encoded['meal_period_bin'] = X_encoded['meal_period_bin'].map(meal_period_mapping)
        
        # Convert boolean columns to int
        bool_columns = ['unlimited', 'is_flex', 'has_class_in_bin', 'is_large_gap']
        for col in bool_columns:
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].astype(int)
        
        return X_encoded
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """
        Train the ML model.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training ML model...")
        
        # Set feature columns if not already set
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns 
                                  if col not in ['student_id', 'bin_index', 'target_probability']]
        
        # Prepare features and target
        X = df[self.feature_columns].copy()
        
        # Encode categorical features
        X = self._encode_categorical_features(X)
        
        y = df['target_probability']
        
        # Create binary target for classification (probability > 0)
        y_binary = (y > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test, y_train_prob, y_test_prob = train_test_split(
            X, y_binary, y, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=y_binary
        )
        
        # Train base model
        if self.config.model_type == 'logistic':
            self.model = LogisticRegression(
                C=self.config.C, 
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            )
        else:  # gradient_boosting
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state
            )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # For probability estimation, we'll use the model's predict_proba directly
        # and apply our own calibration in the prediction method
        self.calibrated_model = self.model
        
        # Evaluate model
        train_pred = self.calibrated_model.predict_proba(X_train)[:, 1]
        test_pred = self.calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_auc = roc_auc_score(y_train, train_pred)
        test_auc = roc_auc_score(y_test, test_pred)
        
        # For log loss, we need to use binary targets
        train_logloss = log_loss(y_train, train_pred)
        test_logloss = log_loss(y_test, test_pred)
        
        results = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_logloss': train_logloss,
            'test_logloss': test_logloss,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': self._get_feature_importance()
        }
        
        logger.info(f"Model training completed - Test AUC: {test_auc:.3f}, Test LogLoss: {test_logloss:.3f}")
        
        return results
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Gradient boosting
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Logistic regression
            importance = np.abs(self.model.coef_[0])
        else:
            return {}
        
        return dict(zip(self.feature_columns, importance))
    
    def calibrate_for_meal_plans(self, df: pd.DataFrame) -> None:
        """
        Apply meal plan budget constraints to the model.
        
        Args:
            df: Training DataFrame with meal plan information
        """
        logger.info("Calibrating model for meal plan constraints...")
        
        # Calculate daily meal budget constraints
        for student_id in df['student_id'].unique():
            student_data = df[df['student_id'] == student_id]
            
            # Get meal plan info
            weekly_allowance = student_data['weekly_allowance'].iloc[0]
            unlimited = student_data['unlimited'].iloc[0]
            
            if unlimited:
                # Unlimited plans: no constraint
                daily_budget = 7.0  # Assume 7 meals per day max
            else:
                # Limited plans: weekly_allowance / 7 per day
                daily_budget = weekly_allowance / 7.0
            
            self.meal_plan_constraints[student_id] = daily_budget
        
        logger.info(f"Applied meal plan constraints for {len(self.meal_plan_constraints)} students")
    
    def predict_swipe_probability(self, 
                                 student_id: str, 
                                 bin_index: int, 
                                 features: Optional[Dict] = None) -> float:
        """
        Predict swipe probability for a specific student and bin.
        
        Args:
            student_id: Student identifier
            bin_index: Time bin index
            features: Optional feature dictionary (if None, will be calculated)
            
        Returns:
            Predicted swipe probability (0.0 to 1.0)
        """
        if self.calibrated_model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if features is None:
            # Calculate features using FeatureBuilder
            features = self._calculate_features_for_prediction(student_id, bin_index)
        
        # Prepare feature vector
        feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # Get base prediction
        base_prob = self.calibrated_model.predict_proba(feature_vector)[0, 1]
        
        # Apply meal plan constraints
        constrained_prob = self._apply_meal_plan_constraints(student_id, bin_index, base_prob)
        
        return constrained_prob
    
    def _calculate_features_for_prediction(self, student_id: str, bin_index: int) -> Dict:
        """Calculate features for a specific student and bin."""
        # This would integrate with FeatureBuilder to get features
        # For now, return a simplified version
        bin_obj = self.time_grid.get_bin(bin_index)
        
        return {
            'meal_period_bin': 1 if bin_obj.meal_period.value == 'Breakfast' else 
                             2 if bin_obj.meal_period.value == 'Lunch' else
                             3 if bin_obj.meal_period.value == 'Dinner' else 0,
            'minutes_from_meal_center': self._calculate_minutes_from_meal_center(
                bin_obj.start_minutes, bin_obj.meal_period.value
            ),
            'bin_index': bin_index,
            'bin_index_normalized': bin_index / len(self.time_grid.time_bins),
            'p_breakfast': 0.333,  # Would get from MealPropensityService
            'p_lunch': 0.333,
            'p_dinner': 0.333,
            'weekly_allowance': 14.0,  # Would get from MealPlanNormalizer
            'unlimited': False,
            'is_flex': False,
            'has_class_in_bin': False,  # Would get from AvailabilityService
            'free_gap_before_min': 0,
            'free_gap_after_min': 0,
            'is_large_gap': False
        }
    
    def _calculate_minutes_from_meal_center(self, bin_start_minutes: int, meal_period: str) -> int:
        """Calculate minutes from meal center."""
        meal_centers = {
            'Breakfast': 8 * 60 + 45,  # 08:45
            'Lunch': 12 * 60 + 45,     # 12:45
            'Dinner': 18 * 60 + 45     # 18:45
        }
        
        if meal_period in meal_centers:
            return abs(bin_start_minutes - meal_centers[meal_period])
        return 0
    
    def _apply_meal_plan_constraints(self, 
                                   student_id: str, 
                                   bin_index: int, 
                                   base_prob: float) -> float:
        """Apply meal plan budget constraints to probability."""
        if student_id not in self.meal_plan_constraints:
            return base_prob
        
        daily_budget = self.meal_plan_constraints[student_id]
        
        # Scale probability based on daily budget
        # Higher budget = higher probability
        budget_factor = min(1.0, daily_budget / 3.0)  # Normalize to 3 meals per day
        
        constrained_prob = base_prob * budget_factor
        
        return min(1.0, constrained_prob)
    
    def predict_demand_matrix(self, student_ids: List[str]) -> pd.DataFrame:
        """
        Predict demand matrix for all students and bins.
        
        Args:
            student_ids: List of student IDs
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Predicting demand matrix for {len(student_ids)} students...")
        
        predictions = []
        
        for student_id in student_ids:
            for bin_idx in range(len(self.time_grid.time_bins)):
                prob = self.predict_swipe_probability(student_id, bin_idx)
                
                predictions.append({
                    'student_id': student_id,
                    'bin_index': bin_idx,
                    'swipe_probability': prob
                })
        
        df = pd.DataFrame(predictions)
        
        logger.info(f"Generated {len(df)} predictions")
        return df
    
    def _log_training_data_summary(self, df: pd.DataFrame) -> None:
        """Log summary of training data."""
        logger.info("=== Training Data Summary ===")
        logger.info(f"Total examples: {len(df)}")
        logger.info(f"Unique students: {df['student_id'].nunique()}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        
        # Target distribution
        target_stats = df['target_probability'].describe()
        logger.info(f"Target probability - Mean: {target_stats['mean']:.3f}, "
                   f"Std: {target_stats['std']:.3f}, "
                   f"Max: {target_stats['max']:.3f}")
        
        # Class distribution
        positive_examples = (df['target_probability'] > 0).sum()
        logger.info(f"Positive examples: {positive_examples} ({positive_examples/len(df)*100:.1f}%)")
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if self.calibrated_model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_data = {
            'calibrated_model': self.calibrated_model,
            'feature_columns': self.feature_columns,
            'meal_plan_constraints': self.meal_plan_constraints,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.calibrated_model = model_data['calibrated_model']
        self.feature_columns = model_data['feature_columns']
        self.meal_plan_constraints = model_data['meal_plan_constraints']
        self.config = model_data['config']
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """Example usage of MLModelTrainer."""
    from dataloader import DataLoader
    from time_grid_service import TimeGridService
    from meal_propensity_service import MealPropensityService
    from availability_service import AvailabilityService
    from sections_catalog import SectionsCatalog
    from feature_builder import FeatureBuilder
    
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
    
    # Create and configure trainer
    config = ModelConfig(
        model_type='gradient_boosting',
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1
    )
    
    trainer = MLModelTrainer(feature_builder, time_grid, config)
    
    # Prepare training data
    training_df = trainer.prepare_training_data(student_ids)
    
    # Train model
    results = trainer.train_model(training_df)
    
    print("=== ML Model Trainer Demo ===")
    print(f"Training Results:")
    print(f"  Train AUC: {results['train_auc']:.3f}")
    print(f"  Test AUC: {results['test_auc']:.3f}")
    print(f"  Train LogLoss: {results['train_logloss']:.3f}")
    print(f"  Test LogLoss: {results['test_logloss']:.3f}")
    print()
    
    # Show feature importance
    print("Top Feature Importance:")
    importance = results['feature_importance']
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_features[:10]:
        print(f"  {feature}: {imp:.3f}")
    print()
    
    # Test predictions
    print("Sample Predictions:")
    sample_student = student_ids[0]
    for bin_idx in [0, 10, 20, 30, 40, 50]:
        prob = trainer.predict_swipe_probability(sample_student, bin_idx)
        bin_obj = time_grid.get_bin(bin_idx)
        print(f"  Student {sample_student}, Bin {bin_idx} ({bin_obj}): {prob:.3f}")
    print()
    
    # Generate demand matrix for subset
    print("Generating demand matrix for first 3 students...")
    demand_df = trainer.predict_demand_matrix(student_ids[:3])
    print(f"Demand matrix shape: {demand_df.shape}")
    print(f"Average probability: {demand_df['swipe_probability'].mean():.3f}")
    print(f"Max probability: {demand_df['swipe_probability'].max():.3f}")


if __name__ == "__main__":
    main()
