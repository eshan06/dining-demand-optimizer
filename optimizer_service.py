"""
OptimizerService module for dining demand optimization system.

This module implements the optimization logic for schedule improvement using
precomputed impact vectors to solve MILP/MIQP optimization problems.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Enumeration of optimization objectives."""
    PEAK_MINIMIZATION = "peak_minimization"
    SMOOTH_MINIMIZATION = "smooth_minimization"


@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters."""
    objective: OptimizationObjective = OptimizationObjective.PEAK_MINIMIZATION
    max_shift_minutes: int = 60
    penalty_per_minute: float = 0.0  # Optional penalty for minutes moved
    time_limit_seconds: int = 300  # 5 minutes default
    solver_tolerance: float = 1e-6
    verbose: bool = False


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    success: bool
    objective_value: float
    chosen_slots: Dict[str, int]  # section_id -> slot_minutes
    final_demand: Dict[int, float]  # bin_index -> demand
    metrics: Dict[str, float]
    solve_time: float
    message: str


class OptimizerService:
    """
    Service for optimizing dining schedules using precomputed impact vectors.
    
    This service solves the optimization problem:
    - Variables: x_{s,t} ∈ {0,1} for section s choosing slot t
    - Demand: D(b) = D_base(b) + Σ Δ_b^(s,t) x_{s,t}
    - Objective: Peak minimization (MILP) or Smooth minimization (MIQP)
    - Constraints: Σ_t x_{s,t}=1 per section
    """
    
    def __init__(self, impact_vector_precomputer, time_grid_service):
        """
        Initialize the OptimizerService.
        
        Args:
            impact_vector_precomputer: Precomputed impact vectors
            time_grid_service: Time grid service for bin information
        """
        self.ivp = impact_vector_precomputer
        self.time_grid = time_grid_service
        self.config = OptimizationConfig()
        
        # Check if impact vectors are precomputed
        if not self.ivp.is_precomputed:
            raise ValueError("Impact vectors must be precomputed before optimization")
    
    def optimize_schedule(self, config: Optional[OptimizationConfig] = None) -> OptimizationResult:
        """
        Optimize the dining schedule based on the specified objective.
        
        Args:
            config: Optimization configuration (uses default if None)
            
        Returns:
            OptimizationResult with chosen slots and metrics
        """
        if config is not None:
            self.config = config
        
        logger.info(f"Starting optimization with objective: {self.config.objective.value}")
        
        try:
            if self.config.objective == OptimizationObjective.PEAK_MINIMIZATION:
                return self._optimize_peak_minimization()
            elif self.config.objective == OptimizationObjective.SMOOTH_MINIMIZATION:
                return self._optimize_smooth_minimization()
            else:
                raise ValueError(f"Unknown objective: {self.config.objective}")
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                success=False,
                objective_value=float('inf'),
                chosen_slots={},
                final_demand={},
                metrics={},
                solve_time=0.0,
                message=f"Optimization failed: {e}"
            )
    
    def _optimize_peak_minimization(self) -> OptimizationResult:
        """
        Optimize for peak demand minimization (MILP).
        
        Minimize Z subject to:
        - D(b) ≤ Z for all bins b
        - Σ_t x_{s,t} = 1 for all sections s
        - x_{s,t} ∈ {0,1}
        """
        logger.info("Solving peak minimization problem (MILP)")
        
        # Get available sections and their allowed slots
        sections_data = self._get_sections_and_slots()
        if not sections_data:
            return OptimizationResult(
                success=False,
                objective_value=float('inf'),
                chosen_slots={},
                final_demand={},
                metrics={},
                solve_time=0.0,
                message="No sections available for optimization"
            )
        
        # For now, implement a greedy heuristic approach
        # In production, this would use a proper MILP solver like Gurobi or CPLEX
        chosen_slots, final_demand, metrics = self._greedy_peak_optimization(sections_data)
        
        # Calculate objective value (peak demand)
        objective_value = max(final_demand.values()) if final_demand else float('inf')
        
        return OptimizationResult(
            success=True,
            objective_value=objective_value,
            chosen_slots=chosen_slots,
            final_demand=final_demand,
            metrics=metrics,
            solve_time=0.0,  # Placeholder for actual solve time
            message="Peak minimization completed (greedy heuristic)"
        )
    
    def _optimize_smooth_minimization(self) -> OptimizationResult:
        """
        Optimize for smooth demand minimization (MIQP).
        
        Minimize Σ_b D(b)^2 subject to:
        - Σ_t x_{s,t} = 1 for all sections s
        - x_{s,t} ∈ {0,1}
        """
        logger.info("Solving smooth minimization problem (MIQP)")
        
        # Get available sections and their allowed slots
        sections_data = self._get_sections_and_slots()
        if not sections_data:
            return OptimizationResult(
                success=False,
                objective_value=float('inf'),
                chosen_slots={},
                final_demand={},
                metrics={},
                solve_time=0.0,
                message="No sections available for optimization"
            )
        
        # For now, implement a greedy heuristic approach
        # In production, this would use a proper MIQP solver
        chosen_slots, final_demand, metrics = self._greedy_smooth_optimization(sections_data)
        
        # Calculate objective value (sum of squared demands)
        objective_value = sum(demand**2 for demand in final_demand.values()) if final_demand else float('inf')
        
        return OptimizationResult(
            success=True,
            objective_value=objective_value,
            chosen_slots=chosen_slots,
            final_demand=final_demand,
            metrics=metrics,
            solve_time=0.0,  # Placeholder for actual solve time
            message="Smooth minimization completed (greedy heuristic)"
        )
    
    def _get_sections_and_slots(self) -> Dict[str, List[int]]:
        """
        Get available sections and their allowed slots.
        
        Returns:
            Dictionary mapping section_id to list of allowed slot_minutes
        """
        sections_data = {}
        
        # Get all sections from the impact vector precomputer
        for section_id in self.ivp.impact_vectors.keys():
            allowed_slots = []
            
            # Get all slots for this section
            for slot_minutes in self.ivp.impact_vectors[section_id].keys():
                allowed_slots.append(slot_minutes)
            
            if allowed_slots:
                sections_data[section_id] = allowed_slots
        
        logger.info(f"Found {len(sections_data)} sections with allowed slots")
        return sections_data
    
    def _greedy_peak_optimization(self, sections_data: Dict[str, List[int]]) -> Tuple[Dict[str, int], Dict[int, float], Dict[str, float]]:
        """
        Greedy heuristic for peak minimization.
        
        Args:
            sections_data: Dictionary mapping section_id to allowed slots
            
        Returns:
            Tuple of (chosen_slots, final_demand, metrics)
        """
        # Start with baseline demand
        final_demand = {bin_idx: bin_demand.expected_demand 
                       for bin_idx, bin_demand in self.ivp.baseline_demand.items()}
        
        chosen_slots = {}
        
        # Process sections in order (could be randomized for better results)
        for section_id, allowed_slots in sections_data.items():
            best_slot = None
            best_peak = float('inf')
            
            # Try each allowed slot for this section
            for slot_minutes in allowed_slots:
                # Calculate what the peak would be if we chose this slot
                temp_demand = final_demand.copy()
                
                # Apply impact vectors for this section-slot combination
                impact_vectors = self.ivp.get_impact_vector(section_id, slot_minutes)
                for vector in impact_vectors:
                    temp_demand[vector.bin_index] += vector.demand_delta
                
                # Calculate peak demand
                peak_demand = max(temp_demand.values()) if temp_demand else float('inf')
                
                if peak_demand < best_peak:
                    best_peak = peak_demand
                    best_slot = slot_minutes
            
            # Choose the best slot for this section
            if best_slot is not None:
                chosen_slots[section_id] = best_slot
                
                # Update final demand
                impact_vectors = self.ivp.get_impact_vector(section_id, best_slot)
                for vector in impact_vectors:
                    final_demand[vector.bin_index] += vector.demand_delta
        
        # Calculate metrics
        metrics = self._calculate_metrics(chosen_slots, final_demand)
        
        return chosen_slots, final_demand, metrics
    
    def _greedy_smooth_optimization(self, sections_data: Dict[str, List[int]]) -> Tuple[Dict[str, int], Dict[int, float], Dict[str, float]]:
        """
        Greedy heuristic for smooth minimization.
        
        Args:
            sections_data: Dictionary mapping section_id to allowed slots
            
        Returns:
            Tuple of (chosen_slots, final_demand, metrics)
        """
        # Start with baseline demand
        final_demand = {bin_idx: bin_demand.expected_demand 
                       for bin_idx, bin_demand in self.ivp.baseline_demand.items()}
        
        chosen_slots = {}
        
        # Process sections in order (could be randomized for better results)
        for section_id, allowed_slots in sections_data.items():
            best_slot = None
            best_smoothness = float('inf')
            
            # Try each allowed slot for this section
            for slot_minutes in allowed_slots:
                # Calculate what the smoothness would be if we chose this slot
                temp_demand = final_demand.copy()
                
                # Apply impact vectors for this section-slot combination
                impact_vectors = self.ivp.get_impact_vector(section_id, slot_minutes)
                for vector in impact_vectors:
                    temp_demand[vector.bin_index] += vector.demand_delta
                
                # Calculate smoothness (sum of squared demands)
                smoothness = sum(demand**2 for demand in temp_demand.values()) if temp_demand else float('inf')
                
                if smoothness < best_smoothness:
                    best_smoothness = smoothness
                    best_slot = slot_minutes
            
            # Choose the best slot for this section
            if best_slot is not None:
                chosen_slots[section_id] = best_slot
                
                # Update final demand
                impact_vectors = self.ivp.get_impact_vector(section_id, best_slot)
                for vector in impact_vectors:
                    final_demand[vector.bin_index] += vector.demand_delta
        
        # Calculate metrics
        metrics = self._calculate_metrics(chosen_slots, final_demand)
        
        return chosen_slots, final_demand, metrics
    
    def _calculate_metrics(self, chosen_slots: Dict[str, int], final_demand: Dict[int, float]) -> Dict[str, float]:
        """
        Calculate optimization metrics.
        
        Args:
            chosen_slots: Dictionary mapping section_id to chosen slot_minutes
            final_demand: Dictionary mapping bin_index to final demand
            
        Returns:
            Dictionary of metrics
        """
        if not final_demand:
            return {}
        
        demand_values = list(final_demand.values())
        
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
        total_shifts = 0
        total_shift_minutes = 0
        sections_moved = 0
        
        for section_id, chosen_slot in chosen_slots.items():
            # Get original slot (assuming 0 is the original slot)
            original_slot = 0  # This would need to be tracked from the original schedule
            if chosen_slot != original_slot:
                sections_moved += 1
                total_shift_minutes += abs(chosen_slot - original_slot)
                total_shifts += 1
        
        avg_shift_minutes = total_shift_minutes / total_shifts if total_shifts > 0 else 0
        percent_moved = (sections_moved / len(chosen_slots)) * 100 if chosen_slots else 0
        
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
            'total_shifts': total_shifts
        }
    
    def get_optimization_summary(self, result: OptimizationResult) -> str:
        """
        Get a human-readable summary of optimization results.
        
        Args:
            result: OptimizationResult object
            
        Returns:
            Formatted summary string
        """
        if not result.success:
            return f"Optimization failed: {result.message}"
        
        summary = f"""
Optimization Results:
====================
Objective: {self.config.objective.value}
Success: {result.success}
Objective Value: {result.objective_value:.2f}
Solve Time: {result.solve_time:.2f}s

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

Message: {result.message}
"""
        return summary


def main():
    """Test the OptimizerService with actual data."""
    logger.info("Testing OptimizerService...")
    
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
        
        # Initialize optimizer
        optimizer = OptimizerService(ivp, time_grid)
        
        # Test peak minimization
        logger.info("Testing peak minimization...")
        peak_config = OptimizationConfig(
            objective=OptimizationObjective.PEAK_MINIMIZATION,
            max_shift_minutes=60,
            penalty_per_minute=0.1
        )
        
        peak_result = optimizer.optimize_schedule(peak_config)
        print(optimizer.get_optimization_summary(peak_result))
        
        # Test smooth minimization
        logger.info("Testing smooth minimization...")
        smooth_config = OptimizationConfig(
            objective=OptimizationObjective.SMOOTH_MINIMIZATION,
            max_shift_minutes=60,
            penalty_per_minute=0.1
        )
        
        smooth_result = optimizer.optimize_schedule(smooth_config)
        print(optimizer.get_optimization_summary(smooth_result))
        
        logger.info("OptimizerService test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
