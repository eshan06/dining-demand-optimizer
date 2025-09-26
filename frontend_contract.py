"""
Frontend Interaction Contract module for dining demand optimization system.

This module defines the backend expectations for frontend UI interactions,
enabling responsive side-by-side curves and KPIs that change with user controls.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Enumeration of optimization objectives."""
    PEAK_MINIMIZATION = "peak_minimization"
    SMOOTH_MINIMIZATION = "smooth_minimization"


@dataclass
class FrontendInputs:
    """Frontend input parameters from sliders and toggles."""
    bin_size_min: int = 15
    max_shift_min: int = 60
    meal_windows: Dict[str, Tuple[int, int]] = None  # {"breakfast": (420, 630), "lunch": (630, 900), "dinner": (900, 1200)}
    objective: str = "peak_minimization"  # "peak_minimization" or "smooth_minimization"
    smooth_weight: Optional[float] = None  # Only used for smooth_minimization
    
    def __post_init__(self):
        """Initialize default meal windows if not provided."""
        if self.meal_windows is None:
            self.meal_windows = {
                "breakfast": (420, 630),  # 7:00 AM - 10:30 AM
                "lunch": (630, 900),      # 10:30 AM - 3:00 PM
                "dinner": (900, 1200)     # 3:00 PM - 8:00 PM
            }


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""
    bin_index: int
    time_label: str  # e.g., "09:00", "09:15"
    demand: float
    meal_period: str  # "breakfast", "lunch", "dinner", "none"


@dataclass
class AlignedSeries:
    """Aligned time series for current vs optimized demand."""
    current_series: List[TimeSeriesPoint]
    optimized_series: List[TimeSeriesPoint]
    time_labels: List[str]
    bin_indices: List[int]


@dataclass
class MealStack:
    """Per-meal demand breakdown."""
    meal_period: str
    current_demand: float
    optimized_demand: float
    change: float  # optimized - current
    change_percent: float  # (change / current) * 100


@dataclass
class KPIMetrics:
    """Key Performance Indicators for demand optimization."""
    # Current metrics
    current_peak_demand: float
    current_p95_demand: float
    current_mean_demand: float
    current_variance: float
    current_total_demand: float
    
    # Optimized metrics
    optimized_peak_demand: float
    optimized_p95_demand: float
    optimized_mean_demand: float
    optimized_variance: float
    optimized_total_demand: float
    
    # Improvement metrics
    peak_reduction: float
    peak_reduction_percent: float
    p95_reduction: float
    p95_reduction_percent: float
    variance_reduction: float
    variance_reduction_percent: float
    
    # Optimization metrics
    sections_moved: int
    total_sections: int
    percent_moved: float
    total_shift_minutes: int
    avg_shift_minutes: float
    objective_value: float
    solve_time: float


@dataclass
class AffectedSection:
    """Information about a section that was moved during optimization."""
    section_id: str
    old_start_time: str  # e.g., "09:00"
    new_start_time: str  # e.g., "10:15"
    old_start_minutes: int
    new_start_minutes: int
    minutes_moved: int
    enrolled_count: int
    course_info: Optional[str] = None  # Optional course name or code


@dataclass
class FrontendResponse:
    """Complete response for frontend UI."""
    success: bool
    message: str
    
    # Time series data
    aligned_series: AlignedSeries
    meal_stacks: List[MealStack]
    
    # Metrics
    kpi_metrics: KPIMetrics
    
    # Affected sections
    affected_sections: List[AffectedSection]
    
    # Metadata
    computation_time: float
    timestamp: float
    input_parameters: FrontendInputs


class FrontendContractValidator:
    """Validates frontend inputs against backend constraints."""
    
    def __init__(self):
        """Initialize the validator with default constraints."""
        self.min_bin_size = 5
        self.max_bin_size = 60
        self.min_shift_minutes = 15
        self.max_shift_minutes = 180
        self.min_smooth_weight = 0.0
        self.max_smooth_weight = 1.0
    
    def validate_inputs(self, inputs: FrontendInputs) -> Tuple[bool, List[str]]:
        """
        Validate frontend inputs.
        
        Args:
            inputs: FrontendInputs object to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate bin_size_min
        if not (self.min_bin_size <= inputs.bin_size_min <= self.max_bin_size):
            errors.append(f"bin_size_min must be between {self.min_bin_size} and {self.max_bin_size} minutes")
        
        # Validate max_shift_min
        if not (self.min_shift_minutes <= inputs.max_shift_min <= self.max_shift_minutes):
            errors.append(f"max_shift_min must be between {self.min_shift_minutes} and {self.max_shift_minutes} minutes")
        
        # Validate objective
        if inputs.objective not in ["peak_minimization", "smooth_minimization"]:
            errors.append("objective must be 'peak_minimization' or 'smooth_minimization'")
        
        # Validate smooth_weight if provided
        if inputs.smooth_weight is not None:
            if not (self.min_smooth_weight <= inputs.smooth_weight <= self.max_smooth_weight):
                errors.append(f"smooth_weight must be between {self.min_smooth_weight} and {self.max_smooth_weight}")
        
        # Validate meal windows
        if inputs.meal_windows:
            for meal, (start, end) in inputs.meal_windows.items():
                if start >= end:
                    errors.append(f"meal window {meal} start time must be before end time")
                if start < 0 or end > 1440:  # 24 hours in minutes
                    errors.append(f"meal window {meal} times must be between 0 and 1440 minutes")
        
        return len(errors) == 0, errors


class FrontendResponseBuilder:
    """Builds frontend responses from backend data."""
    
    def __init__(self, time_grid_service, sections_catalog):
        """
        Initialize the response builder.
        
        Args:
            time_grid_service: TimeGridService instance
            sections_catalog: SectionsCatalog instance
        """
        self.time_grid = time_grid_service
        self.sections_catalog = sections_catalog
    
    def build_response(self, 
                      inputs: FrontendInputs,
                      current_demand: Dict[int, float],
                      optimized_demand: Dict[int, float],
                      optimization_result: Dict[str, Any],
                      computation_time: float) -> FrontendResponse:
        """
        Build a complete frontend response.
        
        Args:
            inputs: Frontend input parameters
            current_demand: Current demand series (bin_index -> demand)
            optimized_demand: Optimized demand series (bin_index -> demand)
            optimization_result: Result from optimization service
            computation_time: Total computation time
            
        Returns:
            Complete FrontendResponse object
        """
        try:
            # Build aligned series
            aligned_series = self._build_aligned_series(current_demand, optimized_demand)
            
            # Build meal stacks
            meal_stacks = self._build_meal_stacks(current_demand, optimized_demand, inputs.meal_windows)
            
            # Build KPI metrics
            kpi_metrics = self._build_kpi_metrics(current_demand, optimized_demand, optimization_result)
            
            # Build affected sections
            affected_sections = self._build_affected_sections(optimization_result)
            
            return FrontendResponse(
                success=True,
                message="Optimization completed successfully",
                aligned_series=aligned_series,
                meal_stacks=meal_stacks,
                kpi_metrics=kpi_metrics,
                affected_sections=affected_sections,
                computation_time=computation_time,
                timestamp=time.time(),
                input_parameters=inputs
            )
            
        except Exception as e:
            logger.error(f"Error building frontend response: {e}")
            return FrontendResponse(
                success=False,
                message=f"Error building response: {e}",
                aligned_series=self._build_empty_aligned_series(),
                meal_stacks=[],
                kpi_metrics=self._build_empty_kpi_metrics(),
                affected_sections=[],
                computation_time=computation_time,
                timestamp=time.time(),
                input_parameters=inputs
            )
    
    def _build_aligned_series(self, current_demand: Dict[int, float], optimized_demand: Dict[int, float]) -> AlignedSeries:
        """Build aligned time series for current vs optimized demand."""
        # Get all time bins
        time_bins = self.time_grid.time_bins
        
        current_series = []
        optimized_series = []
        time_labels = []
        bin_indices = []
        
        for bin_obj in time_bins:
            bin_idx = bin_obj.index
            time_label = self._format_time_label(bin_obj.start_minutes)
            meal_period = self._get_meal_period_label(bin_obj.meal_period)
            
            current_demand_val = current_demand.get(bin_idx, 0.0)
            optimized_demand_val = optimized_demand.get(bin_idx, 0.0)
            
            current_series.append(TimeSeriesPoint(
                bin_index=bin_idx,
                time_label=time_label,
                demand=current_demand_val,
                meal_period=meal_period
            ))
            
            optimized_series.append(TimeSeriesPoint(
                bin_index=bin_idx,
                time_label=time_label,
                demand=optimized_demand_val,
                meal_period=meal_period
            ))
            
            time_labels.append(time_label)
            bin_indices.append(bin_idx)
        
        return AlignedSeries(
            current_series=current_series,
            optimized_series=optimized_series,
            time_labels=time_labels,
            bin_indices=bin_indices
        )
    
    def _build_meal_stacks(self, current_demand: Dict[int, float], optimized_demand: Dict[int, float], meal_windows: Dict[str, Tuple[int, int]]) -> List[MealStack]:
        """Build per-meal demand breakdown."""
        meal_stacks = []
        
        for meal_period, (start_min, end_min) in meal_windows.items():
            current_total = 0.0
            optimized_total = 0.0
            
            # Sum demand for bins within this meal window
            for bin_idx, bin_obj in enumerate(self.time_grid.time_bins):
                if start_min <= bin_obj.start_minutes < end_min:
                    current_total += current_demand.get(bin_idx, 0.0)
                    optimized_total += optimized_demand.get(bin_idx, 0.0)
            
            change = optimized_total - current_total
            change_percent = (change / current_total * 100) if current_total > 0 else 0.0
            
            meal_stacks.append(MealStack(
                meal_period=meal_period,
                current_demand=current_total,
                optimized_demand=optimized_total,
                change=change,
                change_percent=change_percent
            ))
        
        return meal_stacks
    
    def _build_kpi_metrics(self, current_demand: Dict[int, float], optimized_demand: Dict[int, float], optimization_result: Dict[str, Any]) -> KPIMetrics:
        """Build KPI metrics for current vs optimized demand."""
        # Calculate current metrics
        current_values = list(current_demand.values())
        current_peak = max(current_values) if current_values else 0.0
        current_p95 = self._calculate_percentile(current_values, 95)
        current_mean = sum(current_values) / len(current_values) if current_values else 0.0
        current_variance = sum((x - current_mean) ** 2 for x in current_values) / len(current_values) if current_values else 0.0
        current_total = sum(current_values)
        
        # Calculate optimized metrics
        optimized_values = list(optimized_demand.values())
        optimized_peak = max(optimized_values) if optimized_values else 0.0
        optimized_p95 = self._calculate_percentile(optimized_values, 95)
        optimized_mean = sum(optimized_values) / len(optimized_values) if optimized_values else 0.0
        optimized_variance = sum((x - optimized_mean) ** 2 for x in optimized_values) / len(optimized_values) if optimized_values else 0.0
        optimized_total = sum(optimized_values)
        
        # Calculate improvements
        peak_reduction = current_peak - optimized_peak
        peak_reduction_percent = (peak_reduction / current_peak * 100) if current_peak > 0 else 0.0
        
        p95_reduction = current_p95 - optimized_p95
        p95_reduction_percent = (p95_reduction / current_p95 * 100) if current_p95 > 0 else 0.0
        
        variance_reduction = current_variance - optimized_variance
        variance_reduction_percent = (variance_reduction / current_variance * 100) if current_variance > 0 else 0.0
        
        # Get optimization metrics
        chosen_slots = optimization_result.get('chosen_slots', {})
        sections_moved = len(chosen_slots)
        total_sections = len(self.sections_catalog.sections_dict)
        percent_moved = (sections_moved / total_sections * 100) if total_sections > 0 else 0.0
        
        # Calculate shift metrics
        total_shift_minutes = 0
        for section_id, new_start_min in chosen_slots.items():
            if section_id in self.sections_catalog.sections_dict:
                section = self.sections_catalog.sections_dict[section_id]
                shift_minutes = abs(new_start_min - section.start_minutes)
                total_shift_minutes += shift_minutes
        
        avg_shift_minutes = total_shift_minutes / sections_moved if sections_moved > 0 else 0.0
        
        return KPIMetrics(
            current_peak_demand=current_peak,
            current_p95_demand=current_p95,
            current_mean_demand=current_mean,
            current_variance=current_variance,
            current_total_demand=current_total,
            optimized_peak_demand=optimized_peak,
            optimized_p95_demand=optimized_p95,
            optimized_mean_demand=optimized_mean,
            optimized_variance=optimized_variance,
            optimized_total_demand=optimized_total,
            peak_reduction=peak_reduction,
            peak_reduction_percent=peak_reduction_percent,
            p95_reduction=p95_reduction,
            p95_reduction_percent=p95_reduction_percent,
            variance_reduction=variance_reduction,
            variance_reduction_percent=variance_reduction_percent,
            sections_moved=sections_moved,
            total_sections=total_sections,
            percent_moved=percent_moved,
            total_shift_minutes=total_shift_minutes,
            avg_shift_minutes=avg_shift_minutes,
            objective_value=optimization_result.get('objective_value', 0.0),
            solve_time=optimization_result.get('solve_time', 0.0)
        )
    
    def _build_affected_sections(self, optimization_result: Dict[str, Any]) -> List[AffectedSection]:
        """Build affected sections table."""
        affected_sections = []
        chosen_slots = optimization_result.get('chosen_slots', {})
        
        for section_id, new_start_min in chosen_slots.items():
            if section_id in self.sections_catalog.sections_dict:
                section = self.sections_catalog.sections_dict[section_id]
                old_start_min = section.start_minutes
                minutes_moved = abs(new_start_min - old_start_min)
                
                affected_sections.append(AffectedSection(
                    section_id=section_id,
                    old_start_time=self._format_time_label(old_start_min),
                    new_start_time=self._format_time_label(new_start_min),
                    old_start_minutes=old_start_min,
                    new_start_minutes=new_start_min,
                    minutes_moved=minutes_moved,
                    enrolled_count=section.student_count,
                    course_info=f"Section {section_id}"  # Could be enhanced with course name
                ))
        
        # Sort by minutes moved (descending)
        affected_sections.sort(key=lambda x: x.minutes_moved, reverse=True)
        return affected_sections
    
    def _format_time_label(self, minutes: int) -> str:
        """Format minutes since midnight to time label."""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def _get_meal_period_label(self, meal_period) -> str:
        """Get meal period label from enum."""
        if meal_period is None:
            return "none"
        return meal_period.value.lower() if hasattr(meal_period, 'value') else str(meal_period).lower()
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]
    
    def _build_empty_aligned_series(self) -> AlignedSeries:
        """Build empty aligned series for error cases."""
        return AlignedSeries(
            current_series=[],
            optimized_series=[],
            time_labels=[],
            bin_indices=[]
        )
    
    def _build_empty_kpi_metrics(self) -> KPIMetrics:
        """Build empty KPI metrics for error cases."""
        return KPIMetrics(
            current_peak_demand=0.0,
            current_p95_demand=0.0,
            current_mean_demand=0.0,
            current_variance=0.0,
            current_total_demand=0.0,
            optimized_peak_demand=0.0,
            optimized_p95_demand=0.0,
            optimized_mean_demand=0.0,
            optimized_variance=0.0,
            optimized_total_demand=0.0,
            peak_reduction=0.0,
            peak_reduction_percent=0.0,
            p95_reduction=0.0,
            p95_reduction_percent=0.0,
            variance_reduction=0.0,
            variance_reduction_percent=0.0,
            sections_moved=0,
            total_sections=0,
            percent_moved=0.0,
            total_shift_minutes=0,
            avg_shift_minutes=0.0,
            objective_value=0.0,
            solve_time=0.0
        )


class FrontendContractService:
    """Main service for handling frontend interactions."""
    
    def __init__(self, api_layer, time_grid_service, sections_catalog):
        """
        Initialize the frontend contract service.
        
        Args:
            api_layer: APILayer instance
            time_grid_service: TimeGridService instance
            sections_catalog: SectionsCatalog instance
        """
        self.api_layer = api_layer
        self.time_grid = time_grid_service
        self.sections_catalog = sections_catalog
        self.validator = FrontendContractValidator()
        self.response_builder = FrontendResponseBuilder(time_grid_service, sections_catalog)
    
    def process_frontend_request(self, inputs: FrontendInputs) -> FrontendResponse:
        """
        Process a frontend request and return formatted response.
        
        Args:
            inputs: Frontend input parameters
            
        Returns:
            Formatted FrontendResponse
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            is_valid, errors = self.validator.validate_inputs(inputs)
            if not is_valid:
                return FrontendResponse(
                    success=False,
                    message=f"Invalid inputs: {', '.join(errors)}",
                    aligned_series=self.response_builder._build_empty_aligned_series(),
                    meal_stacks=[],
                    kpi_metrics=self.response_builder._build_empty_kpi_metrics(),
                    affected_sections=[],
                    computation_time=time.time() - start_time,
                    timestamp=time.time(),
                    input_parameters=inputs
                )
            
            # Get current demand
            current_demand = self._get_current_demand()
            
            # Run optimization
            optimization_result = self._run_optimization(inputs)
            
            if not optimization_result['success']:
                return FrontendResponse(
                    success=False,
                    message=f"Optimization failed: {optimization_result['message']}",
                    aligned_series=self.response_builder._build_empty_aligned_series(),
                    meal_stacks=[],
                    kpi_metrics=self.response_builder._build_empty_kpi_metrics(),
                    affected_sections=[],
                    computation_time=time.time() - start_time,
                    timestamp=time.time(),
                    input_parameters=inputs
                )
            
            # Build response
            response = self.response_builder.build_response(
                inputs=inputs,
                current_demand=current_demand,
                optimized_demand=optimization_result['demand_series'],
                optimization_result=optimization_result,
                computation_time=time.time() - start_time
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing frontend request: {e}")
            return FrontendResponse(
                success=False,
                message=f"Error processing request: {e}",
                aligned_series=self.response_builder._build_empty_aligned_series(),
                meal_stacks=[],
                kpi_metrics=self.response_builder._build_empty_kpi_metrics(),
                affected_sections=[],
                computation_time=time.time() - start_time,
                timestamp=time.time(),
                input_parameters=inputs
            )
    
    def _get_current_demand(self) -> Dict[int, float]:
        """Get current demand series."""
        # Get current demand from the API layer
        current_state = self.api_layer.get_current_state()
        return current_state.get('demand_series', {})
    
    def _run_optimization(self, inputs: FrontendInputs) -> Dict[str, Any]:
        """Run optimization with the given inputs."""
        from api_layer import OptimizeRequest
        
        # Convert meal windows to the format expected by the API
        meal_windows_dict = {}
        if inputs.meal_windows:
            for meal, (start, end) in inputs.meal_windows.items():
                meal_windows_dict[meal] = (start, end)
        
        # Create optimization request
        request = OptimizeRequest(
            objective=inputs.objective,
            smooth_weight=inputs.smooth_weight,
            bin_size_min=inputs.bin_size_min,
            meal_windows=meal_windows_dict,
            max_shift_min=inputs.max_shift_min
        )
        
        # Run optimization
        result = self.api_layer.optimize(request)
        
        return {
            'success': result.success,
            'message': result.message,
            'chosen_slots': result.chosen_slots,
            'demand_series': result.demand_series,
            'objective_value': result.objective_value,
            'solve_time': result.solve_time
        }


def main():
    """Test the Frontend Interaction Contract with sample data."""
    logger.info("Testing Frontend Interaction Contract...")
    
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
    from optimizer_service import OptimizerService
    from scenario_preview_service import ScenarioPreviewService
    from api_layer import APILayer
    
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
        
        # Initialize preview service
        preview_service = ScenarioPreviewService(ivp, demand_service, time_grid, sections_catalog)
        
        # Initialize API layer
        api_layer = APILayer(preview_service, optimizer, sections_catalog, time_grid)
        
        # Initialize frontend contract service
        frontend_service = FrontendContractService(api_layer, time_grid, sections_catalog)
        
        # Test 1: Peak minimization
        logger.info("Testing peak minimization...")
        inputs1 = FrontendInputs(
            bin_size_min=15,
            max_shift_min=60,
            objective="peak_minimization"
        )
        response1 = frontend_service.process_frontend_request(inputs1)
        print(f"Peak minimization success: {response1.success}")
        print(f"Peak reduction: {response1.kpi_metrics.peak_reduction:.2f} ({response1.kpi_metrics.peak_reduction_percent:.1f}%)")
        print(f"Sections moved: {response1.kpi_metrics.sections_moved}")
        print(f"Affected sections: {len(response1.affected_sections)}")
        
        # Test 2: Smooth minimization
        logger.info("Testing smooth minimization...")
        inputs2 = FrontendInputs(
            bin_size_min=15,
            max_shift_min=90,
            objective="smooth_minimization",
            smooth_weight=0.5
        )
        response2 = frontend_service.process_frontend_request(inputs2)
        print(f"Smooth minimization success: {response2.success}")
        print(f"Variance reduction: {response2.kpi_metrics.variance_reduction:.2f} ({response2.kpi_metrics.variance_reduction_percent:.1f}%)")
        print(f"Sections moved: {response2.kpi_metrics.sections_moved}")
        
        # Test 3: Custom meal windows
        logger.info("Testing custom meal windows...")
        inputs3 = FrontendInputs(
            bin_size_min=30,
            max_shift_min=120,
            meal_windows={
                "breakfast": (480, 720),  # 8:00 AM - 12:00 PM
                "lunch": (720, 960),      # 12:00 PM - 4:00 PM
                "dinner": (960, 1200)     # 4:00 PM - 8:00 PM
            },
            objective="peak_minimization"
        )
        response3 = frontend_service.process_frontend_request(inputs3)
        print(f"Custom meal windows success: {response3.success}")
        print(f"Meal stacks: {len(response3.meal_stacks)}")
        for stack in response3.meal_stacks:
            print(f"  {stack.meal_period}: {stack.current_demand:.1f} -> {stack.optimized_demand:.1f} ({stack.change_percent:+.1f}%)")
        
        # Test 4: Invalid inputs
        logger.info("Testing invalid inputs...")
        inputs4 = FrontendInputs(
            bin_size_min=100,  # Invalid: too large
            max_shift_min=300,  # Invalid: too large
            objective="invalid_objective"  # Invalid objective
        )
        response4 = frontend_service.process_frontend_request(inputs4)
        print(f"Invalid inputs handled: {not response4.success}")
        print(f"Error message: {response4.message}")
        
        logger.info("Frontend Interaction Contract test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
