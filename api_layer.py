"""
API Layer module for dining demand optimization system.

This module provides a clean REST API and real-time WebSocket/SSE interface
for the dining demand optimization system, enabling reactive UI updates.
"""

import logging
import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Enumeration of optimization objectives."""
    PEAK_MINIMIZATION = "peak_minimization"
    SMOOTH_MINIMIZATION = "smooth_minimization"


@dataclass
class SectionInfo:
    """Section information for API responses."""
    section_id: str
    start_min: int
    end_min: int
    enrolled_count: int
    allowed_slots_min: List[int]


@dataclass
class DemandPreviewRequest:
    """Request for demand preview."""
    edits: Dict[str, int]  # section_id -> new_start_min
    grid_settings: Optional[Dict[str, Any]] = None
    window_settings: Optional[Dict[str, Any]] = None


@dataclass
class DemandPreviewResponse:
    """Response for demand preview."""
    success: bool
    demand_series: Dict[int, float]  # bin_index -> demand
    metrics: Dict[str, float]
    applied_edits: Dict[str, int]
    validation_results: List[Dict[str, Any]]
    computation_time: float
    method_used: str
    message: str


@dataclass
class OptimizeRequest:
    """Request for schedule optimization."""
    objective: str
    smooth_weight: Optional[float] = None
    bin_size_min: int = 15
    meal_windows: Optional[Dict[str, Any]] = None
    max_shift_min: int = 60
    max_sections_moved_pct: Optional[float] = None


@dataclass
class OptimizeResponse:
    """Response for schedule optimization."""
    success: bool
    chosen_slots: Dict[str, int]  # section_id -> new_start_min
    demand_series: Dict[int, float]  # bin_index -> demand
    metrics: Dict[str, float]
    objective_value: float
    solve_time: float
    message: str


@dataclass
class EventMessage:
    """Message for real-time events."""
    event_type: str
    data: Dict[str, Any]
    timestamp: float
    client_id: Optional[str] = None


class DebouncedEventManager:
    """Manages debounced events for real-time updates."""
    
    def __init__(self, debounce_delay: float = 0.5):
        """
        Initialize the debounced event manager.
        
        Args:
            debounce_delay: Delay in seconds before processing events
        """
        self.debounce_delay = debounce_delay
        self.pending_events = defaultdict(lambda: deque())
        self.event_timers = {}
        self.subscribers = set()
        self.lock = threading.Lock()
    
    def subscribe(self, callback):
        """Subscribe to events."""
        with self.lock:
            self.subscribers.add(callback)
    
    def unsubscribe(self, callback):
        """Unsubscribe from events."""
        with self.lock:
            self.subscribers.discard(callback)
    
    def emit_event(self, event_type: str, data: Dict[str, Any], client_id: Optional[str] = None):
        """Emit a debounced event."""
        with self.lock:
            # Add event to pending queue
            event = EventMessage(
                event_type=event_type,
                data=data,
                timestamp=time.time(),
                client_id=client_id
            )
            self.pending_events[event_type].append(event)
            
            # Cancel existing timer for this event type
            if event_type in self.event_timers:
                self.event_timers[event_type].cancel()
            
            # Set new timer
            timer = threading.Timer(self.debounce_delay, self._process_events, args=(event_type,))
            self.event_timers[event_type] = timer
            timer.start()
    
    def _process_events(self, event_type: str):
        """Process debounced events."""
        with self.lock:
            if event_type not in self.pending_events:
                return
            
            events = list(self.pending_events[event_type])
            self.pending_events[event_type].clear()
            
            if event_type in self.event_timers:
                del self.event_timers[event_type]
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(event_type, events)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")


class APILayer:
    """
    API Layer for the dining demand optimization system.
    
    Provides REST endpoints and real-time WebSocket/SSE interface
    for reactive UI updates.
    """
    
    def __init__(self, 
                 scenario_preview_service,
                 optimizer_service,
                 sections_catalog,
                 time_grid_service,
                 debounce_delay: float = 0.5):
        """
        Initialize the API Layer.
        
        Args:
            scenario_preview_service: ScenarioPreviewService instance
            optimizer_service: OptimizerService instance
            sections_catalog: SectionsCatalog instance
            time_grid_service: TimeGridService instance
            debounce_delay: Delay for debounced events in seconds
        """
        self.preview_service = scenario_preview_service
        self.optimizer_service = optimizer_service
        self.sections_catalog = sections_catalog
        self.time_grid_service = time_grid_service
        self.event_manager = DebouncedEventManager(debounce_delay)
        
        # Initialize event manager
        self.event_manager.subscribe(self._handle_debounced_event)
        
        # Store current state for real-time updates
        self.current_demand_series = {}
        self.current_metrics = {}
        self.current_sections = {}
        
        logger.info("API Layer initialized")
    
    def get_sections(self) -> List[SectionInfo]:
        """
        GET /sections endpoint.
        
        Returns:
            List of section information with allowed slots
        """
        logger.info("GET /sections - Retrieving sections")
        
        sections = []
        for section_id, section in self.sections_catalog.sections_dict.items():
            # Get allowed slots for this section
            allowed_slots = self.preview_service.get_available_slots_for_section(section_id)
            
            section_info = SectionInfo(
                section_id=section_id,
                start_min=section.start_minutes,
                end_min=section.end_minutes,
                enrolled_count=section.student_count,
                allowed_slots_min=allowed_slots
            )
            sections.append(section_info)
        
        logger.info(f"Retrieved {len(sections)} sections")
        return sections
    
    def demand_preview(self, request: DemandPreviewRequest) -> DemandPreviewResponse:
        """
        POST /demand/preview endpoint.
        
        Args:
            request: DemandPreviewRequest with edits and settings
            
        Returns:
            DemandPreviewResponse with updated demand and metrics
        """
        logger.info(f"POST /demand/preview - Processing {len(request.edits)} edits")
        
        try:
            # Update grid/window settings if provided
            if request.grid_settings:
                self._update_grid_settings(request.grid_settings)
            
            if request.window_settings:
                self._update_window_settings(request.window_settings)
            
            # Get student IDs for demand calculation
            student_ids = self._get_student_ids()
            
            # Process preview
            result = self.preview_service.preview_scenario(request.edits, student_ids)
            
            # Update current state
            self.current_demand_series = result.demand_series
            self.current_metrics = result.metrics
            
            # Emit real-time event
            self.event_manager.emit_event("demand_updated", {
                "demand_series": result.demand_series,
                "metrics": result.metrics,
                "applied_edits": result.applied_edits,
                "computation_time": result.computation_time,
                "method_used": result.method_used
            })
            
            return DemandPreviewResponse(
                success=result.success,
                demand_series=result.demand_series,
                metrics=result.metrics,
                applied_edits=result.applied_edits,
                validation_results=[asdict(vr) for vr in result.validation_results],
                computation_time=result.computation_time,
                method_used=result.method_used,
                message=result.message
            )
            
        except Exception as e:
            logger.error(f"Error in demand preview: {e}")
            return DemandPreviewResponse(
                success=False,
                demand_series={},
                metrics={},
                applied_edits={},
                validation_results=[],
                computation_time=0.0,
                method_used="error",
                message=f"Preview failed: {e}"
            )
    
    def optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        """
        POST /optimize endpoint.
        
        Args:
            request: OptimizeRequest with optimization parameters
            
        Returns:
            OptimizeResponse with optimization results
        """
        logger.info(f"POST /optimize - Starting optimization with objective: {request.objective}")
        
        try:
            # Update grid/window settings if provided
            if request.bin_size_min != self.time_grid_service.bin_size_minutes:
                self._update_bin_size(request.bin_size_min)
            
            if request.meal_windows:
                self._update_meal_windows(request.meal_windows)
            
            # Ensure impact vectors are precomputed after any changes
            if not self.preview_service.ivp.is_precomputed:
                student_ids = self._get_student_ids()
                self.preview_service.ivp.precompute_impact_vectors(student_ids)
            
            # Create optimization config
            from optimizer_service import OptimizationConfig, OptimizationObjective
            
            objective = OptimizationObjective.PEAK_MINIMIZATION
            if request.objective == "smooth_minimization":
                objective = OptimizationObjective.SMOOTH_MINIMIZATION
            
            config = OptimizationConfig(
                objective=objective,
                max_shift_minutes=request.max_shift_min,
                penalty_per_minute=0.0,  # Could be made configurable
                time_limit_seconds=300,
                verbose=False
            )
            
            # Run optimization
            result = self.optimizer_service.optimize_schedule(config)
            
            # Update current state
            self.current_demand_series = result.final_demand
            self.current_metrics = result.metrics
            self.current_sections = result.chosen_slots
            
            # Emit real-time event
            self.event_manager.emit_event("optimization_completed", {
                "chosen_slots": result.chosen_slots,
                "demand_series": result.final_demand,
                "metrics": result.metrics,
                "objective_value": result.objective_value,
                "solve_time": result.solve_time
            })
            
            return OptimizeResponse(
                success=result.success,
                chosen_slots=result.chosen_slots,
                demand_series=result.final_demand,
                metrics=result.metrics,
                objective_value=result.objective_value,
                solve_time=result.solve_time,
                message=result.message
            )
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return OptimizeResponse(
                success=False,
                chosen_slots={},
                demand_series={},
                metrics={},
                objective_value=float('inf'),
                solve_time=0.0,
                message=f"Optimization failed: {e}"
            )
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current system state for real-time updates.
        
        Returns:
            Dictionary with current state
        """
        return {
            "demand_series": self.current_demand_series,
            "metrics": self.current_metrics,
            "sections": self.current_sections,
            "timestamp": time.time()
        }
    
    def _update_grid_settings(self, settings: Dict[str, Any]):
        """Update grid settings."""
        if "bin_size_minutes" in settings:
            self._update_bin_size(settings["bin_size_minutes"])
        
        if "day_start_minutes" in settings:
            self.time_grid_service.day_start_minutes = settings["day_start_minutes"]
            self.time_grid_service._generate_time_bins()
        
        if "day_end_minutes" in settings:
            self.time_grid_service.day_end_minutes = settings["day_end_minutes"]
            self.time_grid_service._generate_time_bins()
    
    def _update_window_settings(self, settings: Dict[str, Any]):
        """Update meal window settings."""
        if "breakfast_window" in settings:
            self.time_grid_service.breakfast_window = settings["breakfast_window"]
        
        if "lunch_window" in settings:
            self.time_grid_service.lunch_window = settings["lunch_window"]
        
        if "dinner_window" in settings:
            self.time_grid_service.dinner_window = settings["dinner_window"]
        
        # Regenerate time bins with new windows
        self.time_grid_service._generate_time_bins()
    
    def _update_bin_size(self, bin_size_minutes: int):
        """Update bin size and regenerate grid."""
        self.time_grid_service.bin_size_minutes = bin_size_minutes
        self.time_grid_service._generate_time_bins()
        
        # Invalidate impact vectors cache
        if hasattr(self.preview_service.ivp, 'invalidate_cache'):
            self.preview_service.ivp.invalidate_cache()
    
    def _update_meal_windows(self, windows: Dict[str, Any]):
        """Update meal windows."""
        from time_grid_service import TimeWindow
        
        if "breakfast" in windows:
            start, end = windows["breakfast"]
            self.time_grid_service.breakfast_window = TimeWindow(start, end)
        
        if "lunch" in windows:
            start, end = windows["lunch"]
            self.time_grid_service.lunch_window = TimeWindow(start, end)
        
        if "dinner" in windows:
            start, end = windows["dinner"]
            self.time_grid_service.dinner_window = TimeWindow(start, end)
        
        # Regenerate time bins
        self.time_grid_service._generate_time_bins()
        
        # Invalidate impact vectors cache
        if hasattr(self.preview_service.ivp, 'invalidate_cache'):
            self.preview_service.ivp.invalidate_cache()
    
    def _get_student_ids(self) -> List[str]:
        """Get list of student IDs for demand calculation."""
        # Get student IDs from the sections catalog
        student_ids = set()
        for section in self.sections_catalog.sections_dict.values():
            # Get students enrolled in this section
            section_students = self.sections_catalog.get_students_in_section(section.section_id)
            student_ids.update(section_students)
        return list(student_ids)
    
    def _handle_debounced_event(self, event_type: str, events: List[EventMessage]):
        """Handle debounced events."""
        logger.info(f"Processing {len(events)} {event_type} events")
        
        # This would typically broadcast to WebSocket clients
        # For now, we'll just log the events
        for event in events:
            logger.debug(f"Event: {event.event_type} - {event.data}")


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self, api_layer: APILayer):
        """
        Initialize WebSocket manager.
        
        Args:
            api_layer: APILayer instance
        """
        self.api_layer = api_layer
        self.connections = set()
        self.lock = threading.Lock()
        
        # Subscribe to events
        self.api_layer.event_manager.subscribe(self._broadcast_event)
    
    def add_connection(self, websocket):
        """Add a WebSocket connection."""
        with self.lock:
            self.connections.add(websocket)
            logger.info(f"WebSocket connection added. Total: {len(self.connections)}")
    
    def remove_connection(self, websocket):
        """Remove a WebSocket connection."""
        with self.lock:
            self.connections.discard(websocket)
            logger.info(f"WebSocket connection removed. Total: {len(self.connections)}")
    
    def _broadcast_event(self, event_type: str, events: List[EventMessage]):
        """Broadcast events to all connected clients."""
        if not events:
            return
        
        # Get the latest event
        latest_event = events[-1]
        
        message = {
            "type": "event",
            "event_type": event_type,
            "data": latest_event.data,
            "timestamp": latest_event.timestamp
        }
        
        # Broadcast to all connections
        with self.lock:
            disconnected = set()
            for websocket in self.connections:
                try:
                    websocket.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
                    disconnected.add(websocket)
            
            # Remove disconnected connections
            for websocket in disconnected:
                self.connections.discard(websocket)
    
    def broadcast_state_update(self):
        """Broadcast current state to all clients."""
        state = self.api_layer.get_current_state()
        message = {
            "type": "state_update",
            "data": state,
            "timestamp": time.time()
        }
        
        with self.lock:
            disconnected = set()
            for websocket in self.connections:
                try:
                    websocket.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending state update: {e}")
                    disconnected.add(websocket)
            
            # Remove disconnected connections
            for websocket in disconnected:
                self.connections.discard(websocket)


def main():
    """Test the API Layer with actual data."""
    logger.info("Testing API Layer...")
    
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
        
        # Test GET /sections
        logger.info("Testing GET /sections...")
        sections = api_layer.get_sections()
        print(f"Retrieved {len(sections)} sections")
        if sections:
            print(f"First section: {sections[0]}")
        
        # Test POST /demand/preview
        logger.info("Testing POST /demand/preview...")
        preview_request = DemandPreviewRequest(
            edits={"0": 600, "1": 700},  # Example edits
            grid_settings={"bin_size_minutes": 15},
            window_settings={"breakfast": (420, 630)}
        )
        preview_response = api_layer.demand_preview(preview_request)
        print(f"Preview response: {preview_response.success}")
        print(f"Applied edits: {preview_response.applied_edits}")
        print(f"Computation time: {preview_response.computation_time:.3f}s")
        
        # Test POST /optimize
        logger.info("Testing POST /optimize...")
        optimize_request = OptimizeRequest(
            objective="peak_minimization",
            bin_size_min=15,
            max_shift_min=60
        )
        optimize_response = api_layer.optimize(optimize_request)
        print(f"Optimization response: {optimize_response.success}")
        print(f"Chosen slots: {len(optimize_response.chosen_slots)}")
        print(f"Objective value: {optimize_response.objective_value:.2f}")
        print(f"Solve time: {optimize_response.solve_time:.3f}s")
        
        # Test WebSocket manager
        logger.info("Testing WebSocket manager...")
        websocket_manager = WebSocketManager(api_layer)
        print(f"WebSocket manager initialized with {len(websocket_manager.connections)} connections")
        
        # Test current state
        current_state = api_layer.get_current_state()
        print(f"Current state keys: {list(current_state.keys())}")
        
        logger.info("API Layer test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
