"""
TimeGridService module for dining demand optimization system.

This module provides a canonical time discretization system with configurable
time bins and meal windows that can be instantly reconfigured from UI sliders.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MealPeriod(Enum):
    """Enumeration of meal periods."""
    BREAKFAST = "Breakfast"
    LUNCH = "Lunch"
    DINNER = "Dinner"
    NONE = "None"


@dataclass
class TimeWindow:
    """Represents a time window with start and end times in minutes since midnight."""
    start_minutes: int
    end_minutes: int
    
    def contains(self, minutes: int) -> bool:
        """Check if a time (in minutes) falls within this window."""
        return self.start_minutes <= minutes < self.end_minutes
    
    def __str__(self) -> str:
        start_hour = self.start_minutes // 60
        start_min = self.start_minutes % 60
        end_hour = self.end_minutes // 60
        end_min = self.end_minutes % 60
        return f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}"


@dataclass
class TimeBin:
    """Represents a single time bin in the grid."""
    index: int
    start_minutes: int
    end_minutes: int
    meal_period: MealPeriod
    
    def contains(self, minutes: int) -> bool:
        """Check if a time (in minutes) falls within this bin."""
        return self.start_minutes <= minutes < self.end_minutes
    
    def __str__(self) -> str:
        start_hour = self.start_minutes // 60
        start_min = self.start_minutes % 60
        end_hour = self.end_minutes // 60
        end_min = self.end_minutes % 60
        return f"Bin {self.index}: {start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d} ({self.meal_period.value})"


class TimeGridService:
    """Service for managing time discretization and meal period classification."""
    
    def __init__(self, 
                 bin_size_minutes: int = 15,
                 day_start_minutes: int = 420,  # 07:00
                 day_end_minutes: int = 1260,  # 21:00
                 breakfast_window: Optional[TimeWindow] = None,
                 lunch_window: Optional[TimeWindow] = None,
                 dinner_window: Optional[TimeWindow] = None):
        """
        Initialize TimeGridService.
        
        Args:
            bin_size_minutes: Size of each time bin in minutes
            day_start_minutes: Start of day in minutes since midnight
            day_end_minutes: End of day in minutes since midnight
            breakfast_window: Breakfast time window (default: 07:00-10:30)
            lunch_window: Lunch time window (default: 11:00-14:30)
            dinner_window: Dinner time window (default: 17:00-20:30)
        """
        self.bin_size_minutes = bin_size_minutes
        self.day_start_minutes = day_start_minutes
        self.day_end_minutes = day_end_minutes
        
        # Set default meal windows if not provided
        self.breakfast_window = breakfast_window or TimeWindow(420, 630)   # 07:00-10:30
        self.lunch_window = lunch_window or TimeWindow(660, 870)           # 11:00-14:30
        self.dinner_window = dinner_window or TimeWindow(1020, 1230)       # 17:00-20:30
        
        # Generate time bins
        self.time_bins = self._generate_time_bins()
        
        logger.info(f"Initialized TimeGridService with {len(self.time_bins)} bins of {bin_size_minutes} minutes each")
        logger.info(f"Day range: {self._minutes_to_time_string(day_start_minutes)} - {self._minutes_to_time_string(day_end_minutes)}")
        logger.info(f"Meal windows - Breakfast: {self.breakfast_window}, Lunch: {self.lunch_window}, Dinner: {self.dinner_window}")
    
    def _generate_time_bins(self) -> List[TimeBin]:
        """Generate all time bins for the day."""
        bins = []
        bin_index = 0
        
        current_time = self.day_start_minutes
        
        while current_time < self.day_end_minutes:
            end_time = min(current_time + self.bin_size_minutes, self.day_end_minutes)
            
            # Determine meal period for this bin
            meal_period = self._get_meal_period_for_time(current_time)
            
            bin_obj = TimeBin(
                index=bin_index,
                start_minutes=current_time,
                end_minutes=end_time,
                meal_period=meal_period
            )
            
            bins.append(bin_obj)
            bin_index += 1
            current_time = end_time
        
        return bins
    
    def _get_meal_period_for_time(self, minutes: int) -> MealPeriod:
        """Determine meal period for a given time."""
        if self.breakfast_window.contains(minutes):
            return MealPeriod.BREAKFAST
        elif self.lunch_window.contains(minutes):
            return MealPeriod.LUNCH
        elif self.dinner_window.contains(minutes):
            return MealPeriod.DINNER
        else:
            return MealPeriod.NONE
    
    def _minutes_to_time_string(self, minutes: int) -> str:
        """Convert minutes since midnight to HH:MM format."""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def minute_to_bin_index(self, minutes: int) -> Optional[int]:
        """
        Convert minutes since midnight to bin index.
        
        Args:
            minutes: Time in minutes since midnight
            
        Returns:
            Bin index or None if time is outside day range
        """
        if minutes < self.day_start_minutes or minutes >= self.day_end_minutes:
            return None
        
        # Find the bin containing this time
        for bin_obj in self.time_bins:
            if bin_obj.contains(minutes):
                return bin_obj.index
        
        return None
    
    def bin_index_to_time_range(self, bin_index: int) -> Optional[Tuple[int, int]]:
        """
        Convert bin index to time range.
        
        Args:
            bin_index: Index of the time bin
            
        Returns:
            Tuple of (start_minutes, end_minutes) or None if invalid index
        """
        if 0 <= bin_index < len(self.time_bins):
            bin_obj = self.time_bins[bin_index]
            return (bin_obj.start_minutes, bin_obj.end_minutes)
        return None
    
    def get_bin(self, bin_index: int) -> Optional[TimeBin]:
        """
        Get TimeBin object by index.
        
        Args:
            bin_index: Index of the time bin
            
        Returns:
            TimeBin object or None if invalid index
        """
        if 0 <= bin_index < len(self.time_bins):
            return self.time_bins[bin_index]
        return None
    
    def get_bins_in_meal_period(self, meal_period: MealPeriod) -> List[TimeBin]:
        """
        Get all bins that belong to a specific meal period.
        
        Args:
            meal_period: The meal period to filter by
            
        Returns:
            List of TimeBin objects
        """
        return [bin_obj for bin_obj in self.time_bins if bin_obj.meal_period == meal_period]
    
    def get_meal_period_bins(self) -> Dict[MealPeriod, List[TimeBin]]:
        """
        Get all bins grouped by meal period.
        
        Returns:
            Dictionary mapping meal periods to lists of bins
        """
        result = {period: [] for period in MealPeriod}
        for bin_obj in self.time_bins:
            result[bin_obj.meal_period].append(bin_obj)
        return result
    
    def update_meal_windows(self, 
                           breakfast_window: Optional[TimeWindow] = None,
                           lunch_window: Optional[TimeWindow] = None,
                           dinner_window: Optional[TimeWindow] = None) -> None:
        """
        Update meal windows and regenerate bins.
        
        Args:
            breakfast_window: New breakfast window
            lunch_window: New lunch window
            dinner_window: New dinner window
        """
        if breakfast_window is not None:
            self.breakfast_window = breakfast_window
        if lunch_window is not None:
            self.lunch_window = lunch_window
        if dinner_window is not None:
            self.dinner_window = dinner_window
        
        # Regenerate bins with new meal windows
        self.time_bins = self._generate_time_bins()
        
        logger.info("Updated meal windows and regenerated time bins")
        logger.info(f"New meal windows - Breakfast: {self.breakfast_window}, Lunch: {self.lunch_window}, Dinner: {self.dinner_window}")
    
    def update_bin_size(self, new_bin_size_minutes: int) -> None:
        """
        Update bin size and regenerate bins.
        
        Args:
            new_bin_size_minutes: New bin size in minutes
        """
        if new_bin_size_minutes <= 0:
            raise ValueError("Bin size must be positive")
        
        self.bin_size_minutes = new_bin_size_minutes
        self.time_bins = self._generate_time_bins()
        
        logger.info(f"Updated bin size to {new_bin_size_minutes} minutes and regenerated {len(self.time_bins)} bins")
    
    def update_day_range(self, start_minutes: int, end_minutes: int) -> None:
        """
        Update day range and regenerate bins.
        
        Args:
            start_minutes: New start time in minutes since midnight
            end_minutes: New end time in minutes since midnight
        """
        if start_minutes >= end_minutes:
            raise ValueError("Start time must be before end time")
        
        self.day_start_minutes = start_minutes
        self.day_end_minutes = end_minutes
        self.time_bins = self._generate_time_bins()
        
        logger.info(f"Updated day range to {self._minutes_to_time_string(start_minutes)} - {self._minutes_to_time_string(end_minutes)}")
        logger.info(f"Generated {len(self.time_bins)} bins")
    
    def get_time_grid_summary(self) -> Dict:
        """
        Get summary of the current time grid configuration.
        
        Returns:
            Dictionary with grid configuration details
        """
        meal_period_counts = {}
        for period in MealPeriod:
            meal_period_counts[period.value] = len(self.get_bins_in_meal_period(period))
        
        return {
            'total_bins': len(self.time_bins),
            'bin_size_minutes': self.bin_size_minutes,
            'day_start': self._minutes_to_time_string(self.day_start_minutes),
            'day_end': self._minutes_to_time_string(self.day_end_minutes),
            'day_start_minutes': self.day_start_minutes,
            'day_end_minutes': self.day_end_minutes,
            'meal_period_counts': meal_period_counts,
            'breakfast_window': str(self.breakfast_window),
            'lunch_window': str(self.lunch_window),
            'dinner_window': str(self.dinner_window)
        }
    
    def get_bins_for_time_range(self, start_minutes: int, end_minutes: int) -> List[TimeBin]:
        """
        Get all bins that overlap with a given time range.
        
        Args:
            start_minutes: Start of time range
            end_minutes: End of time range
            
        Returns:
            List of overlapping TimeBin objects
        """
        overlapping_bins = []
        for bin_obj in self.time_bins:
            # Check if bin overlaps with the time range
            if (bin_obj.start_minutes < end_minutes and bin_obj.end_minutes > start_minutes):
                overlapping_bins.append(bin_obj)
        
        return overlapping_bins
    
    def __str__(self) -> str:
        """String representation of the time grid."""
        summary = self.get_time_grid_summary()
        return (f"TimeGridService: {summary['total_bins']} bins of {summary['bin_size_minutes']} minutes, "
                f"from {summary['day_start']} to {summary['day_end']}")


def main():
    """Example usage of TimeGridService."""
    # Create default time grid
    time_grid = TimeGridService()
    
    print("=== Time Grid Service Demo ===")
    print(f"Grid: {time_grid}")
    print()
    
    # Show summary
    summary = time_grid.get_time_grid_summary()
    print("Configuration Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    
    # Show first 10 bins
    print("First 10 time bins:")
    for i in range(min(10, len(time_grid.time_bins))):
        bin_obj = time_grid.time_bins[i]
        print(f"  {bin_obj}")
    print()
    
    # Test helper methods
    print("Helper Method Tests:")
    test_times = [420, 540, 720, 1080, 1200]  # 07:00, 09:00, 12:00, 18:00, 20:00
    
    for minutes in test_times:
        bin_idx = time_grid.minute_to_bin_index(minutes)
        time_str = time_grid._minutes_to_time_string(minutes)
        if bin_idx is not None:
            bin_obj = time_grid.get_bin(bin_idx)
            print(f"  {time_str} -> Bin {bin_idx} ({bin_obj.meal_period.value})")
        else:
            print(f"  {time_str} -> Outside day range")
    print()
    
    # Show bins by meal period
    print("Bins by Meal Period:")
    meal_bins = time_grid.get_meal_period_bins()
    for period, bins in meal_bins.items():
        if bins:  # Only show non-empty periods
            print(f"  {period.value}: {len(bins)} bins")
            # Show first and last bin for each period
            if len(bins) > 0:
                print(f"    First: {bins[0]}")
                if len(bins) > 1:
                    print(f"    Last:  {bins[-1]}")
    print()
    
    # Test updating configuration
    print("Testing Configuration Updates:")
    
    # Update bin size
    time_grid.update_bin_size(30)  # 30-minute bins
    print(f"Updated to 30-minute bins: {len(time_grid.time_bins)} total bins")
    
    # Update meal windows
    new_breakfast = TimeWindow(450, 600)  # 07:30-10:00
    new_lunch = TimeWindow(690, 840)      # 11:30-14:00
    new_dinner = TimeWindow(1050, 1200)   # 17:30-20:00
    
    time_grid.update_meal_windows(
        breakfast_window=new_breakfast,
        lunch_window=new_lunch,
        dinner_window=new_dinner
    )
    
    print("Updated meal windows:")
    print(f"  Breakfast: {time_grid.breakfast_window}")
    print(f"  Lunch: {time_grid.lunch_window}")
    print(f"  Dinner: {time_grid.dinner_window}")


if __name__ == "__main__":
    main()
