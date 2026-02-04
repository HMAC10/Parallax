"""
Path and mission optimization module.

Provides algorithms for:
- Path planning and optimization
- Coverage pattern generation
- Energy-efficient routing
- Collision avoidance
"""

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Point3D:
    """A 3D point in space."""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point3D":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def distance_to(self, other: "Point3D") -> float:
        """Calculate Euclidean distance to another point."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))


class PathOptimizer:
    """
    Optimizes drone flight paths for efficiency.
    """
    
    def __init__(self, max_speed: float = 5.0, max_altitude: float = 120.0):
        """
        Initialize path optimizer.
        
        Args:
            max_speed: Maximum drone speed in m/s
            max_altitude: Maximum allowed altitude in meters
        """
        self.max_speed = max_speed
        self.max_altitude = max_altitude
    
    def calculate_path_length(self, waypoints: list[Point3D]) -> float:
        """Calculate total path length."""
        if len(waypoints) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(waypoints) - 1):
            total += waypoints[i].distance_to(waypoints[i + 1])
        return total
    
    def estimate_flight_time(
        self,
        waypoints: list[Point3D],
        speed: float = None,
    ) -> float:
        """
        Estimate flight time for a path.
        
        Args:
            waypoints: List of waypoints
            speed: Flight speed (defaults to max_speed)
            
        Returns:
            Estimated time in seconds
        """
        speed = speed or self.max_speed
        distance = self.calculate_path_length(waypoints)
        return distance / speed
    
    def optimize_waypoint_order(
        self,
        waypoints: list[Point3D],
        start: Optional[Point3D] = None,
    ) -> list[Point3D]:
        """
        Optimize waypoint visit order using nearest-neighbor heuristic.
        
        Args:
            waypoints: Waypoints to visit
            start: Starting position (optional)
            
        Returns:
            Reordered waypoints for shorter path
        """
        if len(waypoints) <= 2:
            return waypoints
        
        # Use greedy nearest-neighbor algorithm
        remaining = list(waypoints)
        optimized = []
        
        # Start from provided start point or first waypoint
        if start:
            current = start
        else:
            current = remaining.pop(0)
            optimized.append(current)
        
        while remaining:
            # Find nearest unvisited waypoint
            nearest_idx = 0
            nearest_dist = float('inf')
            
            for i, wp in enumerate(remaining):
                dist = current.distance_to(wp)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = i
            
            current = remaining.pop(nearest_idx)
            optimized.append(current)
        
        return optimized
    
    def generate_survey_pattern(
        self,
        center: Point3D,
        width: float,
        height: float,
        altitude: float,
        spacing: float,
        pattern: str = "lawnmower",
    ) -> list[Point3D]:
        """
        Generate survey pattern waypoints.
        
        Args:
            center: Center point of survey area
            width: Width of area (X direction)
            height: Height of area (Y direction)
            altitude: Flight altitude
            spacing: Distance between survey lines
            pattern: Pattern type ('lawnmower', 'spiral', 'expanding')
            
        Returns:
            List of waypoints for the survey pattern
        """
        waypoints = []
        
        if pattern == "lawnmower":
            # Generate back-and-forth lawnmower pattern
            x_start = center.x - width / 2
            x_end = center.x + width / 2
            y_start = center.y - height / 2
            y_end = center.y + height / 2
            
            y = y_start
            direction = 1  # 1 = forward, -1 = backward
            
            while y <= y_end:
                if direction == 1:
                    waypoints.append(Point3D(x_start, y, altitude))
                    waypoints.append(Point3D(x_end, y, altitude))
                else:
                    waypoints.append(Point3D(x_end, y, altitude))
                    waypoints.append(Point3D(x_start, y, altitude))
                
                y += spacing
                direction *= -1
        
        elif pattern == "spiral":
            # Generate inward spiral pattern
            num_rings = int(min(width, height) / (2 * spacing))
            
            for ring in range(num_rings):
                offset = ring * spacing
                half_w = width / 2 - offset
                half_h = height / 2 - offset
                
                if half_w <= 0 or half_h <= 0:
                    break
                
                # Four corners of the ring
                waypoints.extend([
                    Point3D(center.x - half_w, center.y - half_h, altitude),
                    Point3D(center.x + half_w, center.y - half_h, altitude),
                    Point3D(center.x + half_w, center.y + half_h, altitude),
                    Point3D(center.x - half_w, center.y + half_h, altitude),
                ])
            
            # End at center
            waypoints.append(Point3D(center.x, center.y, altitude))
        
        elif pattern == "expanding":
            # Generate expanding square pattern from center
            waypoints.append(Point3D(center.x, center.y, altitude))
            
            ring = 1
            max_rings = int(max(width, height) / (2 * spacing))
            
            while ring <= max_rings:
                offset = ring * spacing
                
                if offset > width / 2 and offset > height / 2:
                    break
                
                # Expand outward in square pattern
                for angle in [0, 90, 180, 270]:
                    rad = math.radians(angle)
                    x = center.x + offset * math.cos(rad)
                    y = center.y + offset * math.sin(rad)
                    
                    # Clamp to area bounds
                    x = max(center.x - width / 2, min(center.x + width / 2, x))
                    y = max(center.y - height / 2, min(center.y + height / 2, y))
                    
                    waypoints.append(Point3D(x, y, altitude))
                
                ring += 1
        
        return waypoints
    
    def generate_circle_pattern(
        self,
        center: Point3D,
        radius: float,
        altitude: float,
        num_points: int = 8,
    ) -> list[Point3D]:
        """
        Generate circular orbit waypoints.
        
        Args:
            center: Center of circle
            radius: Circle radius
            altitude: Flight altitude
            num_points: Number of waypoints around circle
            
        Returns:
            List of waypoints forming a circle
        """
        waypoints = []
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            waypoints.append(Point3D(x, y, altitude))
        
        # Close the circle
        waypoints.append(waypoints[0])
        
        return waypoints
    
    def smooth_path(
        self,
        waypoints: list[Point3D],
        smoothing_factor: float = 0.5,
    ) -> list[Point3D]:
        """
        Smooth a path using simple averaging.
        
        Args:
            waypoints: Original waypoints
            smoothing_factor: How much to smooth (0-1)
            
        Returns:
            Smoothed waypoints
        """
        if len(waypoints) <= 2:
            return waypoints
        
        smoothed = [waypoints[0]]  # Keep start point
        
        for i in range(1, len(waypoints) - 1):
            prev_arr = waypoints[i - 1].to_array()
            curr_arr = waypoints[i].to_array()
            next_arr = waypoints[i + 1].to_array()
            
            # Average with neighbors
            avg = (prev_arr + curr_arr + next_arr) / 3
            
            # Blend between original and averaged
            blended = curr_arr * (1 - smoothing_factor) + avg * smoothing_factor
            
            smoothed.append(Point3D.from_array(blended))
        
        smoothed.append(waypoints[-1])  # Keep end point
        
        return smoothed


class EnergyOptimizer:
    """
    Optimizes paths for energy efficiency.
    """
    
    def __init__(
        self,
        hover_power: float = 100.0,  # Watts
        forward_power: float = 150.0,  # Watts at cruise speed
        climb_power: float = 200.0,  # Watts
        battery_capacity: float = 5000.0,  # mAh
        voltage: float = 22.2,  # Volts (6S LiPo)
    ):
        """
        Initialize energy optimizer.
        
        Args:
            hover_power: Power consumption while hovering (W)
            forward_power: Power consumption in forward flight (W)
            climb_power: Power consumption while climbing (W)
            battery_capacity: Battery capacity (mAh)
            voltage: Battery voltage (V)
        """
        self.hover_power = hover_power
        self.forward_power = forward_power
        self.climb_power = climb_power
        self.battery_wh = battery_capacity * voltage / 1000  # Watt-hours
    
    def estimate_energy(
        self,
        waypoints: list[Point3D],
        speed: float = 5.0,
    ) -> dict:
        """
        Estimate energy consumption for a path.
        
        Args:
            waypoints: Flight path waypoints
            speed: Cruise speed in m/s
            
        Returns:
            Dictionary with energy estimates
        """
        if len(waypoints) < 2:
            return {"total_wh": 0, "feasible": True, "remaining_percent": 100}
        
        total_energy = 0.0  # Watt-hours
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # Calculate segment properties
            horizontal_dist = math.sqrt(
                (end.x - start.x) ** 2 + (end.y - start.y) ** 2
            )
            vertical_dist = end.z - start.z  # Negative = climbing
            
            # Estimate time for segment
            segment_dist = start.distance_to(end)
            segment_time = segment_dist / speed  # seconds
            
            # Calculate power based on flight phase
            if vertical_dist < -0.5:  # Climbing
                power = self.climb_power
            elif abs(vertical_dist) <= 0.5 and horizontal_dist < 0.5:  # Hovering
                power = self.hover_power
            else:  # Forward flight or descent
                power = self.forward_power
            
            # Energy = Power * Time (convert seconds to hours)
            segment_energy = power * (segment_time / 3600)
            total_energy += segment_energy
        
        remaining_percent = max(0, (1 - total_energy / self.battery_wh) * 100)
        
        return {
            "total_wh": total_energy,
            "feasible": total_energy < self.battery_wh * 0.8,  # 80% threshold
            "remaining_percent": remaining_percent,
            "flight_time_minutes": (self.battery_wh - total_energy) / self.forward_power * 60,
        }
    
    def optimize_altitude(
        self,
        waypoints: list[Point3D],
        min_altitude: float = 10.0,
        max_altitude: float = 100.0,
    ) -> list[Point3D]:
        """
        Optimize altitudes for energy efficiency.
        
        Lower altitudes generally use less energy but may have obstacles.
        
        Args:
            waypoints: Original waypoints
            min_altitude: Minimum safe altitude
            max_altitude: Maximum allowed altitude
            
        Returns:
            Altitude-optimized waypoints
        """
        optimized = []
        
        for wp in waypoints:
            # Use minimum safe altitude for energy efficiency
            optimized_z = max(min_altitude, min(wp.z, max_altitude))
            optimized.append(Point3D(wp.x, wp.y, optimized_z))
        
        return optimized


__all__ = [
    "Point3D",
    "PathOptimizer",
    "EnergyOptimizer",
]
