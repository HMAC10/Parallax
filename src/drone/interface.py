"""
Abstract Drone Interface

Defines the contract for all drone implementations, whether mock,
simulation (Isaac Sim), or real hardware.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import numpy as np
from PIL import Image


class DroneStatus(Enum):
    """Drone operational status."""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ARMED = "armed"
    FLYING = "flying"
    LANDING = "landing"
    ERROR = "error"


@dataclass
class Position:
    """3D position in meters (NED frame - North, East, Down)."""
    x: float = 0.0  # North
    y: float = 0.0  # East
    z: float = 0.0  # Down (negative = altitude)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Position":
        """Create from numpy array."""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
    
    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))
    
    @property
    def altitude(self) -> float:
        """Get altitude (positive up)."""
        return -self.z
    
    def __str__(self) -> str:
        return f"Position(x={self.x:.2f}, y={self.y:.2f}, alt={self.altitude:.2f}m)"


@dataclass
class Orientation:
    """Orientation in degrees (roll, pitch, yaw)."""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0  # Heading, 0 = North, positive = clockwise
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.roll, self.pitch, self.yaw])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Orientation":
        """Create from numpy array."""
        return cls(roll=float(arr[0]), pitch=float(arr[1]), yaw=float(arr[2]))
    
    def __str__(self) -> str:
        return f"Orientation(roll={self.roll:.1f}°, pitch={self.pitch:.1f}°, yaw={self.yaw:.1f}°)"


@dataclass
class DroneState:
    """Complete drone state at a point in time."""
    status: DroneStatus = DroneStatus.DISCONNECTED
    position: Position = field(default_factory=Position)
    orientation: Orientation = field(default_factory=Orientation)
    velocity: Position = field(default_factory=Position)  # m/s
    battery_percent: float = 100.0
    gps_satellites: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_flying(self) -> bool:
        """Check if drone is currently airborne."""
        return self.status in (DroneStatus.FLYING, DroneStatus.LANDING)
    
    @property
    def is_armed(self) -> bool:
        """Check if drone motors are armed."""
        return self.status in (DroneStatus.ARMED, DroneStatus.FLYING, DroneStatus.LANDING)
    
    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            "status": self.status.value,
            "position": {
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z,
                "altitude": self.position.altitude,
            },
            "orientation": {
                "roll": self.orientation.roll,
                "pitch": self.orientation.pitch,
                "yaw": self.orientation.yaw,
            },
            "velocity": {
                "x": self.velocity.x,
                "y": self.velocity.y,
                "z": self.velocity.z,
            },
            "battery_percent": self.battery_percent,
            "gps_satellites": self.gps_satellites,
            "timestamp": self.timestamp.isoformat(),
        }


class DroneInterface(ABC):
    """
    Abstract base class for drone control.
    
    All drone implementations (mock, simulation, real) must implement
    this interface to ensure consistent behavior across the system.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the drone.
        
        Returns:
            True if connection successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the drone safely.
        
        Returns:
            True if disconnection successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def arm(self) -> bool:
        """
        Arm the drone motors.
        
        Returns:
            True if arming successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def disarm(self) -> bool:
        """
        Disarm the drone motors.
        
        Returns:
            True if disarming successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def takeoff(self, altitude: float = 2.0) -> bool:
        """
        Take off to specified altitude.
        
        Args:
            altitude: Target altitude in meters (positive up).
            
        Returns:
            True if takeoff successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def land(self) -> bool:
        """
        Land the drone at current position.
        
        Returns:
            True if landing initiated successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    async def goto(
        self, 
        x: float, 
        y: float, 
        z: Optional[float] = None,
        speed: float = 1.0
    ) -> bool:
        """
        Navigate to specified position.
        
        Args:
            x: Target X position (North) in meters.
            y: Target Y position (East) in meters.
            z: Target Z position (Down) in meters. None = maintain altitude.
            speed: Movement speed in m/s.
            
        Returns:
            True if navigation started successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    async def rotate(self, yaw: float, relative: bool = True) -> bool:
        """
        Rotate the drone to specified heading.
        
        Args:
            yaw: Target yaw angle in degrees.
            relative: If True, rotate by yaw degrees. If False, rotate to absolute heading.
            
        Returns:
            True if rotation started successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    async def hover(self) -> bool:
        """
        Stop all movement and hold current position.
        
        Returns:
            True if hover mode engaged, False otherwise.
        """
        pass
    
    @abstractmethod
    async def get_state(self) -> DroneState:
        """
        Get current drone state.
        
        Returns:
            Current DroneState object.
        """
        pass
    
    @abstractmethod
    async def capture_image(self) -> Optional[Image.Image]:
        """
        Capture image from drone camera.
        
        Returns:
            PIL Image object, or None if capture failed.
        """
        pass
    
    # =========================================================================
    # Convenience Methods (can be overridden)
    # =========================================================================
    
    async def goto_altitude(self, altitude: float, speed: float = 1.0) -> bool:
        """
        Change altitude while maintaining horizontal position.
        
        Args:
            altitude: Target altitude in meters (positive up).
            speed: Movement speed in m/s.
        """
        state = await self.get_state()
        return await self.goto(
            state.position.x,
            state.position.y,
            -altitude,  # Convert to NED
            speed
        )
    
    async def move_forward(self, distance: float, speed: float = 1.0) -> bool:
        """
        Move forward relative to current heading.
        
        Args:
            distance: Distance in meters.
            speed: Movement speed in m/s.
        """
        state = await self.get_state()
        yaw_rad = np.radians(state.orientation.yaw)
        
        new_x = state.position.x + distance * np.cos(yaw_rad)
        new_y = state.position.y + distance * np.sin(yaw_rad)
        
        return await self.goto(new_x, new_y, state.position.z, speed)
    
    async def move_backward(self, distance: float, speed: float = 1.0) -> bool:
        """Move backward relative to current heading."""
        return await self.move_forward(-distance, speed)
    
    async def move_left(self, distance: float, speed: float = 1.0) -> bool:
        """Move left relative to current heading."""
        state = await self.get_state()
        yaw_rad = np.radians(state.orientation.yaw - 90)
        
        new_x = state.position.x + distance * np.cos(yaw_rad)
        new_y = state.position.y + distance * np.sin(yaw_rad)
        
        return await self.goto(new_x, new_y, state.position.z, speed)
    
    async def move_right(self, distance: float, speed: float = 1.0) -> bool:
        """Move right relative to current heading."""
        return await self.move_left(-distance, speed)
