"""
LangGraph-based AI Agents for drone control.

Provides intelligent agents that combine:
- Natural language understanding (Nemotron)
- Visual perception (VILA)
- Mission planning and execution
- Autonomous decision making
"""

from typing import TypedDict, Annotated, Sequence, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class AgentState(TypedDict, total=False):
    """State passed between agent nodes in the graph."""
    # Input
    user_input: str
    
    # Parsed command
    command_type: str
    command_params: dict
    confidence: float
    
    # Drone state
    drone_connected: bool
    drone_position: dict
    drone_status: str
    battery_level: float
    
    # Mission state
    mission_active: bool
    mission_waypoints: list
    current_waypoint_index: int
    
    # Image analysis
    last_image_analysis: dict
    images_captured: int
    
    # Execution
    action_result: str
    error_message: str
    
    # History
    messages: list
    execution_log: list


class MissionType(Enum):
    """Types of autonomous missions."""
    SURVEY = "survey"
    SEARCH = "search"
    PATROL = "patrol"
    INSPECTION = "inspection"
    DELIVERY = "delivery"
    MAPPING = "mapping"
    CUSTOM = "custom"


@dataclass
class Waypoint:
    """A waypoint in a mission."""
    x: float
    y: float
    altitude: float
    action: str = "flythrough"  # flythrough, hover, capture, inspect
    duration: float = 0.0  # seconds to stay if hover
    heading: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "altitude": self.altitude,
            "action": self.action,
            "duration": self.duration,
            "heading": self.heading,
        }


@dataclass
class Mission:
    """A complete drone mission."""
    name: str
    mission_type: MissionType
    waypoints: list[Waypoint] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "created"  # created, running, paused, completed, aborted
    
    def add_waypoint(self, waypoint: Waypoint):
        """Add a waypoint to the mission."""
        self.waypoints.append(waypoint)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.mission_type.value,
            "waypoints": [w.to_dict() for w in self.waypoints],
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# Agent node functions will be implemented in separate files
# This module provides the base types and state definitions

__all__ = [
    "AgentState",
    "MissionType",
    "Waypoint",
    "Mission",
]
