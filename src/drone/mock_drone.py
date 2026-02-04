"""
Mock Drone Implementation

Simulates a drone for testing without real hardware or Isaac Sim.
Generates synthetic telemetry and camera images.
"""

import asyncio
import random
from datetime import datetime
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .interface import (
    DroneInterface,
    DroneState,
    DroneStatus,
    Position,
    Orientation,
)


class MockDrone(DroneInterface):
    """
    Mock drone for testing the AI command system.
    
    Simulates realistic drone behavior including:
    - Movement with configurable speed
    - Battery drain
    - Synthetic camera images with telemetry overlay
    - State tracking
    """
    
    def __init__(
        self,
        simulated_delay: float = 0.1,
        movement_speed: float = 2.0,
        battery_drain_rate: float = 0.01,
    ):
        """
        Initialize mock drone.
        
        Args:
            simulated_delay: Base delay for operations (seconds).
            movement_speed: Default movement speed (m/s).
            battery_drain_rate: Battery drain per operation (%).
        """
        self.simulated_delay = simulated_delay
        self.movement_speed = movement_speed
        self.battery_drain_rate = battery_drain_rate
        
        # Internal state
        self._state = DroneState(
            status=DroneStatus.DISCONNECTED,
            position=Position(0, 0, 0),
            orientation=Orientation(0, 0, 0),
            velocity=Position(0, 0, 0),
            battery_percent=100.0,
            gps_satellites=0,
        )
        
        # Movement tracking
        self._target_position: Optional[Position] = None
        self._is_moving = False
        
        # Image counter for unique filenames
        self._image_counter = 0
    
    async def _simulate_delay(self, multiplier: float = 1.0):
        """Add simulated processing delay."""
        await asyncio.sleep(self.simulated_delay * multiplier)
    
    def _drain_battery(self, amount: float = 1.0):
        """Simulate battery drain."""
        self._state.battery_percent = max(
            0, 
            self._state.battery_percent - (self.battery_drain_rate * amount)
        )
    
    def _update_timestamp(self):
        """Update state timestamp."""
        self._state.timestamp = datetime.now()
    
    async def connect(self) -> bool:
        """Establish connection to mock drone."""
        await self._simulate_delay(2.0)
        
        if self._state.status != DroneStatus.DISCONNECTED:
            return True
        
        self._state.status = DroneStatus.CONNECTED
        self._state.gps_satellites = random.randint(8, 14)
        self._state.battery_percent = random.uniform(85, 100)
        self._update_timestamp()
        
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from mock drone."""
        await self._simulate_delay()
        
        if self._state.status == DroneStatus.FLYING:
            await self.land()
        
        self._state.status = DroneStatus.DISCONNECTED
        self._state.gps_satellites = 0
        self._update_timestamp()
        
        return True
    
    async def arm(self) -> bool:
        """Arm mock drone motors."""
        await self._simulate_delay()
        
        if self._state.status != DroneStatus.CONNECTED:
            return False
        
        self._state.status = DroneStatus.ARMED
        self._drain_battery(0.5)
        self._update_timestamp()
        
        return True
    
    async def disarm(self) -> bool:
        """Disarm mock drone motors."""
        await self._simulate_delay()
        
        if self._state.status == DroneStatus.FLYING:
            return False  # Cannot disarm while flying
        
        if self._state.status == DroneStatus.ARMED:
            self._state.status = DroneStatus.CONNECTED
        
        self._update_timestamp()
        return True
    
    async def takeoff(self, altitude: float = 2.0) -> bool:
        """Take off to specified altitude."""
        await self._simulate_delay()
        
        if self._state.status != DroneStatus.ARMED:
            # Auto-arm if connected
            if self._state.status == DroneStatus.CONNECTED:
                await self.arm()
            else:
                return False
        
        self._state.status = DroneStatus.FLYING
        
        # Simulate gradual ascent
        target_z = -altitude
        steps = int(abs(target_z - self._state.position.z) / 0.5) + 1
        
        for i in range(steps):
            progress = (i + 1) / steps
            self._state.position.z = self._state.position.z + (target_z - self._state.position.z) * progress
            self._state.velocity.z = -0.5  # Ascending
            self._drain_battery(0.2)
            await self._simulate_delay(0.5)
        
        self._state.position.z = target_z
        self._state.velocity = Position(0, 0, 0)
        self._update_timestamp()
        
        return True
    
    async def land(self) -> bool:
        """Land the drone."""
        await self._simulate_delay()
        
        if not self._state.is_flying:
            return True
        
        self._state.status = DroneStatus.LANDING
        
        # Simulate gradual descent
        steps = int(abs(self._state.position.z) / 0.5) + 1
        
        for i in range(steps):
            progress = (i + 1) / steps
            self._state.position.z = self._state.position.z * (1 - progress)
            self._state.velocity.z = 0.5  # Descending
            self._drain_battery(0.1)
            await self._simulate_delay(0.5)
        
        self._state.position.z = 0
        self._state.velocity = Position(0, 0, 0)
        self._state.status = DroneStatus.ARMED
        self._update_timestamp()
        
        return True
    
    async def goto(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        speed: float = 1.0
    ) -> bool:
        """Navigate to specified position."""
        await self._simulate_delay()
        
        if not self._state.is_flying:
            return False
        
        # Use current altitude if not specified
        target_z = z if z is not None else self._state.position.z
        target = Position(x, y, target_z)
        
        # Calculate distance and time
        distance = self._state.position.distance_to(target)
        travel_time = distance / speed
        steps = max(int(travel_time / 0.1), 1)
        
        # Calculate velocity
        direction = target.to_array() - self._state.position.to_array()
        if distance > 0:
            velocity = (direction / distance) * speed
            self._state.velocity = Position.from_array(velocity)
        
        # Simulate movement
        start_pos = self._state.position.to_array()
        for i in range(steps):
            progress = (i + 1) / steps
            new_pos = start_pos + (target.to_array() - start_pos) * progress
            self._state.position = Position.from_array(new_pos)
            self._drain_battery(0.05)
            await self._simulate_delay(0.1)
        
        self._state.position = target
        self._state.velocity = Position(0, 0, 0)
        self._update_timestamp()
        
        return True
    
    async def rotate(self, yaw: float, relative: bool = True) -> bool:
        """Rotate to specified heading."""
        await self._simulate_delay()
        
        if not self._state.is_flying and self._state.status != DroneStatus.ARMED:
            return False
        
        if relative:
            target_yaw = (self._state.orientation.yaw + yaw) % 360
        else:
            target_yaw = yaw % 360
        
        # Simulate rotation (instant for mock)
        self._state.orientation.yaw = target_yaw
        self._drain_battery(0.02)
        self._update_timestamp()
        
        return True
    
    async def hover(self) -> bool:
        """Stop and hover in place."""
        await self._simulate_delay()
        
        self._state.velocity = Position(0, 0, 0)
        self._target_position = None
        self._is_moving = False
        self._update_timestamp()
        
        return True
    
    async def get_state(self) -> DroneState:
        """Get current drone state."""
        self._update_timestamp()
        
        # Add some noise to simulate real sensors
        if self._state.is_flying:
            self._state.orientation.roll = random.gauss(0, 0.5)
            self._state.orientation.pitch = random.gauss(0, 0.5)
        
        return self._state
    
    async def capture_image(self) -> Optional[Image.Image]:
        """
        Generate synthetic camera image.
        
        Creates a simulated aerial view with telemetry overlay.
        """
        await self._simulate_delay(0.5)
        
        if self._state.status == DroneStatus.DISCONNECTED:
            return None
        
        self._image_counter += 1
        self._drain_battery(0.1)
        
        # Create synthetic image
        width, height = 640, 480
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        # Generate terrain-like background
        for y in range(height):
            for x in range(width):
                # Create noise-based terrain colors
                noise = random.random()
                altitude = self._state.position.altitude
                
                # Adjust colors based on altitude
                if altitude < 1:
                    # Close to ground - more detail
                    if noise < 0.3:
                        color = (34, 139, 34)  # Forest green
                    elif noise < 0.5:
                        color = (85, 107, 47)  # Dark olive green
                    elif noise < 0.7:
                        color = (107, 142, 35)  # Olive drab
                    else:
                        color = (154, 205, 50)  # Yellow green
                else:
                    # Higher altitude - more uniform
                    base_green = int(80 + noise * 60)
                    color = (30 + int(noise * 20), base_green, 30)
                
                image.putpixel((x, y), color)
        
        # Add grid pattern (simulating fields/structures)
        grid_spacing = max(20, int(100 / max(altitude, 1)))
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=(50, 80, 50), width=1)
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=(50, 80, 50), width=1)
        
        # Add some random "features"
        num_features = random.randint(3, 8)
        for _ in range(num_features):
            fx = random.randint(50, width - 50)
            fy = random.randint(50, height - 50)
            size = random.randint(10, 30)
            feature_type = random.choice(['circle', 'rect'])
            color = random.choice([
                (139, 69, 19),   # Brown (building)
                (70, 130, 180),  # Steel blue (water)
                (128, 128, 128), # Gray (road)
            ])
            if feature_type == 'circle':
                draw.ellipse([fx - size, fy - size, fx + size, fy + size], fill=color)
            else:
                draw.rectangle([fx - size, fy - size // 2, fx + size, fy + size // 2], fill=color)
        
        # Add telemetry overlay
        overlay_color = (255, 255, 255)
        shadow_color = (0, 0, 0)
        
        # Try to use a font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Telemetry text
        telemetry = [
            f"ALT: {self._state.position.altitude:.1f}m",
            f"POS: ({self._state.position.x:.1f}, {self._state.position.y:.1f})",
            f"HDG: {self._state.orientation.yaw:.0f}°",
            f"BAT: {self._state.battery_percent:.0f}%",
            f"GPS: {self._state.gps_satellites} sats",
            f"IMG: #{self._image_counter:04d}",
        ]
        
        y_offset = 10
        for text in telemetry:
            # Shadow
            draw.text((11, y_offset + 1), text, fill=shadow_color, font=small_font)
            # Text
            draw.text((10, y_offset), text, fill=overlay_color, font=small_font)
            y_offset += 18
        
        # Add crosshair in center
        cx, cy = width // 2, height // 2
        crosshair_size = 20
        draw.line([(cx - crosshair_size, cy), (cx + crosshair_size, cy)], fill=(255, 0, 0), width=2)
        draw.line([(cx, cy - crosshair_size), (cx, cy + crosshair_size)], fill=(255, 0, 0), width=2)
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=(255, 0, 0), width=2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((width - 150, height - 25), timestamp, fill=overlay_color, font=small_font)
        
        # Add drone indicator
        draw.text((width - 100, 10), "MOCK DRONE", fill=(255, 200, 0), font=small_font)
        
        self._update_timestamp()
        return image
    
    def __repr__(self) -> str:
        return f"MockDrone(status={self._state.status.value}, pos={self._state.position})"
