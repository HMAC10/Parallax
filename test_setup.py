"""
Test Setup Script

Verifies that all modules are properly installed and configured.
Runs a basic test with the mock drone to ensure everything works.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_status(name: str, success: bool, message: str = ""):
    """Print status with color."""
    # Use ASCII-safe characters for Windows compatibility
    status = "[OK]" if success else "[FAIL]"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    msg = f" - {message}" if message else ""
    try:
        print(f"  {color}{status}{reset} {name}{msg}")
    except UnicodeEncodeError:
        # Fallback for Windows consoles that don't support ANSI
        print(f"  {status} {name}{msg}")


async def test_imports():
    """Test that all modules can be imported."""
    print_header("Testing Imports")
    
    all_passed = True
    
    # Test config
    try:
        from src.config import settings, validate_config
        print_status("src.config", True, f"Settings loaded, API key set: {bool(settings.nvidia_api_key)}")
    except Exception as e:
        print_status("src.config", False, str(e))
        all_passed = False
    
    # Test drone interface
    try:
        from src.drone import DroneInterface, DroneState, Position, Orientation
        print_status("src.drone.interface", True)
    except Exception as e:
        print_status("src.drone.interface", False, str(e))
        all_passed = False
    
    # Test mock drone
    try:
        from src.drone import MockDrone
        print_status("src.drone.mock_drone", True)
    except Exception as e:
        print_status("src.drone.mock_drone", False, str(e))
        all_passed = False
    
    # Test LLM clients
    try:
        from src.llm import NemotronClient, DroneCommand, CommandType
        print_status("src.llm.nemotron", True)
    except Exception as e:
        print_status("src.llm.nemotron", False, str(e))
        all_passed = False
    
    try:
        from src.llm import VILAClient, ImageAnalysis
        print_status("src.llm.vila", True)
    except Exception as e:
        print_status("src.llm.vila", False, str(e))
        all_passed = False
    
    # Test agents
    try:
        from src.agents import AgentState, Mission, Waypoint, MissionType
        print_status("src.agents", True)
    except Exception as e:
        print_status("src.agents", False, str(e))
        all_passed = False
    
    # Test optimization
    try:
        from src.optimization import PathOptimizer, EnergyOptimizer, Point3D
        print_status("src.optimization", True)
    except Exception as e:
        print_status("src.optimization", False, str(e))
        all_passed = False
    
    return all_passed


async def test_mock_drone():
    """Test basic mock drone operations."""
    print_header("Testing Mock Drone")
    
    from src.drone import MockDrone, DroneStatus
    
    drone = MockDrone(simulated_delay=0.05)  # Fast for testing
    all_passed = True
    
    try:
        # Test connection
        result = await drone.connect()
        state = await drone.get_state()
        print_status("Connect", result and state.status == DroneStatus.CONNECTED, 
                    f"Status: {state.status.value}")
        all_passed = all_passed and result
        
        # Test arm
        result = await drone.arm()
        state = await drone.get_state()
        print_status("Arm", result and state.status == DroneStatus.ARMED,
                    f"Status: {state.status.value}")
        all_passed = all_passed and result
        
        # Test takeoff
        result = await drone.takeoff(altitude=5.0)
        state = await drone.get_state()
        print_status("Takeoff", result and state.is_flying,
                    f"Altitude: {state.position.altitude:.1f}m")
        all_passed = all_passed and result
        
        # Test goto
        result = await drone.goto(10, 5, speed=2.0)
        state = await drone.get_state()
        print_status("Goto", result,
                    f"Position: ({state.position.x:.1f}, {state.position.y:.1f})")
        all_passed = all_passed and result
        
        # Test rotate
        result = await drone.rotate(90)
        state = await drone.get_state()
        print_status("Rotate", result,
                    f"Heading: {state.orientation.yaw:.0f}°")
        all_passed = all_passed and result
        
        # Test capture
        image = await drone.capture_image()
        print_status("Capture Image", image is not None,
                    f"Size: {image.size if image else 'N/A'}")
        all_passed = all_passed and (image is not None)
        
        # Test hover
        result = await drone.hover()
        print_status("Hover", result)
        all_passed = all_passed and result
        
        # Test land
        result = await drone.land()
        state = await drone.get_state()
        print_status("Land", result and state.position.altitude < 0.1,
                    f"Altitude: {state.position.altitude:.2f}m")
        all_passed = all_passed and result
        
        # Test disconnect
        result = await drone.disconnect()
        state = await drone.get_state()
        print_status("Disconnect", result and state.status == DroneStatus.DISCONNECTED,
                    f"Status: {state.status.value}")
        all_passed = all_passed and result
        
    except Exception as e:
        print_status("Mock Drone Test", False, str(e))
        all_passed = False
    
    return all_passed


async def test_path_optimization():
    """Test path optimization functions."""
    print_header("Testing Path Optimization")
    
    from src.optimization import PathOptimizer, EnergyOptimizer, Point3D
    
    all_passed = True
    
    try:
        optimizer = PathOptimizer(max_speed=5.0)
        
        # Test survey pattern generation
        center = Point3D(0, 0, -20)  # 20m altitude
        waypoints = optimizer.generate_survey_pattern(
            center, width=50, height=50, altitude=-20, spacing=10
        )
        print_status("Survey Pattern", len(waypoints) > 0,
                    f"Generated {len(waypoints)} waypoints")
        all_passed = all_passed and len(waypoints) > 0
        
        # Test path length calculation
        length = optimizer.calculate_path_length(waypoints)
        print_status("Path Length", length > 0, f"{length:.1f}m")
        all_passed = all_passed and length > 0
        
        # Test flight time estimation
        time = optimizer.estimate_flight_time(waypoints)
        print_status("Flight Time", time > 0, f"{time:.1f}s ({time/60:.1f}min)")
        all_passed = all_passed and time > 0
        
        # Test waypoint order optimization
        test_points = [
            Point3D(0, 0, -20),
            Point3D(100, 0, -20),
            Point3D(50, 50, -20),
            Point3D(0, 100, -20),
        ]
        optimized = optimizer.optimize_waypoint_order(test_points)
        print_status("Waypoint Optimization", len(optimized) == len(test_points))
        
        # Test energy estimation
        energy_opt = EnergyOptimizer()
        energy = energy_opt.estimate_energy(waypoints)
        print_status("Energy Estimation", "total_wh" in energy,
                    f"{energy['total_wh']:.2f}Wh, Feasible: {energy['feasible']}")
        all_passed = all_passed and "total_wh" in energy
        
        # Test circle pattern
        circle = optimizer.generate_circle_pattern(center, radius=25, altitude=-20, num_points=8)
        print_status("Circle Pattern", len(circle) > 0,
                    f"Generated {len(circle)} waypoints")
        
    except Exception as e:
        print_status("Path Optimization", False, str(e))
        all_passed = False
    
    return all_passed


async def test_llm_clients_init():
    """Test LLM client initialization (no actual API calls)."""
    print_header("Testing LLM Client Initialization")
    
    from src.llm import NemotronClient, VILAClient
    from src.config import settings
    
    all_passed = True
    
    try:
        # Test Nemotron client init
        nemotron = NemotronClient()
        print_status("NemotronClient Init", True,
                    f"Model: {nemotron.model}")
        await nemotron.close()
        
        # Test VILA client init
        vila = VILAClient()
        print_status("VILAClient Init", True,
                    f"Model: {vila.model}")
        await vila.close()
        
        # Check API configuration
        if settings.is_configured:
            print_status("API Configuration", True, "API key is configured")
        else:
            print_status("API Configuration", False, 
                        "API key not set (set NVIDIA_API_KEY in .env)")
            # Don't fail the test, just warn
        
    except Exception as e:
        print_status("LLM Client Init", False, str(e))
        all_passed = False
    
    return all_passed


async def test_agent_types():
    """Test agent type definitions."""
    print_header("Testing Agent Types")
    
    from src.agents import AgentState, Mission, Waypoint, MissionType
    from datetime import datetime
    
    all_passed = True
    
    try:
        # Test waypoint creation
        wp = Waypoint(x=10, y=20, altitude=15, action="capture")
        print_status("Waypoint", wp.x == 10 and wp.altitude == 15)
        
        # Test mission creation
        mission = Mission(
            name="Test Survey",
            mission_type=MissionType.SURVEY,
        )
        mission.add_waypoint(wp)
        mission.add_waypoint(Waypoint(x=20, y=30, altitude=15))
        
        print_status("Mission", len(mission.waypoints) == 2,
                    f"Type: {mission.mission_type.value}, Waypoints: {len(mission.waypoints)}")
        
        # Test mission to dict
        mission_dict = mission.to_dict()
        print_status("Mission Serialization", "waypoints" in mission_dict)
        
        # Test agent state type
        state: AgentState = {
            "user_input": "test command",
            "drone_connected": True,
            "battery_level": 85.0,
        }
        print_status("AgentState", state["user_input"] == "test command")
        
    except Exception as e:
        print_status("Agent Types", False, str(e))
        all_passed = False
    
    return all_passed


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("   DRONE AI COMMAND SYSTEM - Setup Test")
    print("   NVIDIA GTC 2026 Competition")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Imports", await test_imports()))
    results.append(("Mock Drone", await test_mock_drone()))
    results.append(("Path Optimization", await test_path_optimization()))
    results.append(("LLM Clients", await test_llm_clients_init()))
    results.append(("Agent Types", await test_agent_types()))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        print_status(name, result)
    
    print(f"\n  Total: {passed}/{total} test groups passed")
    
    if passed == total:
        try:
            print("\n  \033[92m[SUCCESS] All tests passed! Setup is complete.\033[0m")
        except UnicodeEncodeError:
            print("\n  [SUCCESS] All tests passed! Setup is complete.")
        print("\n  Next steps:")
        print("    1. Copy .env.example to .env")
        print("    2. Add your NVIDIA API key to .env")
        print("    3. Run: python -m src.config  (to verify config)")
        print()
        return 0
    else:
        try:
            print("\n  \033[91m[FAILED] Some tests failed. Check the output above.\033[0m")
        except UnicodeEncodeError:
            print("\n  [FAILED] Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
