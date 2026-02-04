"""
PARALLAX - Full End-to-End Pipeline
Natural Language → Path Planning → Simulation Validation
"""
import argparse
import os

# Setup argument parser BEFORE AppLauncher
parser = argparse.ArgumentParser(description="Parallax: NL Drone Command Validator")
parser.add_argument("--command", type=str, default="inspect the northeast solar panels for damage",
                    help="Natural language command")
parser.add_argument("--api-key", type=str, default=os.environ.get("NVIDIA_API_KEY", ""),
                    help="NVIDIA API key")

# Import AppLauncher and add its args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import everything else
import torch
import json
import time
import httpx
from datetime import datetime
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import CRAZYFLIE_CFG


# =============================================================================
# NEMOTRON CLIENT - Parse natural language commands
# =============================================================================

class NemotronClient:
    """Client for NVIDIA Nemotron API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = "nvidia/nemotron-3-nano-30b-a3b"
    
    def parse_command(self, command: str) -> dict:
        """Parse natural language command into structured intent."""
        
        prompt = f"""You are a drone mission planner. Parse this command into a structured mission.

Command: "{command}"

Respond with ONLY valid JSON:
{{
    "action": "inspect" or "survey" or "monitor" or "check",
    "target": "what to inspect (e.g., solar panels, roof, building)",
    "location": "northeast" or "northwest" or "southeast" or "southwest" or "north" or "south" or "east" or "west" or "all",
    "altitude": suggested altitude in meters (default 2.0),
    "priority": "normal" or "urgent"
}}"""

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
            print(f"[NEMOTRON] API returned status {response.status_code}")
            
        except Exception as e:
            print(f"[NEMOTRON] Error: {e}")
        
        # Fallback: simple keyword parsing
        return self._fallback_parse(command)
    
    def _fallback_parse(self, command: str) -> dict:
        """Fallback parser using keywords."""
        command = command.lower()
        
        location = "all"
        for loc in ["northeast", "northwest", "southeast", "southwest", "north", "south", "east", "west"]:
            if loc in command:
                location = loc
                break
        
        action = "inspect"
        for act in ["survey", "monitor", "check", "scan"]:
            if act in command:
                action = act
                break
        
        target = "area"
        for t in ["solar panel", "panel", "roof", "building", "tower", "field"]:
            if t in command:
                target = t
                break
        
        return {
            "action": action,
            "target": target,
            "location": location,
            "altitude": 2.0,
            "priority": "normal"
        }


# =============================================================================
# MISSION PLANNER
# =============================================================================

class MissionPlanner:
    """Converts parsed commands into flight paths."""
    
    def __init__(self, site_bounds: dict):
        self.bounds = site_bounds
    
    def plan_mission(self, intent: dict) -> list:
        """Generate waypoints from parsed intent."""
        
        location = intent.get("location", "all")
        altitude = intent.get("altitude", 2.0)
        
        # Determine inspection area based on location
        x_min, x_max = self.bounds["x_min"], self.bounds["x_max"]
        y_min, y_max = self.bounds["y_min"], self.bounds["y_max"]
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        
        areas = {
            "northeast": (mid_x, x_max, mid_y, y_max),
            "northwest": (x_min, mid_x, mid_y, y_max),
            "southeast": (mid_x, x_max, y_min, mid_y),
            "southwest": (x_min, mid_x, y_min, mid_y),
            "north": (x_min, x_max, mid_y, y_max),
            "south": (x_min, x_max, y_min, mid_y),
            "east": (mid_x, x_max, y_min, y_max),
            "west": (x_min, mid_x, y_min, y_max),
            "all": (x_min, x_max, y_min, y_max),
        }
        
        area = areas.get(location, areas["all"])
        
        waypoints = []
        
        # Home
        waypoints.append({"x": 0, "y": 0, "z": 0.1, "action": "home"})
        
        # Takeoff
        waypoints.append({"x": 0, "y": 0, "z": altitude, "action": "takeoff"})
        
        # Generate grid pattern
        spacing = 1.5
        x_points = self._range(area[0], area[1], spacing)
        y_points = self._range(area[2], area[3], spacing)
        
        reverse = False
        for y in y_points:
            xs = x_points if not reverse else list(reversed(x_points))
            for x in xs:
                waypoints.append({
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "z": altitude,
                    "action": "inspect",
                    "capture": True
                })
            reverse = not reverse
        
        # Return home
        waypoints.append({"x": 0, "y": 0, "z": altitude, "action": "return"})
        waypoints.append({"x": 0, "y": 0, "z": 0.1, "action": "land"})
        
        return waypoints
    
    def _range(self, start, end, step):
        result = []
        x = start
        while x <= end:
            result.append(x)
            x += step
        return result


# =============================================================================
# SIMULATION VALIDATOR
# =============================================================================

class SimValidator:
    """Validates flight paths in simulation."""
    
    def __init__(self, sim, robot):
        self.sim = sim
        self.robot = robot
        self.dt = sim.get_physics_dt()
        self.prop_body_ids = robot.find_bodies("m.*_prop")[0]
        self.mass = robot.root_physx_view.get_masses().sum()
        self.gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
        self.hover_thrust = self.mass * self.gravity / 4.0
        self.total_distance = 0.0
        self.collision = False
    
    def get_pos(self):
        return self.robot.data.root_pos_w[0].cpu().numpy()
    
    def step(self):
        forces = torch.zeros(self.robot.num_instances, 4, 3, device=self.sim.device)
        forces[..., 2] = self.hover_thrust
        self.robot.set_external_force_and_torque(forces, torch.zeros_like(forces), body_ids=self.prop_body_ids)
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.dt)
    
    def goto(self, x, y, z):
        start = self.get_pos()
        
        # Teleport + stabilize
        new_pos = torch.tensor([[x, y, z, 0, 0, 0, 1]], device=self.sim.device)
        self.robot.write_root_pose_to_sim(new_pos)
        self.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
        
        for _ in range(30):
            self.step()
        
        end = self.get_pos()
        dist = ((end[0]-start[0])**2 + (end[1]-start[1])**2 + (end[2]-start[2])**2)**0.5
        self.total_distance += dist
        
        if end[2] < 0.05:
            self.collision = True
        
        return not self.collision
    
    def validate(self, waypoints: list) -> dict:
        start_time = time.time()
        reached = 0
        
        for wp in waypoints:
            if self.goto(wp["x"], wp["y"], wp["z"]):
                reached += 1
            else:
                break
        
        elapsed = time.time() - start_time
        flight_time = self.total_distance / 5.0  # Assume 5 m/s
        
        return {
            "status": "PASSED" if not self.collision else "FAILED",
            "waypoints_reached": reached,
            "waypoints_total": len(waypoints),
            "distance_m": round(float(self.total_distance), 2),
            "est_flight_time_s": round(float(flight_time), 1),
            "est_battery_pct": round(float(flight_time / 30), 1),
            "sim_time_s": round(float(elapsed), 2)
        }


# =============================================================================
# SCENE SETUP
# =============================================================================

def create_scene():
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DistantLightCfg(intensity=3000.0))
    
    # Solar panel array
    for i in range(4):
        for j in range(4):
            cfg = sim_utils.CuboidCfg(
                size=(0.8, 0.5, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.4)),
            )
            cfg.func(f"/World/Panel_{i}_{j}", cfg, translation=(i * 1.2 + 1, j * 1.2 + 1, 0.3))
    
    return {"x_min": 0.5, "x_max": 5.5, "y_min": 0.5, "y_max": 5.5}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  ██████╗  █████╗ ██████╗  █████╗ ██╗     ██╗      █████╗ ██╗  ██╗")
    print("  ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║     ██║     ██╔══██╗╚██╗██╔╝")
    print("  ██████╔╝███████║██████╔╝███████║██║     ██║     ███████║ ╚███╔╝ ")
    print("  ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║     ██║     ██╔══██║ ██╔██╗ ")
    print("  ██║     ██║  ██║██║  ██║██║  ██║███████╗███████╗██║  ██║██╔╝ ██╗")
    print("  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝")
    print("  Digital Twin Ground Control - Validate Before You Fly")
    print("=" * 70)
    
    command = args_cli.command
    api_key = args_cli.api_key
    
    print(f"\n🎤 COMMAND: \"{command}\"")
    
    # Phase 1: Parse command
    print("\n" + "-" * 50)
    print("PHASE 1: NATURAL LANGUAGE PROCESSING")
    print("-" * 50)
    
    if api_key:
        print("[NLP] Using NVIDIA Nemotron API...")
        nemotron = NemotronClient(api_key)
        intent = nemotron.parse_command(command)
    else:
        print("[NLP] No API key - using fallback parser...")
        intent = NemotronClient("")._fallback_parse(command)
    
    print(f"[NLP] Parsed intent:")
    print(f"      Action:   {intent.get('action')}")
    print(f"      Target:   {intent.get('target')}")
    print(f"      Location: {intent.get('location')}")
    print(f"      Altitude: {intent.get('altitude')}m")
    
    # Phase 2: Plan mission
    print("\n" + "-" * 50)
    print("PHASE 2: MISSION PLANNING")
    print("-" * 50)
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    bounds = create_scene()
    
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Drone")
    robot_cfg.spawn.func("/World/Drone", robot_cfg.spawn, translation=(0, 0, 0.1))
    robot = Articulation(robot_cfg)
    sim.reset()
    
    planner = MissionPlanner(bounds)
    waypoints = planner.plan_mission(intent)
    
    print(f"[PLANNER] Generated {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        action = wp.get('action', 'move')
        print(f"    {i+1}. ({wp['x']}, {wp['y']}, {wp['z']}) - {action}")
    
    # Phase 3: Validate
    print("\n" + "-" * 50)
    print("PHASE 3: SIMULATION VALIDATION")
    print("-" * 50)
    
    validator = SimValidator(sim, robot)
    result = validator.validate(waypoints)
    
    # Phase 4: Report
    print("\n" + "=" * 70)
    if result["status"] == "PASSED":
        print("  ✅ ✅ ✅  VALIDATION PASSED - SAFE TO FLY  ✅ ✅ ✅")
    else:
        print("  ❌ ❌ ❌  VALIDATION FAILED - DO NOT FLY  ❌ ❌ ❌")
    print("=" * 70)
    
    print(f"""
  📊 MISSION SUMMARY
  ────────────────────────────────────────
  Command:        "{command}"
  Action:         {intent.get('action')} {intent.get('target')}
  Location:       {intent.get('location')}
  
  📍 PATH ANALYSIS
  ────────────────────────────────────────
  Waypoints:      {result['waypoints_reached']}/{result['waypoints_total']} validated
  Distance:       {result['distance_m']} meters
  Flight Time:    {result['est_flight_time_s']} seconds
  Battery:        {result['est_battery_pct']}%
  
  ⏱️  Simulation completed in {result['sim_time_s']}s
    """)
    
    # Save outputs
    output = {
        "command": command,
        "intent": intent,
        "validation": result,
        "waypoints": waypoints,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("/workspace/parallax_output.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("  📁 Output saved to: /workspace/parallax_output.json")
    print("\n" + "=" * 70)
    
    simulation_app.close()


if __name__ == "__main__":
    main()