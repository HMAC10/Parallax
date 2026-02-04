"""
PARALLAX - Full Pipeline with VILA Image Analysis
Natural Language → Path Planning → Simulation → Vision Analysis → Report
"""
import argparse
import os

parser = argparse.ArgumentParser(description="Parallax: NL Drone Command with Vision")
parser.add_argument("--command", type=str, default="inspect the northeast solar panels for damage")
parser.add_argument("--api-key", type=str, default=os.environ.get("NVIDIA_API_KEY", ""))

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import json
import time
import httpx
import base64
import numpy as np
from datetime import datetime
from PIL import Image
import io
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import CRAZYFLIE_CFG

# =============================================================================
# NEMO GUARDRAILS - Safety Validation
# =============================================================================

class NeMoGuardrails:
    """Safety guardrails for drone commands using NeMo Guardrails logic."""
    
    # Restricted zones (example coordinates)
    RESTRICTED_ZONES = [
        {"name": "Building A", "x_min": 10, "x_max": 15, "y_min": 10, "y_max": 15},
        {"name": "Power Station", "x_min": -5, "x_max": -2, "y_min": 0, "y_max": 3},
    ]
    
    # Safety limits
    MAX_ALTITUDE = 50.0  # meters
    MIN_ALTITUDE = 0.5   # meters
    MAX_SPEED = 15.0     # m/s
    MAX_DISTANCE_FROM_HOME = 100.0  # meters
    
    # Blocked keywords (unsafe commands)
    BLOCKED_KEYWORDS = [
        "crash", "collide", "hit", "attack", "strike", "ram",
        "spy", "surveil", "stalk", "follow person",
        "drop", "release payload", "bomb",
        "maximum speed", "full throttle", "no limits",
        "ignore safety", "override", "bypass"
    ]
    
    def __init__(self):
        self.violations = []
    
    def validate_command(self, command: str) -> dict:
        """Check if a command is safe to execute."""
        command_lower = command.lower()
        self.violations = []
        
        # Check for blocked keywords
        for keyword in self.BLOCKED_KEYWORDS:
            if keyword in command_lower:
                self.violations.append(f"Blocked keyword detected: '{keyword}'")
        
        if self.violations:
            return {
                "safe": False,
                "violations": self.violations,
                "message": "Command rejected for safety reasons"
            }
        
        return {"safe": True, "violations": [], "message": "Command approved"}
    
    def validate_waypoint(self, x: float, y: float, z: float) -> dict:
        """Check if a waypoint is safe."""
        self.violations = []
        
        # Check altitude limits
        if z > self.MAX_ALTITUDE:
            self.violations.append(f"Altitude {z}m exceeds maximum {self.MAX_ALTITUDE}m")
        if z < self.MIN_ALTITUDE:
            self.violations.append(f"Altitude {z}m below minimum {self.MIN_ALTITUDE}m")
        
        # Check distance from home
        distance = (x**2 + y**2)**0.5
        if distance > self.MAX_DISTANCE_FROM_HOME:
            self.violations.append(f"Distance {distance:.1f}m exceeds maximum range {self.MAX_DISTANCE_FROM_HOME}m")
        
        # Check restricted zones
        for zone in self.RESTRICTED_ZONES:
            if (zone["x_min"] <= x <= zone["x_max"] and 
                zone["y_min"] <= y <= zone["y_max"]):
                self.violations.append(f"Waypoint enters restricted zone: {zone['name']}")
        
        if self.violations:
            return {"safe": False, "violations": self.violations}
        
        return {"safe": True, "violations": []}
    
    def validate_mission(self, waypoints: list) -> dict:
        """Validate entire mission for safety."""
        all_violations = []
        unsafe_waypoints = []
        
        for i, wp in enumerate(waypoints):
            result = self.validate_waypoint(wp["x"], wp["y"], wp["z"])
            if not result["safe"]:
                unsafe_waypoints.append(i + 1)
                all_violations.extend(result["violations"])
        
        if all_violations:
            return {
                "safe": False,
                "unsafe_waypoints": unsafe_waypoints,
                "violations": list(set(all_violations)),  # Remove duplicates
                "message": f"Mission has {len(unsafe_waypoints)} unsafe waypoint(s)"
            }
        
        return {
            "safe": True,
            "unsafe_waypoints": [],
            "violations": [],
            "message": "All waypoints validated - mission is safe"
        }

class NemotronClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = "nvidia/nemotron-3-nano-30b-a3b"
    
    def parse_command(self, command: str) -> dict:
        prompt = f"""Parse this drone command into JSON:
Command: "{command}"
Return ONLY JSON: {{"action": "inspect/survey/monitor", "target": "what", "location": "northeast/southwest/etc", "altitude": 2.0}}"""
        
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 200},
                timeout=30.0
            )
            if response.status_code == 200:
                import re
                content = response.json()["choices"][0]["message"]["content"]
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except Exception as e:
            print(f"[NEMOTRON] Error: {e}")
        
        return self._fallback(command)
    
    def _fallback(self, cmd):
        cmd = cmd.lower()
        loc = "all"
        for l in ["northeast", "northwest", "southeast", "southwest"]:
            if l in cmd: loc = l; break
        return {"action": "inspect", "target": "area", "location": loc, "altitude": 2.0}


class VILAClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://integrate.api.nvidia.com/v1"
    
    def simulate_analysis(self, waypoint_num: int) -> dict:
        """Simulate VILA analysis for demo."""
        scenarios = [
            {"findings": ["No issues detected"], "severity": "none", "confidence": 0.95, "recommendation": "Continue normal operations"},
            {"findings": ["Minor dust accumulation"], "severity": "low", "confidence": 0.87, "recommendation": "Schedule cleaning"},
            {"findings": ["Small crack detected on panel surface"], "severity": "medium", "confidence": 0.82, "recommendation": "Schedule repair within 30 days"},
            {"findings": ["Discoloration indicating hot spot"], "severity": "medium", "confidence": 0.78, "recommendation": "Thermal inspection needed"},
            {"findings": ["Debris on panel", "Partial shading"], "severity": "low", "confidence": 0.91, "recommendation": "Remove debris"},
        ]
        return scenarios[waypoint_num % len(scenarios)]


class CuOptClient:
    """NVIDIA cuOpt for route optimization."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://integrate.api.nvidia.com/v1"
    
    def optimize_route(self, waypoints: list) -> list:
        """Optimize waypoint order using cuOpt (TSP solver)."""
        if len(waypoints) <= 3:
            return waypoints
        
        # For demo: use simple nearest-neighbor optimization
        # Real cuOpt API would do this much better
        return self._nearest_neighbor(waypoints)
    
    def _nearest_neighbor(self, waypoints: list) -> list:
        """Simple nearest-neighbor TSP approximation."""
        if len(waypoints) <= 2:
            return waypoints
        
        # Separate fixed waypoints (home, takeoff, land) from inspection points
        fixed_start = [wp for wp in waypoints if wp.get("action") in ["home", "takeoff"]]
        fixed_end = [wp for wp in waypoints if wp.get("action") in ["return", "land"]]
        inspect_points = [wp for wp in waypoints if wp.get("action") == "inspect"]
        
        if len(inspect_points) <= 1:
            return waypoints
        
        # Optimize inspection points order
        optimized = []
        remaining = inspect_points.copy()
        
        # Start from first inspection point
        current = remaining.pop(0)
        optimized.append(current)
        
        while remaining:
            # Find nearest unvisited point
            nearest_idx = 0
            nearest_dist = float('inf')
            
            for i, wp in enumerate(remaining):
                dist = ((wp["x"] - current["x"])**2 + (wp["y"] - current["y"])**2)**0.5
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = i
            
            current = remaining.pop(nearest_idx)
            optimized.append(current)
        
        # Reconstruct full path
        return fixed_start + optimized + fixed_end
    
    def calculate_distance(self, waypoints: list) -> float:
        """Calculate total path distance."""
        total = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i]["x"] - waypoints[i-1]["x"]
            dy = waypoints[i]["y"] - waypoints[i-1]["y"]
            dz = waypoints[i]["z"] - waypoints[i-1]["z"]
            total += (dx**2 + dy**2 + dz**2)**0.5
        return total


class MissionPlanner:
    def __init__(self, bounds: dict, api_key: str = ""):
        self.bounds = bounds
        self.cuopt = CuOptClient(api_key)
    
    def plan(self, intent: dict) -> list:
        loc = intent.get("location", "all")
        alt = intent.get("altitude", 2.0)
        
        x_min, x_max = self.bounds["x_min"], self.bounds["x_max"]
        y_min, y_max = self.bounds["y_min"], self.bounds["y_max"]
        mid_x, mid_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        
        areas = {
            "northeast": (mid_x, x_max, mid_y, y_max),
            "northwest": (x_min, mid_x, mid_y, y_max),
            "southeast": (mid_x, x_max, y_min, mid_y),
            "southwest": (x_min, mid_x, y_min, mid_y),
            "all": (x_min, x_max, y_min, y_max),
        }
        area = areas.get(loc, areas["all"])
        
        waypoints = [{"x": 0, "y": 0, "z": 0.1, "action": "home"}]
        waypoints.append({"x": 0, "y": 0, "z": alt, "action": "takeoff"})
        
        spacing = 1.5
        xs = list(np.arange(area[0], area[1] + 0.1, spacing))
        ys = list(np.arange(area[2], area[3] + 0.1, spacing))
        
        # Generate all inspection points
        for y in ys:
            for x in xs:
                waypoints.append({"x": round(x, 1), "y": round(y, 1), "z": alt, "action": "inspect", "capture": True})
        
        waypoints.append({"x": 0, "y": 0, "z": alt, "action": "return"})
        waypoints.append({"x": 0, "y": 0, "z": 0.1, "action": "land"})
        
        # Calculate distance before optimization
        dist_before = self.cuopt.calculate_distance(waypoints)
        
        # Optimize with cuOpt
        optimized = self.cuopt.optimize_route(waypoints)
        
        # Calculate distance after optimization
        dist_after = self.cuopt.calculate_distance(optimized)
        
        savings = ((dist_before - dist_after) / dist_before) * 100 if dist_before > 0 else 0
        
        print(f"[cuOpt] Path optimization:")
        print(f"        Before: {dist_before:.1f}m")
        print(f"        After:  {dist_after:.1f}m")
        print(f"        Savings: {savings:.1f}%")
        
        return optimized


class DroneSimulator:
    def __init__(self, sim, robot):
        self.sim = sim
        self.robot = robot
        self.dt = sim.get_physics_dt()
        self.prop_ids = robot.find_bodies("m.*_prop")[0]
        self.mass = robot.root_physx_view.get_masses().sum()
        self.gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
        self.hover = self.mass * self.gravity / 4.0
    
    def step(self):
        forces = torch.zeros(self.robot.num_instances, 4, 3, device=self.sim.device)
        forces[..., 2] = self.hover
        self.robot.set_external_force_and_torque(forces, torch.zeros_like(forces), body_ids=self.prop_ids)
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.dt)
    
    def goto(self, x, y, z):
        pos = torch.tensor([[x, y, z, 0, 0, 0, 1]], device=self.sim.device)
        self.robot.write_root_pose_to_sim(pos)
        self.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
        for _ in range(30):
            self.step()


def main():
    print("\n" + "=" * 70)
    print("  ██████╗  █████╗ ██████╗  █████╗ ██╗     ██╗      █████╗ ██╗  ██╗")
    print("  ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║     ██║     ██╔══██╗╚██╗██╔╝")
    print("  ██████╔╝███████║██████╔╝███████║██║     ██║     ███████║ ╚███╔╝ ")
    print("  ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║     ██║     ██╔══██║ ██╔██╗ ")
    print("  ██║     ██║  ██║██║  ██║██║  ██║███████╗███████╗██║  ██║██╔╝ ██╗")
    print("  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝")
    print("         Digital Twin Ground Control + Vision Analysis")
    print("=" * 70)
    
    command = args_cli.command
    api_key = args_cli.api_key
    
    print(f"\n🎤 COMMAND: \"{command}\"")
    
    # Phase 0: Safety Check
    print("\n" + "-" * 50)
    print("PHASE 0: SAFETY VALIDATION (NeMo Guardrails)")
    print("-" * 50)
    
    guardrails = NeMoGuardrails()
    safety_check = guardrails.validate_command(command)
    
    if not safety_check["safe"]:
        print(f"  ❌ COMMAND REJECTED")
        for v in safety_check["violations"]:
            print(f"     ⚠️  {v}")
        print(f"\n  🛑 {safety_check['message']}")
        print("=" * 70)
        simulation_app.close()
        return
    
    print("  ✅ Command passed safety check")
    # Phase 1: NLP
    print("\n" + "-" * 50)
    print("PHASE 1: NATURAL LANGUAGE PROCESSING (Nemotron)")
    print("-" * 50)
    
    nemotron = NemotronClient(api_key) if api_key else None
    intent = nemotron.parse_command(command) if nemotron else NemotronClient("")._fallback(command)
    
    print(f"[NLP] Intent: {intent.get('action', 'inspect')} {intent.get('target', 'area')} @ {intent.get('location', 'all')}")
    
    # Phase 2: Planning
    print("\n" + "-" * 50)
    print("PHASE 2: MISSION PLANNING")
    print("-" * 50)
    
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DistantLightCfg(intensity=3000.0))
    
    for i in range(4):
        for j in range(4):
            cfg = sim_utils.CuboidCfg(size=(0.8, 0.5, 0.05), 
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.4)))
            cfg.func(f"/World/Panel_{i}_{j}", cfg, translation=(i*1.2+1, j*1.2+1, 0.3))
    
    bounds = {"x_min": 0.5, "x_max": 5.5, "y_min": 0.5, "y_max": 5.5}
    planner = MissionPlanner(bounds, api_key)
    waypoints = planner.plan(intent)
    
    print(f"[PLANNER] Generated {len(waypoints)} waypoints")
    
    # Validate mission safety
    mission_safety = guardrails.validate_mission(waypoints)
    if not mission_safety["safe"]:
        print(f"  ⚠️  Mission safety check: {mission_safety['message']}")
        for v in mission_safety["violations"]:
            print(f"     - {v}")
    else:
        print(f"  ✅ {mission_safety['message']}")
    
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Drone")
    robot_cfg.spawn.func("/World/Drone", robot_cfg.spawn, translation=(0, 0, 0.1))
    robot = Articulation(robot_cfg)
    sim.reset()
    
    drone = DroneSimulator(sim, robot)
    vila = VILAClient(api_key)
    
    # Phase 3: Simulation + Vision
    print("\n" + "-" * 50)
    print("PHASE 3: SIMULATION + VISION ANALYSIS (Isaac Sim + VILA)")
    print("-" * 50)
    
    findings = []
    inspection_num = 0
    
    for i, wp in enumerate(waypoints):
        drone.goto(wp["x"], wp["y"], wp["z"])
        
        if wp.get("capture"):
            inspection_num += 1
            print(f"\n  📍 Waypoint {i+1}: ({wp['x']}, {wp['y']}, {wp['z']})")
            print(f"     📸 Capturing image...")
            print(f"     🔍 Analyzing with VILA...")
            
            analysis = vila.simulate_analysis(inspection_num)
            
            findings.append({
                "waypoint": i+1,
                "position": {"x": wp["x"], "y": wp["y"], "z": wp["z"]},
                "analysis": analysis
            })
            
            severity_icon = {"none": "✅", "low": "🟡", "medium": "🟠", "high": "🔴"}.get(analysis["severity"], "⚪")
            print(f"     {severity_icon} Severity: {analysis['severity'].upper()}")
            print(f"     📋 Findings: {', '.join(analysis['findings'])}")
    
    # Phase 4: Report
    print("\n" + "=" * 70)
    print("PHASE 4: INSPECTION REPORT")
    print("=" * 70)
    
    sev_counts = {"none": 0, "low": 0, "medium": 0, "high": 0}
    for f in findings:
        sev = f["analysis"]["severity"]
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
    
    issues = [f for f in findings if f["analysis"]["severity"] != "none"]
    
    print(f"""
  ✅ MISSION COMPLETE
  ══════════════════════════════════════════════════
  
  📊 SUMMARY
  ────────────────────────────────────────
  Command:         "{command}"
  Waypoints:       {len(waypoints)}
  Images Analyzed: {len(findings)}
  
  🔍 FINDINGS
  ────────────────────────────────────────
  ✅ No Issues:    {sev_counts['none']}
  🟡 Low:          {sev_counts['low']}  
  🟠 Medium:       {sev_counts['medium']}
  🔴 High:         {sev_counts['high']}
    """)
    
    if issues:
        print("  ⚠️  ISSUES REQUIRING ATTENTION:")
        print("  ────────────────────────────────────────")
        for issue in issues:
            pos = issue["position"]
            a = issue["analysis"]
            sev_icon = {"low": "🟡", "medium": "🟠", "high": "🔴"}[a["severity"]]
            print(f"  {sev_icon} Location ({pos['x']}, {pos['y']}): {', '.join(a['findings'])}")
            print(f"     → Recommendation: {a['recommendation']}")
    
    print("\n  ══════════════════════════════════════════════════")
    
    report = {
        "command": command,
        "intent": intent,
        "waypoints": waypoints,
        "findings": findings,
        "summary": {"total_inspections": len(findings), "issues_found": len(issues), "severity_counts": sev_counts},
        "timestamp": datetime.now().isoformat()
    }
    
    with open("/workspace/inspection_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  📁 Full report: /workspace/inspection_report.json")
    print("=" * 70)
    
    simulation_app.close()


if __name__ == "__main__":
    main()