"""
Simplified Drone Controller - uses position targets with better stability
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import CRAZYFLIE_CFG


class SimpleDrone:
    def __init__(self, sim, robot):
        self.sim = sim
        self.robot = robot
        self.dt = sim.get_physics_dt()
        self.prop_body_ids = robot.find_bodies("m.*_prop")[0]
        self.mass = robot.root_physx_view.get_masses().sum()
        self.gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
        self.hover_thrust = self.mass * self.gravity / 4.0
        
    def get_pos(self):
        return self.robot.data.root_pos_w[0].cpu().numpy()
    
    def apply_thrust(self, thrust_per_prop):
        """Apply equal thrust to all 4 propellers."""
        forces = torch.zeros(self.robot.num_instances, 4, 3, device=self.sim.device)
        forces[..., 2] = thrust_per_prop
        self.robot.set_external_force_and_torque(forces, torch.zeros_like(forces), body_ids=self.prop_body_ids)
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.dt)
    
    def hover(self, steps=100):
        """Hover in place."""
        for _ in range(steps):
            self.apply_thrust(self.hover_thrust)
    
    def fly_to_altitude(self, target_z, tolerance=0.05):
        """Fly straight up/down to target altitude."""
        print(f"  Flying to altitude {target_z}m...")
        for step in range(1000):
            pos = self.get_pos()
            error = target_z - pos[2]
            
            # Simple proportional control for altitude only
            thrust = self.hover_thrust * (1.0 + 0.5 * error)
            thrust = max(0, min(thrust, self.hover_thrust * 2))  # Clamp
            
            self.apply_thrust(thrust)
            
            if abs(error) < tolerance:
                print(f"  Reached altitude {pos[2]:.2f}m in {step} steps")
                return True
        return False
    
    def set_position(self, x, y, z):
        """Teleport drone to position (for testing waypoints)."""
        new_pos = torch.tensor([[x, y, z, 0, 0, 0, 1]], device=self.sim.device)
        self.robot.write_root_pose_to_sim(new_pos)
        self.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
        # Let physics settle
        for _ in range(50):
            self.apply_thrust(self.hover_thrust)


def main():
    print("=" * 60)
    print("PARALLAX - Simple Drone Test")
    print("=" * 60)
    
    # Setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5, 5, 5], target=[0, 0, 1])
    
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DistantLightCfg(intensity=3000.0))
    
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Drone")
    robot_cfg.spawn.func("/World/Drone", robot_cfg.spawn, translation=(0, 0, 0.5))
    robot = Articulation(robot_cfg)
    sim.reset()
    
    drone = SimpleDrone(sim, robot)
    
    print(f"\nDrone mass: {drone.mass.item():.4f} kg")
    print(f"Starting pos: {drone.get_pos()}")
    
    # Test 1: Altitude control
    print("\n--- Test 1: Takeoff to 2m ---")
    drone.fly_to_altitude(2.0)
    drone.hover(200)
    print(f"Position after hover: {drone.get_pos()}")
    
    # Test 2: Waypoint teleport + hover (simulates inspection pattern)
    print("\n--- Test 2: Inspection waypoints (teleport mode) ---")
    waypoints = [
        (0, 0, 2.0),
        (2, 0, 2.0),
        (2, 2, 2.0),
        (0, 2, 2.0),
        (0, 0, 2.0),
    ]
    
    for i, (x, y, z) in enumerate(waypoints):
        print(f"\nWaypoint {i+1}: ({x}, {y}, {z})")
        drone.set_position(x, y, z)
        drone.hover(100)
        pos = drone.get_pos()
        print(f"  Hovering at: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # Test 3: Land
    print("\n--- Test 3: Landing ---")
    drone.fly_to_altitude(0.3)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    
    simulation_app.close()


if __name__ == "__main__":
    main()