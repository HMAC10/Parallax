"""
PARALLAX - Waypoint Export Module
==================================

Exports waypoints to various drone-compatible formats:
- Litchi CSV (popular mission planning app)
- JSON (custom format)
- DJI SDK (DJI Waypoint Mission format)

Includes mission analysis with distance calculations using haversine formula.
"""

import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class WaypointExporter:
    """
    Exports drone waypoints to various formats.
    
    Supports Litchi, DJI SDK, and custom JSON formats.
    Includes mission analysis and distance calculations.
    """
    
    # Earth radius in meters (mean radius)
    EARTH_RADIUS_M = 6371000
    
    @staticmethod
    def haversine_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Calculate distance between two GPS coordinates using haversine formula.
        
        Args:
            lat1: Latitude of point 1 (degrees)
            lon1: Longitude of point 1 (degrees)
            lat2: Latitude of point 2 (degrees)
            lon2: Longitude of point 2 (degrees)
            
        Returns:
            Distance in meters
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        
        distance = WaypointExporter.EARTH_RADIUS_M * c
        return distance
    
    def to_litchi_csv(
        self,
        waypoints: List[Dict[str, Any]],
        output_path: str,
    ) -> Path:
        """
        Export waypoints to Litchi-compatible CSV format.
        
        Litchi CSV columns:
        latitude, longitude, altitude(m), heading(deg), curvesize(m),
        rotationdir, gimbalmode, gimbalpitchangle, actiontype1, actionparam1,
        altitudemode, speed(m/s), poi_latitude, poi_longitude, poi_altitude(m),
        poi_altitudemode, photo_timeinterval, photo_distinterval
        
        Args:
            waypoints: List of waypoint dictionaries
            output_path: Path to save CSV file
            
        Returns:
            Path to saved file
        """
        logger.info(f"Exporting {len(waypoints)} waypoints to Litchi CSV")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Litchi CSV header
        headers = [
            "latitude", "longitude", "altitude(m)", "heading(deg)", "curvesize(m)",
            "rotationdir", "gimbalmode", "gimbalpitchangle", "actiontype1", "actionparam1",
            "altitudemode", "speed(m/s)", "poi_latitude", "poi_longitude", "poi_altitude(m)",
            "poi_altitudemode", "photo_timeinterval", "photo_distinterval"
        ]
        
        rows = []
        
        for i, wp in enumerate(waypoints):
            action = wp.get("action", "transit")
            altitude = wp.get("altitude_m", 0)
            
            # Determine speed based on action
            if action == "inspect":
                speed = 2.0  # Slow for inspection
                gimbal_pitch = -90  # Look straight down
                action_type = 1  # Take photo
            elif action in ("descend", "ascend"):
                speed = 2.0  # Slow for altitude changes
                gimbal_pitch = -45  # Angled view
                action_type = 0  # No action
            else:  # transit, takeoff, land
                speed = 5.0  # Fast transit
                gimbal_pitch = -45  # Angled view
                action_type = 0
            
            # Calculate heading (direction to next waypoint)
            if i < len(waypoints) - 1:
                next_wp = waypoints[i + 1]
                dlat = next_wp["lat"] - wp["lat"]
                dlon = next_wp["lon"] - wp["lon"]
                heading = math.degrees(math.atan2(dlon, dlat)) % 360
            else:
                heading = 0  # Last waypoint
            
            row = [
                wp["lat"],                    # latitude
                wp["lon"],                    # longitude
                altitude,                     # altitude(m)
                heading,                      # heading(deg)
                0,                            # curvesize(m) - 0 for straight lines
                0,                            # rotationdir - 0 for clockwise
                2,                            # gimbalmode - 2 for interpolate
                gimbal_pitch,                 # gimbalpitchangle
                action_type,                  # actiontype1 - 1 for take photo
                0,                            # actionparam1
                0,                            # altitudemode - 0 for AGL
                speed,                        # speed(m/s)
                0,                            # poi_latitude - 0 for no POI
                0,                            # poi_longitude
                0,                            # poi_altitude(m)
                0,                            # poi_altitudemode
                -1,                           # photo_timeinterval - -1 for disabled
                -1,                           # photo_distinterval - -1 for disabled
            ]
            
            rows.append(row)
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        logger.info(f"Litchi CSV saved to: {output_path}")
        return output_path
    
    def to_json(
        self,
        waypoints: List[Dict[str, Any]],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export waypoints to JSON format with metadata.
        
        Args:
            waypoints: List of waypoint dictionaries
            output_path: Path to save JSON file
            metadata: Optional mission metadata
            
        Returns:
            Path to saved file
        """
        logger.info(f"Exporting {len(waypoints)} waypoints to JSON")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build output structure
        output = {
            "metadata": {
                "format": "PARALLAX Waypoint JSON v1.0",
                "generated_at": datetime.now().isoformat(),
                "num_waypoints": len(waypoints),
            },
            "mission_summary": self.generate_mission_summary(waypoints),
            "waypoints": waypoints,
        }
        
        # Add custom metadata if provided
        if metadata:
            output["metadata"].update(metadata)
        
        # Write JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON saved to: {output_path}")
        return output_path
    
    def to_dji_sdk(
        self,
        waypoints: List[Dict[str, Any]],
        output_path: str,
    ) -> Path:
        """
        Export waypoints in DJI SDK Waypoint Mission format.
        
        DJI format includes mission configuration and waypoint actions.
        
        Args:
            waypoints: List of waypoint dictionaries
            output_path: Path to save JSON file
            
        Returns:
            Path to saved file
        """
        logger.info(f"Exporting {len(waypoints)} waypoints to DJI SDK format")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build DJI waypoint mission structure
        dji_waypoints = []
        
        for i, wp in enumerate(waypoints):
            action = wp.get("action", "transit")
            
            # Determine flight speed
            if action == "inspect":
                speed = 2.0
            elif action in ("descend", "ascend"):
                speed = 2.0
            else:
                speed = 5.0
            
            # Build waypoint actions
            actions = []
            if action == "inspect":
                actions.append({
                    "type": "TAKE_PHOTO",
                    "param": 0,
                })
                actions.append({
                    "type": "ROTATE_GIMBAL",
                    "param": -90,  # Pitch angle
                })
            
            dji_wp = {
                "waypointIndex": i,
                "coordinate": {
                    "latitude": wp["lat"],
                    "longitude": wp["lon"],
                    "altitude": wp.get("altitude_m", 0),
                },
                "speed": speed,
                "heading": 0,  # Auto heading
                "turnMode": "AUTO",
                "actions": actions,
                "metadata": {
                    "action": action,
                    "asset_id": wp.get("asset_id"),
                    "notes": wp.get("notes", ""),
                }
            }
            
            dji_waypoints.append(dji_wp)
        
        # Mission configuration
        dji_mission = {
            "version": "1.0",
            "type": "WAYPOINT_MISSION",
            "mission": {
                "missionConfig": {
                    "autoFlightSpeed": 5.0,
                    "maxFlightSpeed": 8.0,
                    "finishedAction": "GO_HOME",
                    "headingMode": "AUTO",
                    "flightPathMode": "NORMAL",
                },
                "waypoints": dji_waypoints,
            }
        }
        
        # Write JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dji_mission, f, indent=2, ensure_ascii=False)
        
        logger.info(f"DJI SDK format saved to: {output_path}")
        return output_path
    
    def generate_mission_summary(
        self,
        waypoints: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate mission summary with distance and duration estimates.
        
        Args:
            waypoints: List of waypoint dictionaries
            
        Returns:
            Dictionary with mission statistics
        """
        if not waypoints:
            return {
                "total_distance_m": 0,
                "estimated_duration_s": 0,
                "num_waypoints": 0,
                "num_inspections": 0,
                "max_altitude_m": 0,
            }
        
        total_distance = 0.0
        total_duration = 0.0
        num_inspections = 0
        max_altitude = 0.0
        
        # Calculate distance and duration
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
            # Horizontal distance
            h_distance = self.haversine_distance(
                wp1["lat"], wp1["lon"],
                wp2["lat"], wp2["lon"]
            )
            
            # Vertical distance
            alt1 = wp1.get("altitude_m", 0)
            alt2 = wp2.get("altitude_m", 0)
            v_distance = abs(alt2 - alt1)
            
            # Total 3D distance
            distance = math.sqrt(h_distance ** 2 + v_distance ** 2)
            total_distance += distance
            
            # Update max altitude
            max_altitude = max(max_altitude, alt1, alt2)
            
            # Estimate duration based on action
            action = wp2.get("action", "transit")
            
            if action == "inspect":
                # Slow speed + hover time
                speed = 2.0  # m/s
                hover_time = 5.0  # seconds for capture
                duration = (distance / speed) + hover_time
                num_inspections += 1
            elif action in ("descend", "ascend"):
                # Vertical movement speed
                speed = 2.0  # m/s
                duration = distance / speed
            elif action == "land":
                # Landing time
                duration = 10.0  # seconds
            elif action == "takeoff":
                # Takeoff time
                duration = 10.0  # seconds
            else:  # transit
                # Fast transit speed
                speed = 5.0  # m/s
                duration = distance / speed
            
            total_duration += duration
        
        return {
            "total_distance_m": round(total_distance, 2),
            "total_distance_km": round(total_distance / 1000, 3),
            "estimated_duration_s": round(total_duration, 1),
            "estimated_duration_min": round(total_duration / 60, 2),
            "num_waypoints": len(waypoints),
            "num_inspections": num_inspections,
            "max_altitude_m": round(max_altitude, 2),
        }
    
    def export_all_formats(
        self,
        waypoints: List[Dict[str, Any]],
        output_dir: str,
        mission_name: str = "mission",
    ) -> Dict[str, Path]:
        """
        Export waypoints to all supported formats.
        
        Args:
            waypoints: List of waypoint dictionaries
            output_dir: Directory to save files
            mission_name: Base name for output files
            
        Returns:
            Dictionary mapping format names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        paths = {}
        
        # Litchi CSV
        litchi_path = output_dir / f"{mission_name}_litchi_{timestamp}.csv"
        paths["litchi"] = self.to_litchi_csv(waypoints, litchi_path)
        
        # JSON
        json_path = output_dir / f"{mission_name}_waypoints_{timestamp}.json"
        paths["json"] = self.to_json(waypoints, json_path)
        
        # DJI SDK
        dji_path = output_dir / f"{mission_name}_dji_{timestamp}.json"
        paths["dji_sdk"] = self.to_dji_sdk(waypoints, dji_path)
        
        logger.info(f"Exported mission to {len(paths)} formats")
        return paths


# Standalone test
if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("PARALLAX - Waypoint Export Test")
    print("="*70 + "\n")
    
    try:
        # Import site_config
        from site_config import SiteConfig
        
        # Load demo site
        print("Loading demo_site.json...")
        site = SiteConfig()
        if not site.load("demo_site.json"):
            print("ERROR: Failed to load demo_site.json")
            sys.exit(1)
        
        print(f"[OK] Loaded site: {site.site_name}")
        print(f"     Anchor: {site.anchor}")
        print(f"     Assets: {len(site.assets)}\n")
        
        # Generate waypoints for all 4 poles
        print("Generating waypoints for all 4 utility poles...")
        asset_ids = ["pole_1", "pole_2", "pole_3", "pole_4"]
        waypoints = site.generate_waypoints(asset_ids)
        
        print(f"[OK] Generated {len(waypoints)} waypoints\n")
        
        # Export to all formats
        print("Exporting to all formats...")
        exporter = WaypointExporter()
        
        output_dir = "output/waypoints"
        paths = exporter.export_all_formats(waypoints, output_dir, "demo_mission")
        
        print("\n[OK] Exported to:")
        for format_name, path in paths.items():
            print(f"  - {format_name}: {path}")
        
        # Generate and print summary
        print("\n" + "="*70)
        print("Mission Summary")
        print("="*70)
        
        summary = exporter.generate_mission_summary(waypoints)
        
        print(f"Total Distance:      {summary['total_distance_m']:.1f}m ({summary['total_distance_km']:.2f}km)")
        print(f"Estimated Duration:  {summary['estimated_duration_s']:.1f}s ({summary['estimated_duration_min']:.2f}min)")
        print(f"Waypoints:           {summary['num_waypoints']}")
        print(f"Inspections:         {summary['num_inspections']}")
        print(f"Max Altitude:        {summary['max_altitude_m']:.1f}m")
        
        print("\n" + "="*70)
        print("[SUCCESS] Test completed successfully!")
        print("="*70 + "\n")
    
    except ImportError as e:
        print(f"ERROR: Import failed - {e}")
        print("Make sure site_config.py is in the same directory")
        sys.exit(1)
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
