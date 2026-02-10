"""
PARALLAX - Site Configuration Module
=====================================

Loads and manages pre-mapped inspection site configurations.
Handles asset databases, GPS coordinates, and waypoint generation.

Part of the PARALLAX pipeline:
NeMoGuard → Nemotron → cuOpt → Isaac Sim → Cosmos Reason 2 → Report
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class SiteConfig:
    """
    Manages inspection site configuration and asset database.
    
    Loads site definitions with GPS coordinates, asset metadata,
    and generates inspection waypoints for drone missions.
    """
    
    def __init__(self):
        """Initialize empty site configuration."""
        self.site_name: str = ""
        self.anchor: Tuple[float, float] = (0.0, 0.0)
        self.default_transit_altitude_m: float = 25.0
        self.default_inspection_altitude_m: float = 12.0
        self.assets: List[Dict[str, Any]] = []
        self._config_path: Optional[Path] = None
    
    def load(self, filepath: str) -> bool:
        """
        Load and validate site configuration from JSON file.
        
        Args:
            filepath: Path to site config JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Site config file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate required fields
            if "site_name" not in config:
                logger.error("Missing required field: site_name")
                return False
            
            if "anchor" not in config or len(config["anchor"]) != 2:
                logger.error("Missing or invalid anchor point")
                return False
            
            if "assets" not in config or not isinstance(config["assets"], list):
                logger.error("Missing or invalid assets list")
                return False
            
            # Load data
            self.site_name = config["site_name"]
            self.anchor = tuple(config["anchor"])
            self.default_transit_altitude_m = config.get("default_transit_altitude_m", 25.0)
            self.default_inspection_altitude_m = config.get("default_inspection_altitude_m", 12.0)
            self.assets = config["assets"]
            self._config_path = filepath
            
            # Validate assets
            for i, asset in enumerate(self.assets):
                required_fields = ["id", "type", "lat", "lon"]
                for field in required_fields:
                    if field not in asset:
                        logger.error(f"Asset {i} missing required field: {field}")
                        return False
            
            logger.info(f"Loaded site: {self.site_name}")
            logger.info(f"Anchor: {self.anchor}")
            logger.info(f"Assets: {len(self.assets)}")
            
            return True
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading site config: {e}")
            return False
    
    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single asset by ID.
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Asset dictionary or None if not found
        """
        for asset in self.assets:
            if asset["id"] == asset_id:
                return asset
        
        logger.warning(f"Asset not found: {asset_id}")
        return None
    
    def get_assets_by_type(self, asset_type: str) -> List[Dict[str, Any]]:
        """
        Get all assets of a specific type.
        
        Args:
            asset_type: Type of asset (e.g., "utility_pole", "traffic_light")
            
        Returns:
            List of matching assets
        """
        matches = [asset for asset in self.assets if asset["type"] == asset_type]
        logger.info(f"Found {len(matches)} assets of type '{asset_type}'")
        return matches
    
    def resolve_natural_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse natural language asset references.
        
        Handles patterns like:
        - "pole 1" or "pole_1" → single asset
        - "poles 1 through 3" or "poles 1-3" → range
        - "all poles" → all assets of type
        - "all assets" → all assets
        
        Args:
            text: Natural language description
            
        Returns:
            List of matching assets
        """
        text = text.lower().strip()
        logger.info(f"Resolving natural language: '{text}'")
        
        # Pattern: "all assets"
        if text in ("all assets", "all", "everything"):
            logger.info(f"Resolved to all {len(self.assets)} assets")
            return self.assets
        
        # Pattern: "all [type]" or "all [type]s"
        all_type_match = re.match(r'all\s+(\w+?)s?$', text)
        if all_type_match:
            asset_type = all_type_match.group(1)
            # Try with underscore for multi-word types
            matches = self.get_assets_by_type(asset_type)
            if not matches:
                matches = self.get_assets_by_type(asset_type.replace(' ', '_'))
            return matches
        
        # Pattern: "[type] N" or "[type]_N" (single asset)
        single_match = re.match(r'(\w+?)s?\s*[_\s]?\s*(\d+)$', text)
        if single_match:
            asset_type = single_match.group(1)
            asset_num = single_match.group(2)
            
            # Try both formats: type_N and typeN
            asset_id = f"{asset_type}_{asset_num}"
            asset = self.get_asset(asset_id)
            
            if asset:
                logger.info(f"Resolved to single asset: {asset_id}")
                return [asset]
            
            # Try without underscore
            asset_id = f"{asset_type}{asset_num}"
            asset = self.get_asset(asset_id)
            if asset:
                logger.info(f"Resolved to single asset: {asset_id}")
                return [asset]
        
        # Pattern: "[type]s N through M" or "[type]s N-M" (range)
        range_match = re.match(r'(\w+?)s?\s*(\d+)\s*(?:through|to|-)\s*(\d+)$', text)
        if range_match:
            asset_type = range_match.group(1)
            start_num = int(range_match.group(2))
            end_num = int(range_match.group(3))
            
            matches = []
            for num in range(start_num, end_num + 1):
                # Try with underscore first
                asset_id = f"{asset_type}_{num}"
                asset = self.get_asset(asset_id)
                
                if not asset:
                    # Try without underscore
                    asset_id = f"{asset_type}{num}"
                    asset = self.get_asset(asset_id)
                
                if asset:
                    matches.append(asset)
            
            logger.info(f"Resolved range to {len(matches)} assets")
            return matches
        
        # Pattern: exact asset ID
        asset = self.get_asset(text)
        if asset:
            logger.info(f"Resolved to exact asset ID: {text}")
            return [asset]
        
        logger.warning(f"Could not resolve: '{text}'")
        return []
    
    def get_anchor(self) -> Tuple[float, float]:
        """
        Get the GPS anchor point for the site.
        
        Returns:
            Tuple of (latitude, longitude)
        """
        return self.anchor
    
    def generate_waypoints(
        self,
        asset_ids: List[str],
        transit_alt: Optional[float] = None,
        inspection_alt: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate complete mission waypoints for inspecting assets.
        
        Mission sequence:
        1. Takeoff at anchor
        2. For each asset:
           - Transit to asset at transit altitude
           - Descend to inspection altitude
           - Inspect (capture imagery)
           - Ascend to transit altitude
        3. Return to anchor
        4. Land
        
        Args:
            asset_ids: List of asset IDs to inspect
            transit_alt: Transit altitude in meters (default: site default)
            inspection_alt: Inspection altitude in meters (default: site default)
            
        Returns:
            List of waypoint dictionaries with lat, lon, altitude_m, action, asset_id
        """
        transit_alt = transit_alt or self.default_transit_altitude_m
        inspection_alt = inspection_alt or self.default_inspection_altitude_m
        
        logger.info(f"Generating waypoints for {len(asset_ids)} assets")
        logger.info(f"Transit altitude: {transit_alt}m, Inspection altitude: {inspection_alt}m")
        
        waypoints = []
        anchor_lat, anchor_lon = self.anchor
        
        # 1. Takeoff
        waypoints.append({
            "lat": anchor_lat,
            "lon": anchor_lon,
            "altitude_m": transit_alt,
            "action": "takeoff",
            "asset_id": None,
            "notes": "Takeoff from anchor point",
        })
        
        # 2. Inspect each asset
        for i, asset_id in enumerate(asset_ids):
            asset = self.get_asset(asset_id)
            
            if not asset:
                logger.warning(f"Skipping unknown asset: {asset_id}")
                continue
            
            asset_lat = asset["lat"]
            asset_lon = asset["lon"]
            
            # Transit to asset
            waypoints.append({
                "lat": asset_lat,
                "lon": asset_lon,
                "altitude_m": transit_alt,
                "action": "transit",
                "asset_id": asset_id,
                "notes": f"Transit to {asset_id}",
            })
            
            # Descend to inspection altitude
            waypoints.append({
                "lat": asset_lat,
                "lon": asset_lon,
                "altitude_m": inspection_alt,
                "action": "descend",
                "asset_id": asset_id,
                "notes": f"Descend for inspection of {asset_id}",
            })
            
            # Inspect (hover and capture)
            waypoints.append({
                "lat": asset_lat,
                "lon": asset_lon,
                "altitude_m": inspection_alt,
                "action": "inspect",
                "asset_id": asset_id,
                "notes": f"Inspect {asset_id} - {asset.get('type', 'unknown')}",
            })
            
            # Ascend to transit altitude
            waypoints.append({
                "lat": asset_lat,
                "lon": asset_lon,
                "altitude_m": transit_alt,
                "action": "ascend",
                "asset_id": asset_id,
                "notes": f"Ascend after inspecting {asset_id}",
            })
        
        # 3. Return to anchor
        waypoints.append({
            "lat": anchor_lat,
            "lon": anchor_lon,
            "altitude_m": transit_alt,
            "action": "transit",
            "asset_id": None,
            "notes": "Return to anchor point",
        })
        
        # 4. Land
        waypoints.append({
            "lat": anchor_lat,
            "lon": anchor_lon,
            "altitude_m": 0,
            "action": "land",
            "asset_id": None,
            "notes": "Land at anchor point",
        })
        
        logger.info(f"Generated {len(waypoints)} waypoints")
        return waypoints
    
    def get_site_info(self) -> Dict[str, Any]:
        """
        Get summary information about the site.
        
        Returns:
            Dictionary with site metadata
        """
        asset_types = {}
        for asset in self.assets:
            asset_type = asset.get("type", "unknown")
            asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
        
        return {
            "site_name": self.site_name,
            "anchor": self.anchor,
            "total_assets": len(self.assets),
            "asset_types": asset_types,
            "default_transit_altitude_m": self.default_transit_altitude_m,
            "default_inspection_altitude_m": self.default_inspection_altitude_m,
            "config_path": str(self._config_path) if self._config_path else None,
        }
    
    def __repr__(self) -> str:
        return f"SiteConfig(site='{self.site_name}', assets={len(self.assets)})"


# Convenience function
def load_site(filepath: str) -> Optional[SiteConfig]:
    """
    Quick utility to load a site configuration.
    
    Args:
        filepath: Path to site config JSON
        
    Returns:
        SiteConfig instance or None if load failed
    """
    config = SiteConfig()
    if config.load(filepath):
        return config
    return None
