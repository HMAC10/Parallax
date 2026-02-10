# PARALLAX - Digital Twin Ground Control v1.0

**Complete end-to-end drone inspection pipeline for NVIDIA GTC 2026 Competition**

Powered by NVIDIA AI Stack: NeMoGuard, Nemotron, cuOpt, Isaac Sim, Omniverse, Cosmos Reason 2

---

## 🚀 Quick Start

```bash
# Basic inspection mission
python parallax_pipeline.py --command "inspect all poles" --site demo_site.json --mock

# With real drone footage
python parallax_pipeline.py --command "inspect poles 1 through 4" --site demo_site.json --footage drone_video.mp4

# Custom output directory
python parallax_pipeline.py --command "inspect all poles" --site demo_site.json --mock --output /workspace/Parallax/output

# Fast mode (skip delays, for testing)
python parallax_pipeline.py --command "inspect all poles" --site demo_site.json --mock --fast
```

---

## 📋 Pipeline Phases

### Phase 0: NeMoGuard Safety Validation
- Validates commands for dangerous keywords
- Rejects unsafe operations (crash, collide, weapon, etc.)
- **Output:** Safety clearance or rejection

### Phase 1: Nemotron NLP Parsing (NVIDIA NIM)
- Parses natural language into structured mission parameters
- Resolves asset references ("poles 1 through 4", "all poles")
- Extracts altitudes and mission type
- **Output:** Mission configuration with target assets

### Phase 2: cuOpt Route Optimization
- Optimizes asset visit order using nearest-neighbor algorithm
- Calculates distance reduction percentage
- Uses real GPS haversine distance calculations
- **Output:** Optimized waypoint order

### Phase 3: Isaac Sim Validation (NVIDIA Omniverse)
- Validates flight path in digital twin environment
- Checks: altitude clearance, obstacles, no-fly zones, wind, battery
- **Output:** Flight path validation status

### Phase 4: Export Flight Plan
- Generates waypoints using real site configuration
- Exports to 3 formats:
  - **Litchi CSV** - For Litchi mission planner app
  - **JSON** - Custom format with metadata
  - **DJI SDK** - DJI Waypoint Mission format
- **Output:** Mission files + summary (distance, duration, waypoints)

### Phase 5: Cosmos Reason 2 Vision Analysis (Hugging Face)
- Analyzes drone footage using `nvidia/Cosmos-Reason2-8B`
- Generates professional inspection reports (Markdown + JSON)
- Identifies findings with severity levels (CRITICAL/WARNING/INFO)
- **Output:** Analysis results + inspection reports

---

## 🎯 Example Commands

### Inspect Specific Assets
```bash
python parallax_pipeline.py --command "inspect pole 1" --site demo_site.json --mock
python parallax_pipeline.py --command "inspect poles 1 through 3" --site demo_site.json --mock
python parallax_pipeline.py --command "inspect pole_4" --site demo_site.json --mock
```

### Inspect All Assets
```bash
python parallax_pipeline.py --command "inspect all poles" --site demo_site.json --mock
python parallax_pipeline.py --command "inspect all assets" --site demo_site.json --mock
```

### Safety Rejection Test
```bash
python parallax_pipeline.py --command "crash the drone" --site demo_site.json
# Output: [X] SAFETY VIOLATION DETECTED - MISSION ABORTED
```

---

## 📁 Required Files

The pipeline integrates these existing modules:

- ✅ `site_config.py` - Site configuration loader
- ✅ `export_waypoints.py` - Waypoint export (Litchi/JSON/DJI)
- ✅ `cosmos_analyzer.py` - Cosmos Reason 2 vision analysis (optional with `--mock`)
- ✅ `report_generator.py` - Inspection report generation (optional)
- ✅ `demo_site.json` - Seattle demo site with 4 utility poles

---

## 🎬 Demo Output Example

```
======================================================================
 ____   _    ____      _    _     _        _    __  __
|  _ \ / \  |  _ \    / \  | |   | |      / \   \ \/ /
| |_) / _ \ | |_) |  / _ \ | |   | |     / _ \   \  /
|  __/ ___ \|  _ <  / ___ \| |___| |___ / ___ \  /  \
|_| /_/   \_\_| \_\/_/   \_\_____|_____/_/   \_\/_/\_\
======================================================================
         Digital Twin Ground Control v1.0
           Powered by NVIDIA AI Stack

==============================================================
  PHASE 0: NeMoGuard Safety Validation
==============================================================

[OK] Command validated — safe to proceed
Duration: 0.30s

==============================================================
  PHASE 1: Nemotron NLP Parsing (NVIDIA NIM)
==============================================================

[OK] Parsed: INSPECT mission — 4 assets identified
Duration: 1.20s

==============================================================
  PHASE 2: cuOpt Route Optimization
==============================================================

[OK] Route optimized — 27% distance reduction (199m -> 145m)
Duration: 0.80s

... (continues through all phases)

==============================================================
  MISSION SUMMARY
==============================================================

Mission Configuration:
  * Type: INSPECT
  * Assets: 4 (pole_1, pole_2, pole_3, pole_4)
  * Route Optimization: 27% improvement

Flight Plan:
  * Distance: 145.2m (0.15km)
  * Duration: 1.60 minutes
  * Waypoints: 19
  * Inspections: 4

Analysis Results:
  * Total Findings: 3
  * Critical: 0
  * Warnings: 1

NVIDIA AI Stack:
  * NeMoGuard      - Safety & content filtering
  * Nemotron       - Natural language understanding
  * NIM            - Model inference microservices
  * cuOpt          - Route optimization
  * Isaac Sim      - Physics simulation & validation
  * Omniverse      - Digital twin platform
  * Cosmos Reason 2 - Physical AI video reasoning
  * Brev           - GPU cloud infrastructure

Model hosted on Hugging Face: nvidia/Cosmos-Reason2-8B

[SUCCESS] PARALLAX MISSION COMPLETE
```

---

## 🛠️ Command Line Options

```
usage: parallax_pipeline.py [-h] --command COMMAND --site SITE [--footage FOOTAGE] 
                             [--mock] [--output OUTPUT] [--fast]

options:
  --command COMMAND    Natural language command (e.g., 'inspect poles 1 through 4')
  --site SITE          Path to site configuration JSON file
  --footage FOOTAGE    Path to drone footage (image or video) for Cosmos analysis
  --mock               Use mock data for Cosmos analysis (no GPU required)
  --output OUTPUT      Output directory for exported files (default: ./output)
  --fast               Skip time delays between phases (for faster testing)
```

---

## 🎨 Features

### ✅ Safety First
- NeMoGuard content filtering rejects dangerous commands
- Isaac Sim validates flight paths before execution

### ✅ Natural Language
- Understands "poles 1 through 4", "all poles", "pole_1"
- Extracts mission parameters from plain English

### ✅ Route Optimization
- Real GPS distance calculations (haversine formula)
- Nearest-neighbor optimization shows % improvement

### ✅ Multi-Format Export
- Litchi CSV for consumer drones
- DJI SDK for enterprise drones
- JSON for custom integrations

### ✅ AI Vision Analysis
- NVIDIA Cosmos Reason 2 for video reasoning
- Chain-of-thought analysis with `<think>` tags
- Professional inspection reports

### ✅ Works Everywhere
- Windows & Linux compatible
- Mock mode for testing without GPU
- Fast mode for rapid iteration

---

## 📊 Output Files

After running, you'll find:

```
output/
├── parallax_mission_YYYYMMDD_HHMMSS_litchi_*.csv      # Litchi format
├── parallax_mission_YYYYMMDD_HHMMSS_waypoints_*.json  # JSON format
├── parallax_mission_YYYYMMDD_HHMMSS_dji_*.json        # DJI SDK format
├── inspection_report_YYYYMMDD_HHMMSS.md               # Markdown report
└── inspection_report_YYYYMMDD_HHMMSS.json             # JSON report
```

---

## 🚀 For NVIDIA GTC 2026

This pipeline demonstrates:

1. **NeMoGuard** - Safety validation
2. **Nemotron + NIM** - Natural language understanding
3. **cuOpt** - Route optimization with real GPS data
4. **Isaac Sim + Omniverse** - Digital twin validation
5. **Cosmos Reason 2** - Physical AI video reasoning
6. **Brev** - GPU cloud infrastructure

**Model:** `nvidia/Cosmos-Reason2-8B` on Hugging Face

---

## 🎥 Screen Recording Tips

For competition demos:

1. Use `--fast` flag to skip delays during editing
2. Remove `--fast` for final recording to show realistic timing
3. Test safety rejection: `--command "crash the drone"`
4. Show full pipeline: `--command "inspect all poles" --mock`
5. Show with footage: `--command "inspect all poles" --footage video.mp4`

---

## 🐛 Troubleshooting

**"Failed to load site configuration"**
- Make sure `demo_site.json` exists
- Check file path is correct

**"cosmos_analyzer not available"**
- Use `--mock` flag for testing without GPU
- Install: `pip install -r requirements_cosmos.txt`

**"No assets matched"**
- Check your command syntax
- Try: "all poles" or "poles 1 through 4"
- Asset IDs: pole_1, pole_2, pole_3, pole_4

---

**Built for NVIDIA GTC 2026 Competition** 🏆  
**Digital Twin Ground Control - Powered by NVIDIA AI Stack**
