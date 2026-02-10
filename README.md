# PARALLAX — Digital Twin Ground Control

### Natural Language Drone Inspection Powered by NVIDIA AI

**Speak a command. Plan a mission. Inspect infrastructure. Powered by 8 NVIDIA AI technologies.**

---

## 🎬 Demo Video

**[Demo Video Coming Soon]**

Watch PARALLAX plan and execute a real-world utility pole inspection in Seattle, WA.

---

## 🚨 The Problem

I co-founded Voltair, an AI drone infrastructure startup, and hit a wall that every drone operator faces: **the intelligence gap**. 

The hardware is incredible — modern drones can fly autonomously for 40+ minutes with centimeter-level GPS accuracy. But commanding them? Still stuck in 2015. Operators spend hours manually programming waypoints, need FAA-certified pilots for every flight, and review footage frame-by-frame to find a single corroded bolt on a utility pole.

**This doesn't scale.** The US alone has 180 million utility poles, 600,000 bridges, and 3.5 million miles of power lines that need regular inspection. The bottleneck isn't the drone — it's the brain.

**PARALLAX is the brain I wished we had at Voltair.**

---

## ⚡ How It Works

```
  "Inspect poles 1 through 4"
              ↓
       [NeMoGuard]
    Safety validation
              ↓
       [Nemotron]
   NLP command parsing
              ↓
         [cuOpt]
   Route optimization
              ↓
      [Isaac Sim]
Flight path simulation
     & validation
              ↓
        [Export]
GPS waypoints → Litchi CSV
      DJI SDK / JSON
              ↓
   [Cosmos Reason 2]
 Drone footage analysis
              ↓
        [Report]
Professional inspection
        report
```

---

## 🔧 NVIDIA AI Stack

| Tool | Purpose | Status |
|------|---------|--------|
| **NeMoGuard** | Safety & content filtering | ✅ Real API |
| **Nemotron** (`nemotron-mini-4b-instruct`) | Natural language parsing | ✅ Real API |
| **NIM** | Model inference microservices | ✅ Real |
| **cuOpt** | Route optimization | ⚠️ Simulated (nearest-neighbor + haversine) |
| **Isaac Sim** | Physics simulation & path validation | ✅ Real on Brev |
| **Omniverse** | Digital twin platform | ✅ Via Isaac Sim |
| **Cosmos Reason 2** (8B) | Physical AI video reasoning | ✅ Real on Brev |
| **Brev** | GPU cloud infrastructure (L40S 48GB) | ✅ Running |

**Model weights hosted on Hugging Face:** `nvidia/Cosmos-Reason2-8B`

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/HMAC10/Parallax.git
cd Parallax

# Install dependencies
pip install -r requirements_cosmos.txt
pip install accelerate

# Run with mock data (no GPU needed)
python parallax_pipeline.py --command "inspect all poles" --site demo_site.json --mock

# Run with real Cosmos Reason 2 (requires 32GB+ VRAM)
python parallax_pipeline.py --command "inspect all poles" --site demo_site.json --footage drone_video.mp4

# Test safety rejection
python parallax_pipeline.py --command "crash the drone" --site demo_site.json
```

---

## 📁 Project Structure

- **`parallax_pipeline.py`** — Full end-to-end pipeline (hero script)
- **`cosmos_analyzer.py`** — Cosmos Reason 2 integration for video/image analysis
- **`report_generator.py`** — Professional inspection report generator (Markdown + JSON)
- **`site_config.py`** — Site configuration loader with GPS asset mapping
- **`export_waypoints.py`** — Waypoint exporter (Litchi CSV, JSON, DJI SDK)
- **`demo_site.json`** — Real inspection site: 4 utility poles in Seattle, WA
- **`parallax_web_demo_v7.html`** — Interactive 3D web visualization (Three.js)
- **`test_cosmos.py`** — Cosmos integration test script

---

## 🌍 Real-World Demo

PARALLAX was tested on real infrastructure in Seattle, WA. Four utility poles were pre-mapped with GPS coordinates, and the pipeline generated a validated 19-waypoint flight plan covering 199m. A DJI Mini 4K captured 4K inspection footage which was analyzed by Cosmos Reason 2 running on an NVIDIA L40S GPU via Brev. The AI identified vegetation encroachment and generated a professional inspection report with severity ratings and recommended actions.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    USER COMMAND                           │
│          "Inspect poles 1 through 4"                      │
└──────────────────┬───────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────────────┐
│  PHASE 0: NeMoGuard        │  Safety validation          │
├──────────────────────────────────────────────────────────┤
│  PHASE 1: Nemotron (NIM)   │  NLP parsing → assets       │
├──────────────────────────────────────────────────────────┤
│  PHASE 2: cuOpt            │  Route optimization          │
├──────────────────────────────────────────────────────────┤
│  PHASE 3: Isaac Sim        │  Flight path validation      │
├──────────────────────────────────────────────────────────┤
│  PHASE 4: Export           │  Litchi CSV / DJI SDK / JSON │
├──────────────────────────────────────────────────────────┤
│  PHASE 5: Cosmos Reason 2  │  Drone footage analysis      │
├──────────────────────────────────────────────────────────┤
│  OUTPUT: Inspection Report │  Markdown + JSON             │
└──────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Natural language command → NeMoGuard safety filter
2. Nemotron parses intent → extracts target assets
3. cuOpt optimizes visit order → minimizes flight distance
4. Isaac Sim validates trajectory → checks collisions
5. Export generates GPS waypoints → Litchi/DJI formats
6. Drone captures footage → Cosmos Reason 2 analyzes
7. Report generator produces deliverable → Markdown + JSON

---

## 🔮 Vision

Today, drone inspection requires a trained pilot, a mission planner, and an analyst. PARALLAX collapses all three roles into a single natural language command. 

The long-term vision: **any utility worker, construction manager, or insurance adjuster can inspect infrastructure by simply telling the drone what to look at.** One operator managing a fleet of autonomous inspection drones — that's the future PARALLAX is building toward.

---

## 👨‍💻 About

Built by **Hunter McKay** — UW Foster School of Business senior (Finance + Information Systems, 3.98 GPA). Co-founded Voltair, an AI drone infrastructure startup that won the **$25K Dempsey Startup Competition** (1st of 174 teams) and **$15K Environmental Innovation Challenge**. Former Summer Analyst at J.P. Morgan CIB (Technology & Disruptive Commerce). 

PARALLAX was born from a real problem encountered building Voltair — the drone industry has incredible hardware but lacks the AI brain to make autonomous inspection accessible to everyone, not just trained pilots.

**GitHub:** https://github.com/HMAC10/Parallax

---

## 📄 License

MIT License

---

**Built for NVIDIA GTC 2026 Golden Ticket Contest & NVIDIA Cosmos Cookoff Hackathon**
