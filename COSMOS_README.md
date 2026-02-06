# PARALLAX - Cosmos Reason 2 Integration

**NVIDIA GTC 2026 Competition Entry**

This module integrates NVIDIA's Cosmos-Reason2-8B vision-language model for automated drone infrastructure inspection and report generation.

---

## 🚀 Quick Start

### Prerequisites

- **GPU:** 32GB+ VRAM (tested on L40S 48GB)
- **Python:** 3.9+
- **CUDA:** 11.8+ or 12.1+

### Installation

```bash
# Install Cosmos-specific dependencies
pip install -r requirements_cosmos.txt

# Verify transformers version
python -c "import transformers; print(transformers.__version__)"
# Should be >= 4.57.0
```

### Run Tests

**Option 1: Mock Mode (No GPU)**
```bash
python test_cosmos.py --mock
```

**Option 2: With Test Image (Requires GPU)**
```bash
python test_cosmos.py
```

**Option 3: Analyze Real Drone Footage**
```bash
# Image
python test_cosmos.py --image /path/to/drone_photo.jpg

# Video
python test_cosmos.py --video /path/to/drone_video.mp4
```

---

## 📁 File Structure

```
drone-ground-control/
├── cosmos_analyzer.py          # Cosmos Reason 2 model wrapper
├── report_generator.py         # Professional report generation
├── test_cosmos.py              # Test script
├── requirements_cosmos.txt     # Dependencies
└── COSMOS_README.md           # This file
```

---

## 🔧 Usage Examples

### Example 1: Analyze a Drone Image

```python
from cosmos_analyzer import CosmosAnalyzer

# Initialize model
analyzer = CosmosAnalyzer()

# Analyze infrastructure footage
result = analyzer.analyze_drone_footage(
    footage_path="solar_panel_inspection.jpg",
    is_video=False
)

print("AI Thinking:", result["thinking"])
print("Assessment:", result["answer"])
print("Findings:", result["findings"])
```

### Example 2: Generate Inspection Report

```python
from report_generator import InspectionReportGenerator
from datetime import datetime

# Create generator
generator = InspectionReportGenerator()

# Mission metadata
metadata = {
    "location": "Solar Farm Site A",
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "drone_model": "DJI Mavic 3 Enterprise",
    "flight_duration": "15 minutes",
    "waypoints": 12,
    "operator": "John Doe",
}

# Generate reports
md_report = generator.generate_report(cosmos_output, metadata)
json_report = generator.generate_json_report(cosmos_output, metadata)

# Save
generator.save_report(md_report, "output/report.md")
generator.save_json_report(json_report, "output/report.json")
```

### Example 3: Full Pipeline

```python
from cosmos_analyzer import CosmosAnalyzer
from report_generator import generate_quick_report

# Analyze
analyzer = CosmosAnalyzer()
result = analyzer.analyze_drone_footage("inspection_video.mp4", is_video=True)

# Generate and save reports
md_path, json_path = generate_quick_report(
    result,
    location="Powerline Corridor B",
    output_dir="./reports"
)

print(f"Reports saved: {md_path}, {json_path}")
```

---

## 🎯 Key Features

### CosmosAnalyzer

- **Chain-of-Thought Reasoning:** Extracts `<think>` and `<answer>` tags
- **Video Processing:** Samples at 4 FPS for efficient analysis
- **Infrastructure Focus:** Specialized prompts for equipment inspection
- **Structured Output:** Parses findings with severity levels (CRITICAL/WARNING/INFO)

### InspectionReportGenerator

- **Professional Format:** Markdown reports for engineering teams
- **Severity Classification:** Color-coded badges and priority ordering
- **JSON Export:** Structured data for databases and APIs
- **AI Transparency:** Includes reasoning chain in reports

### test_cosmos.py

- **Mock Mode:** Test report generation without GPU
- **Test Image Generation:** Creates synthetic solar panel images
- **Video Support:** Processes drone video footage
- **Automated Pipeline:** End-to-end testing

---

## 📊 Output Format

### Markdown Report Sections

1. **Executive Summary** - Status and key findings
2. **Flight Details** - Mission metadata table
3. **Assets Inspected** - What was surveyed
4. **Findings** - Organized by severity (Critical → Warning → Info)
5. **AI Reasoning Chain** - Full `<think>` output (collapsible)
6. **Recommended Actions** - Prioritized next steps
7. **Appendix** - Raw data and technical details

### JSON Report Structure

```json
{
  "metadata": {
    "report_id": "PARALLAX-20260205-143022",
    "generated_at": "2026-02-05T14:30:22",
    "system_version": "PARALLAX v1.0",
    "ai_model": "NVIDIA Cosmos-Reason2-8B"
  },
  "mission": { ... },
  "analysis": {
    "thinking": "...",
    "answer": "..."
  },
  "assessment": {
    "overall_status": "WARNING",
    "total_findings": 3,
    "critical_count": 0,
    "warning_count": 1,
    "info_count": 2
  },
  "findings": [
    {
      "severity": "WARNING",
      "description": "Corrosion detected on panel 2..."
    }
  ]
}
```

---

## ⚙️ Configuration

### Model Parameters

Edit in `cosmos_analyzer.py`:

```python
analyzer = CosmosAnalyzer(
    model_id="nvidia/Cosmos-Reason2-8B",  # Model identifier
    device="cuda",                         # cuda or cpu
    torch_dtype=torch.bfloat16,           # bfloat16 recommended
    max_new_tokens=4096,                  # Max response length
)
```

### Custom Inspection Prompts

```python
custom_prompt = """Analyze this image for:
- Structural integrity
- Rust or corrosion
- Missing components
- Safety violations
"""

result = analyzer.analyze_image("photo.jpg", custom_prompt)
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: transformers"

```bash
pip install transformers>=4.57.0
```

### "CUDA out of memory"

- Reduce `max_new_tokens` to 2048 or 1024
- Use a GPU with more VRAM (32GB minimum)
- Close other GPU-using processes

### "qwen-vl-utils not found"

```bash
pip install qwen-vl-utils
```

Or the analyzer will use fallback video processing.

### Model Download Fails

```bash
# Pre-download the model
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason2-8B")
```

---

## 📈 Performance

**L40S 48GB GPU (Brev.dev):**
- **Image Analysis:** ~15-30 seconds per image
- **Video Analysis:** ~45-90 seconds per video (4 FPS sampling)
- **VRAM Usage:** ~28-32GB for BF16
- **Batch Processing:** Not recommended (sequential is more stable)

---

## 🔗 Integration with Existing Codebase

### With MockDrone

```python
from src.drone import MockDrone
from cosmos_analyzer import CosmosAnalyzer

# Capture image from mock drone
drone = MockDrone()
await drone.connect()
await drone.takeoff()
image = await drone.capture_image()

# Save and analyze
image.save("capture.jpg")
analyzer = CosmosAnalyzer()
result = analyzer.analyze_drone_footage("capture.jpg", is_video=False)
```

### With Nemotron Commands

```python
from src.llm import NemotronClient
from cosmos_analyzer import CosmosAnalyzer

# Parse command
nemotron = NemotronClient()
cmd = await nemotron.parse_command("Inspect the solar panels")

if cmd.command_type.value == "inspect":
    # Capture and analyze
    image = await drone.capture_image()
    analyzer = CosmosAnalyzer()
    result = analyzer.analyze_drone_footage("image.jpg", is_video=False)
```

---

## 📝 License & Credits

**Created for:** NVIDIA GTC 2026 Competition  
**Model:** nvidia/Cosmos-Reason2-8B (Qwen3-VL architecture)  
**Platform:** Brev.dev GPU instances  

---

## 🎓 Next Steps

1. **Test on Real Drone Footage:** Replace test images with actual inspection videos
2. **Tune Prompts:** Customize for your specific infrastructure (powerlines, bridges, etc.)
3. **Integrate with LangGraph:** Build multi-agent workflows
4. **Deploy to Production:** Set up continuous inspection pipeline
5. **Add Isaac Sim:** Integrate with NVIDIA Isaac for simulation training

---

**Questions?** Check the code comments or run `python test_cosmos.py --help`
