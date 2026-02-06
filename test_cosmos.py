#!/usr/bin/env python3
"""
PARALLAX - Cosmos Integration Test Script
==========================================

Tests the full pipeline:
1. CosmosAnalyzer for image/video analysis
2. InspectionReportGenerator for report creation
3. Output verification

Usage:
    # Test with mock data (no GPU required)
    python test_cosmos.py --mock
    
    # Test with generated test image
    python test_cosmos.py
    
    # Test with real image
    python test_cosmos.py --image path/to/image.jpg
    
    # Test with real video
    python test_cosmos.py --video path/to/video.mp4
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image(output_path: str = "/tmp/test_solar_panel.jpg") -> str:
    """
    Create a simple test image simulating solar panels for testing.
    
    Args:
        output_path: Where to save the test image
        
    Returns:
        Path to created image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random
        
        logger.info("Generating test image (simulated solar panel inspection)")
        
        # Create 1920x1080 image
        img = Image.new('RGB', (1920, 1080), color=(135, 206, 235))  # Sky blue background
        draw = ImageDraw.Draw(img)
        
        # Draw ground
        draw.rectangle([(0, 800), (1920, 1080)], fill=(34, 139, 34))  # Green ground
        
        # Draw solar panel array (3x4 grid)
        panel_width = 200
        panel_height = 120
        spacing = 20
        start_x = 400
        start_y = 400
        
        for row in range(3):
            for col in range(4):
                x = start_x + col * (panel_width + spacing)
                y = start_y + row * (panel_height + spacing)
                
                # Draw panel (dark blue)
                draw.rectangle(
                    [(x, y), (x + panel_width, y + panel_height)],
                    fill=(25, 25, 112),
                    outline=(0, 0, 0),
                    width=3
                )
                
                # Add grid lines (simulating cells)
                for i in range(1, 4):
                    grid_x = x + i * (panel_width // 4)
                    draw.line([(grid_x, y), (grid_x, y + panel_height)], fill=(50, 50, 150), width=1)
                
                for i in range(1, 3):
                    grid_y = y + i * (panel_height // 3)
                    draw.line([(x, grid_y), (x + panel_width, grid_y)], fill=(50, 50, 150), width=1)
        
        # Add some "damage" to one panel (brown spot for corrosion)
        damage_x = start_x + panel_width + spacing
        damage_y = start_y
        draw.ellipse(
            [(damage_x + 50, damage_y + 30), (damage_x + 100, damage_y + 70)],
            fill=(139, 69, 19)  # Brown for rust/damage
        )
        
        # Add some vegetation (green patches)
        for _ in range(5):
            veg_x = random.randint(0, 1920)
            veg_y = random.randint(800, 1000)
            veg_size = random.randint(20, 50)
            draw.ellipse(
                [(veg_x, veg_y), (veg_x + veg_size, veg_y + veg_size)],
                fill=(0, 100, 0)
            )
        
        # Add text label
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        draw.text((50, 50), "SOLAR PANEL ARRAY - TEST SITE A", fill=(255, 255, 255), font=font)
        draw.text((50, 100), "Inspection Date: 2026-02-05", fill=(255, 255, 255), font=font)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        
        logger.info(f"Test image saved to: {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Failed to create test image: {e}")
        raise


def get_mock_cosmos_output() -> Dict[str, Any]:
    """
    Generate mock Cosmos output for testing without GPU.
    
    Returns:
        Mock analysis output
    """
    logger.info("Using mock Cosmos output (--mock mode)")
    
    return {
        "thinking": """Let me analyze this infrastructure inspection image systematically.

First, I'll examine the overall structure:
- I can see a solar panel array arranged in a 3x4 grid configuration
- The panels appear to be ground-mounted on what looks like a field or cleared area
- The background shows clear sky conditions, good for inspection

Equipment Condition Assessment:
- Most panels (11 out of 12) appear to be in good condition with intact frames and glass
- Panel surfaces show the typical grid pattern of photovoltaic cells
- Mounting structures appear stable

Damage/Corrosion Analysis:
- I notice a brown discoloration on the second panel in the top row
- This appears to be corrosion or rust, possibly from moisture ingress or frame degradation
- The affected area is approximately 50-70 square centimeters
- This could impact panel efficiency and requires attention

Vegetation Encroachment:
- I observe several patches of vegetation (appears to be grass or low shrubs) growing near the base of the array
- At least 5 distinct patches visible in the lower portion of the image
- While not currently blocking panels, unchecked growth could lead to shading issues

Safety Hazards:
- No immediate safety hazards visible (no exposed wiring, broken glass, or structural damage)
- The corrosion on panel 2 should be monitored as it could worsen

Overall Assessment:
This is a functional solar array with one panel showing signs of degradation. The vegetation growth is minor but should be addressed during routine maintenance.""",
        
        "answer": """**Infrastructure Inspection Results:**

**Equipment Condition:** GOOD
- 11 of 12 panels in good working condition
- Grid configuration: 3 rows × 4 columns
- Mounting structures stable

**Damage/Corrosion:** WARNING
- Panel 2 (top row, second from left) shows brown corrosion/rust spot
- Affected area: ~50-70 cm²
- Likely cause: moisture ingress or frame deterioration
- Recommendation: Schedule maintenance to prevent spreading

**Safety Hazards:** NONE
- No exposed wiring or structural damage detected
- No broken glass or immediate risks

**Vegetation Encroachment:** INFO
- 5 patches of vegetation growth detected near array base
- Current impact: minimal
- Recommendation: routine clearing to prevent future shading

**Overall Assessment:** WARNING
The solar array is generally in good condition with one panel requiring attention. The corrosion on panel 2 should be addressed within 2-4 weeks to prevent further degradation. Vegetation should be cleared during next scheduled maintenance.""",
        
        "raw_output": "<think>Analyzing solar panel array...</think>\n<answer>Equipment mostly good, one panel with corrosion warning...</answer>",
        
        "findings": [
            {
                "severity": "WARNING",
                "description": "Panel 2 shows brown corrosion spot (~50-70 cm²), requires maintenance within 2-4 weeks"
            },
            {
                "severity": "INFO",
                "description": "5 patches of vegetation growth near array base, minimal current impact"
            },
            {
                "severity": "INFO",
                "description": "11 of 12 panels in good working condition"
            }
        ]
    }


def run_real_analysis(image_path: str = None, video_path: str = None) -> Dict[str, Any]:
    """
    Run real Cosmos analysis on image or video.
    
    Args:
        image_path: Path to image file
        video_path: Path to video file
        
    Returns:
        Analysis output
    """
    try:
        from cosmos_analyzer import CosmosAnalyzer
        
        logger.info("Loading Cosmos Reason 2 model...")
        analyzer = CosmosAnalyzer()
        
        if video_path:
            logger.info(f"Analyzing video: {video_path}")
            result = analyzer.analyze_drone_footage(video_path, is_video=True)
        else:
            logger.info(f"Analyzing image: {image_path}")
            result = analyzer.analyze_drone_footage(image_path, is_video=False)
        
        # Cleanup
        del analyzer
        
        return result
    
    except ImportError as e:
        logger.error(f"Failed to import cosmos_analyzer: {e}")
        logger.error("Make sure transformers>=4.57.0 is installed")
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


def generate_and_save_reports(cosmos_output: Dict[str, Any], output_dir: str = "/workspace/output"):
    """
    Generate and save both Markdown and JSON reports.
    
    Args:
        cosmos_output: Analysis output from Cosmos
        output_dir: Directory to save reports
    """
    try:
        from report_generator import InspectionReportGenerator
        
        logger.info("Generating inspection reports...")
        
        generator = InspectionReportGenerator()
        
        # Create mission metadata
        mission_metadata = {
            "location": "Solar Farm Site A - Test Installation",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "drone_model": "DJI Mavic 3 Enterprise",
            "flight_duration": "12 minutes",
            "waypoints": 8,
            "operator": "PARALLAX Autonomous System",
            "assets_inspected": "Solar panel array (12 panels in 3×4 configuration)",
        }
        
        # Generate reports
        md_report = generator.generate_report(cosmos_output, mission_metadata)
        json_report = generator.generate_json_report(cosmos_output, mission_metadata)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = generator.save_report(md_report, output_path / f"inspection_report_{timestamp}.md")
        json_file = generator.save_json_report(json_report, output_path / f"inspection_report_{timestamp}.json")
        
        logger.info(f"✓ Markdown report: {md_file}")
        logger.info(f"✓ JSON report: {json_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("INSPECTION REPORT SUMMARY")
        print("="*70)
        print(f"Status: {json_report['assessment']['overall_status']}")
        print(f"Total Findings: {json_report['assessment']['total_findings']}")
        print(f"  - Critical: {json_report['assessment']['critical_count']}")
        print(f"  - Warnings: {json_report['assessment']['warning_count']}")
        print(f"  - Info: {json_report['assessment']['info_count']}")
        print("\nKey Findings:")
        for i, finding in enumerate(cosmos_output.get('findings', [])[:3], 1):
            print(f"  {i}. [{finding.get('severity')}] {finding.get('description')}")
        print("\nReports saved to:", output_dir)
        print("="*70 + "\n")
        
        return md_file, json_file
    
    except ImportError as e:
        logger.error(f"Failed to import report_generator: {e}")
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


def main():
    """Main test script."""
    parser = argparse.ArgumentParser(
        description="Test PARALLAX Cosmos integration pipeline"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of running real model (no GPU required)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to analyze"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/output",
        help="Output directory for reports (default: /workspace/output)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PARALLAX - COSMOS REASON 2 INTEGRATION TEST")
    print("NVIDIA GTC 2026 Competition")
    print("="*70 + "\n")
    
    try:
        # Step 1: Get or generate input
        if args.mock:
            logger.info("Running in MOCK mode (no GPU required)")
            cosmos_output = get_mock_cosmos_output()
        
        elif args.video:
            logger.info(f"Video analysis mode: {args.video}")
            if not Path(args.video).exists():
                logger.error(f"Video file not found: {args.video}")
                sys.exit(1)
            cosmos_output = run_real_analysis(video_path=args.video)
        
        elif args.image:
            logger.info(f"Image analysis mode: {args.image}")
            if not Path(args.image).exists():
                logger.error(f"Image file not found: {args.image}")
                sys.exit(1)
            cosmos_output = run_real_analysis(image_path=args.image)
        
        else:
            # Generate test image and analyze
            logger.info("No input provided, generating test image")
            test_image = create_test_image()
            cosmos_output = run_real_analysis(image_path=test_image)
        
        # Step 2: Generate reports
        logger.info("Analysis complete, generating reports...")
        md_file, json_file = generate_and_save_reports(cosmos_output, args.output)
        
        # Step 3: Success
        print("\n✓ Test completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. View markdown report: {md_file}")
        print(f"  2. View JSON data: {json_file}")
        print(f"  3. Integrate into your drone control pipeline\n")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Test cancelled by user")
        return 1
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
