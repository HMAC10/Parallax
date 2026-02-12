"""
PARALLAX - Cosmos Reason 2 Analyzer
====================================

Integrates NVIDIA's Cosmos-Reason2-8B model for drone footage analysis.
Uses Qwen3-VL architecture with chain-of-thought reasoning.

Requirements:
    - 32GB+ VRAM (tested on L40S 48GB)
    - transformers>=4.57.0
    - torch with CUDA support
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union, List
import torch
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CosmosAnalyzer:
    """
    Analyzer for drone footage using NVIDIA Cosmos Reason 2 model.
    
    This model provides chain-of-thought reasoning for visual inspection tasks,
    outputting structured thinking process and final answers.
    """
    
    MODEL_ID = "nvidia/Cosmos-Reason2-8B"
    
    # System prompt template for chain-of-thought reasoning
    SYSTEM_PROMPT = """You are an expert infrastructure inspection AI assistant. Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>"""
    
    # Infrastructure inspection prompt template
    INFRASTRUCTURE_INSPECTION_PROMPT = """You are analyzing aerial drone inspection footage of urban utility poles in Seattle, WA.

This footage shows a sequential inspection of 4 utility poles along a city block. Analyze EACH pole individually.

For EACH pole visible in the footage, report:
1. Pole ID (Pole 1, Pole 2, etc. in order of appearance)
2. Pole Condition: Surface condition, rust, corrosion, leaning, paint condition
3. Hardware & Connections: State of light fixtures, wire attachments, mounting brackets
4. Vegetation Encroachment: Any trees or branches within 3 feet of the pole or wires
5. Surrounding Hazards: Nearby construction, obstructions, or safety concerns
6. Severity Rating: CRITICAL / WARNING / INFO

Pay special attention to:
- Vegetation growing near or touching poles and wires (this is a common and serious issue)
- Condition of wire attachment points and insulators
- Any poles that appear to be leaning or damaged

Provide findings for EACH pole separately, then give an overall site assessment."""
    
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        max_new_tokens: int = 4096,
    ):
        """
        Initialize Cosmos Reason 2 analyzer.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run inference on ('cuda' or 'cpu')
            torch_dtype: Torch dtype for inference (bfloat16 recommended)
            max_new_tokens: Maximum tokens to generate
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"Initializing Cosmos Analyzer with model: {model_id}")
        logger.info(f"Device: {device}, dtype: {torch_dtype}")
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the Cosmos Reason 2 model and processor."""
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            
            logger.info("Loading model and processor...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Log VRAM usage if on CUDA
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        except ImportError as e:
            logger.error("Failed to import transformers. Install with: pip install transformers>=4.57.0")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _parse_model_output(self, text: str) -> Dict[str, str]:
        """
        Parse model output to extract thinking and answer sections.
        
        Args:
            text: Raw model output with <think> and <answer> tags
            
        Returns:
            Dictionary with 'thinking' and 'answer' keys
        """
        # Extract thinking section
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
        thinking = think_match.group(1).strip() if think_match else ""
        
        # Extract answer section
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else text.strip()
        
        return {
            "thinking": thinking,
            "answer": answer,
            "raw_output": text,
        }
    
    def _prepare_messages(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        Prepare messages for the model in the expected format.
        
        Args:
            prompt: Text prompt for analysis
            image_path: Path to image file
            video_path: Path to video file
            
        Returns:
            List of message dictionaries
        """
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            }
        ]
        
        # Build user message content
        content = []
        
        if image_path:
            content.append({
                "type": "image",
                "image": image_path,
            })
        
        if video_path:
            content.append({
                "type": "video",
                "video": video_path,
                "fps": 4,  # Sample at 4 FPS as recommended
            })
        
        content.append({
            "type": "text",
            "text": prompt,
        })
        
        messages.append({
            "role": "user",
            "content": content,
        })
        
        return messages
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
    ) -> Dict[str, str]:
        """
        Analyze a single image with a custom prompt.
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            
        Returns:
            Dictionary with 'thinking' and 'answer' keys
        """
        image_path = str(Path(image_path).resolve())
        
        logger.info(f"Analyzing image: {image_path}")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt=prompt, image_path=image_path)
            
            # Apply chat template
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text_prompt],
                images=[Image.open(image_path)],
                return_tensors="pt",
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            logger.info("Generating analysis...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Deterministic for inspection tasks
                )
            
            # Decode output
            generated_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
            )[0]
            
            logger.info("Analysis complete")
            
            # Parse and return
            return self._parse_model_output(generated_text)
        
        except Exception as e:
            logger.error(f"Error during image analysis: {e}")
            return {
                "thinking": "",
                "answer": f"Error: {str(e)}",
                "raw_output": "",
                "error": str(e),
            }
    
    def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: str,
        fps: int = 4,
    ) -> Dict[str, str]:
        """
        Analyze a video with a custom prompt.
        
        Args:
            video_path: Path to video file
            prompt: Analysis prompt
            fps: Frame sampling rate (default: 4 FPS)
            
        Returns:
            Dictionary with 'thinking' and 'answer' keys
        """
        video_path = str(Path(video_path).resolve())
        
        logger.info(f"Analyzing video: {video_path}")
        logger.info(f"Sampling at {fps} FPS")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt=prompt, video_path=video_path)
            
            # Apply chat template
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Process video
            # Note: qwen-vl-utils handles video loading and frame extraction
            try:
                from qwen_vl_utils import process_vision_info
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except ImportError:
                logger.warning("qwen-vl-utils not available, using basic video processing")
                # Fallback: extract frames manually
                inputs = self.processor(
                    text=[text_prompt],
                    videos=[video_path],
                    return_tensors="pt",
                )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            logger.info("Generating analysis...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            
            # Decode output
            generated_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
            )[0]
            
            logger.info("Analysis complete")
            
            # Parse and return
            return self._parse_model_output(generated_text)
        
        except Exception as e:
            logger.error(f"Error during video analysis: {e}")
            return {
                "thinking": "",
                "answer": f"Error: {str(e)}",
                "raw_output": "",
                "error": str(e),
            }
    
    def analyze_drone_footage(
        self,
        footage_path: Union[str, Path],
        is_video: bool = True,
    ) -> Dict[str, str]:
        """
        Analyze drone footage for infrastructure inspection.
        
        This is a convenience wrapper that uses a specialized infrastructure
        inspection prompt.
        
        Args:
            footage_path: Path to video or image file
            is_video: Whether the footage is a video (True) or image (False)
            
        Returns:
            Dictionary with 'thinking' and 'answer' keys, plus parsed findings
        """
        logger.info(f"Starting infrastructure inspection analysis")
        
        if is_video:
            result = self.analyze_video(
                footage_path,
                self.INFRASTRUCTURE_INSPECTION_PROMPT
            )
        else:
            result = self.analyze_image(
                footage_path,
                self.INFRASTRUCTURE_INSPECTION_PROMPT
            )
        
        # Parse findings from answer
        result["findings"] = self._parse_findings(result.get("answer", ""))
        
        return result
    
    def _parse_findings(self, answer: str) -> List[Dict[str, str]]:
        """
        Parse structured findings from the answer text.
        
        Args:
            answer: The answer text from model output
            
        Returns:
            List of finding dictionaries with category, description, and severity
        """
        findings = []
        
        # Look for severity markers
        for severity in ["CRITICAL", "WARNING", "INFO"]:
            # Find all instances of this severity level
            pattern = rf'{severity}[:\s]+([^\n]+)'
            matches = re.finditer(pattern, answer, re.IGNORECASE)
            
            for match in matches:
                findings.append({
                    "severity": severity,
                    "description": match.group(1).strip(),
                })
        
        # If no structured findings found, return the whole answer as INFO
        if not findings:
            findings.append({
                "severity": "INFO",
                "description": answer,
            })
        
        return findings
    
    def __del__(self):
        """Cleanup VRAM on deletion."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("VRAM cleared")


# Convenience function for quick analysis
def analyze_image_quick(image_path: str, prompt: str) -> Dict[str, str]:
    """
    Quick utility to analyze a single image.
    
    Args:
        image_path: Path to image
        prompt: Analysis prompt
        
    Returns:
        Analysis result dictionary
    """
    analyzer = CosmosAnalyzer()
    result = analyzer.analyze_image(image_path, prompt)
    del analyzer
    return result
