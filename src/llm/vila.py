"""
VILA Client for NVIDIA Vision-Language API.

Handles image analysis using NVIDIA's VILA model for:
- Object detection and identification
- Scene understanding
- Anomaly detection
- Search and rescue scenarios
"""

import base64
import io
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path
import httpx
from PIL import Image

from ..config import settings


@dataclass
class ImageAnalysis:
    """Result of VILA image analysis."""
    description: str = ""
    objects_detected: list[str] = field(default_factory=list)
    scene_type: str = ""
    anomalies: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "objects_detected": self.objects_detected,
            "scene_type": self.scene_type,
            "anomalies": self.anomalies,
            "confidence": self.confidence,
        }
    
    def has_anomaly(self) -> bool:
        """Check if any anomalies were detected."""
        return len(self.anomalies) > 0
    
    def contains_object(self, obj: str) -> bool:
        """Check if a specific object was detected."""
        obj_lower = obj.lower()
        return any(obj_lower in detected.lower() for detected in self.objects_detected)


class VILAClient:
    """
    Client for NVIDIA VILA vision-language model.
    
    Provides image analysis capabilities including:
    - General scene description
    - Object detection
    - Anomaly identification
    - Custom query answering about images
    """
    
    ANALYSIS_PROMPT = """Analyze this aerial/drone image and provide:
1. A brief description of the scene
2. List of objects/features detected
3. Scene type (urban, rural, forest, water, industrial, etc.)
4. Any anomalies or unusual observations

Respond in this exact format:
DESCRIPTION: <scene description>
OBJECTS: <comma-separated list of objects>
SCENE_TYPE: <scene classification>
ANOMALIES: <comma-separated list, or "none">"""

    SEARCH_PROMPT = """You are analyzing aerial drone imagery for search and rescue.
Look for any signs of:
- People or human activity
- Vehicles or equipment
- Structures or shelters
- Paths or trails
- Distress signals or markers
- Heat signatures or unusual patterns

Describe what you see and highlight anything that could indicate human presence."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize VILA client.
        
        Args:
            api_key: NVIDIA API key (defaults to settings)
            base_url: API base URL (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.nvidia_api_key
        self.base_url = base_url or settings.nvidia_base_url
        self.model = model or settings.vila_model
        
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,  # Vision models may take longer
        )
    
    def _encode_image(self, image: Image.Image, format: str = "JPEG") -> str:
        """
        Encode PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            format: Image format for encoding
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _load_image(self, source: str | Path | Image.Image) -> Image.Image:
        """
        Load image from various sources.
        
        Args:
            source: File path, URL, or PIL Image
            
        Returns:
            PIL Image object
        """
        if isinstance(source, Image.Image):
            return source
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                return Image.open(path)
        
        raise ValueError(f"Could not load image from: {source}")
    
    async def _call_api(
        self,
        image_b64: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """
        Make API call to VILA.
        
        Args:
            image_b64: Base64 encoded image
            prompt: Text prompt for analysis
            max_tokens: Maximum response tokens
            
        Returns:
            Model response text
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def _parse_analysis_response(self, response: str) -> ImageAnalysis:
        """
        Parse structured analysis response.
        
        Args:
            response: Raw model response
            
        Returns:
            Parsed ImageAnalysis object
        """
        analysis = ImageAnalysis(raw_response=response)
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("DESCRIPTION:"):
                analysis.description = line.replace("DESCRIPTION:", "").strip()
            
            elif line.startswith("OBJECTS:"):
                objects_str = line.replace("OBJECTS:", "").strip()
                if objects_str.lower() != "none":
                    analysis.objects_detected = [
                        obj.strip() for obj in objects_str.split(",")
                        if obj.strip()
                    ]
            
            elif line.startswith("SCENE_TYPE:"):
                analysis.scene_type = line.replace("SCENE_TYPE:", "").strip()
            
            elif line.startswith("ANOMALIES:"):
                anomalies_str = line.replace("ANOMALIES:", "").strip()
                if anomalies_str.lower() != "none":
                    analysis.anomalies = [
                        a.strip() for a in anomalies_str.split(",")
                        if a.strip()
                    ]
        
        # If structured parsing failed, use the whole response as description
        if not analysis.description:
            analysis.description = response
        
        # Set confidence based on how well the response was structured
        structured_fields = sum([
            bool(analysis.description),
            bool(analysis.objects_detected),
            bool(analysis.scene_type),
        ])
        analysis.confidence = structured_fields / 3.0
        
        return analysis
    
    async def analyze_image(
        self,
        image: str | Path | Image.Image,
    ) -> ImageAnalysis:
        """
        Perform general analysis on an image.
        
        Args:
            image: Image source (path, PIL Image, or base64)
            
        Returns:
            ImageAnalysis with results
        """
        try:
            pil_image = self._load_image(image)
            image_b64 = self._encode_image(pil_image)
            
            response = await self._call_api(image_b64, self.ANALYSIS_PROMPT)
            return self._parse_analysis_response(response)
            
        except Exception as e:
            return ImageAnalysis(
                description=f"Analysis failed: {str(e)}",
                confidence=0.0,
            )
    
    async def search_for_target(
        self,
        image: str | Path | Image.Image,
        target_description: str,
    ) -> dict:
        """
        Search image for a specific target.
        
        Args:
            image: Image source
            target_description: Description of what to find
            
        Returns:
            Dictionary with search results
        """
        prompt = f"""Search this aerial image for: {target_description}

Report:
1. Is the target visible? (yes/no/possibly)
2. If visible, describe its location in the image
3. Confidence level (low/medium/high)
4. Any related observations

Format:
TARGET_FOUND: <yes/no/possibly>
LOCATION: <description or "not visible">
CONFIDENCE: <low/medium/high>
OBSERVATIONS: <additional notes>"""
        
        try:
            pil_image = self._load_image(image)
            image_b64 = self._encode_image(pil_image)
            
            response = await self._call_api(image_b64, prompt)
            
            # Parse response
            result = {
                "target": target_description,
                "found": False,
                "location": "",
                "confidence": "low",
                "observations": "",
                "raw_response": response,
            }
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith("TARGET_FOUND:"):
                    value = line.replace("TARGET_FOUND:", "").strip().lower()
                    result["found"] = value in ("yes", "possibly")
                elif line.startswith("LOCATION:"):
                    result["location"] = line.replace("LOCATION:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    result["confidence"] = line.replace("CONFIDENCE:", "").strip().lower()
                elif line.startswith("OBSERVATIONS:"):
                    result["observations"] = line.replace("OBSERVATIONS:", "").strip()
            
            return result
            
        except Exception as e:
            return {
                "target": target_description,
                "found": False,
                "error": str(e),
            }
    
    async def detect_anomalies(
        self,
        image: str | Path | Image.Image,
        baseline_description: Optional[str] = None,
    ) -> list[str]:
        """
        Detect anomalies in an image.
        
        Args:
            image: Image source
            baseline_description: Optional description of normal conditions
            
        Returns:
            List of detected anomalies
        """
        prompt = "Identify any anomalies, unusual features, or potential hazards in this aerial image."
        
        if baseline_description:
            prompt += f"\n\nNormal conditions should look like: {baseline_description}"
        
        prompt += "\n\nList each anomaly on a new line, prefixed with '- '. If no anomalies, respond with 'No anomalies detected.'"
        
        try:
            pil_image = self._load_image(image)
            image_b64 = self._encode_image(pil_image)
            
            response = await self._call_api(image_b64, prompt)
            
            if "no anomalies" in response.lower():
                return []
            
            anomalies = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    anomalies.append(line[1:].strip())
                elif line and not line.startswith(('No', 'None')):
                    anomalies.append(line)
            
            return anomalies
            
        except Exception as e:
            return [f"Error during analysis: {str(e)}"]
    
    async def ask_about_image(
        self,
        image: str | Path | Image.Image,
        question: str,
    ) -> str:
        """
        Ask a custom question about an image.
        
        Args:
            image: Image source
            question: Question to ask about the image
            
        Returns:
            Model's answer
        """
        try:
            pil_image = self._load_image(image)
            image_b64 = self._encode_image(pil_image)
            
            return await self._call_api(image_b64, question)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def analyze_for_search_rescue(
        self,
        image: str | Path | Image.Image,
    ) -> dict:
        """
        Specialized analysis for search and rescue scenarios.
        
        Args:
            image: Image source
            
        Returns:
            Dictionary with SAR-specific analysis
        """
        try:
            pil_image = self._load_image(image)
            image_b64 = self._encode_image(pil_image)
            
            response = await self._call_api(image_b64, self.SEARCH_PROMPT, max_tokens=1024)
            
            # Look for key indicators in response
            response_lower = response.lower()
            
            return {
                "analysis": response,
                "human_presence_indicated": any(word in response_lower for word in 
                    ["person", "people", "human", "figure", "body", "individual"]),
                "vehicle_detected": any(word in response_lower for word in 
                    ["car", "vehicle", "truck", "boat", "aircraft"]),
                "structure_detected": any(word in response_lower for word in 
                    ["building", "structure", "tent", "shelter", "camp"]),
                "distress_signal": any(word in response_lower for word in 
                    ["signal", "sos", "marker", "flag", "fire"]),
                "priority": "high" if "person" in response_lower or "human" in response_lower else "normal",
            }
            
        except Exception as e:
            return {
                "analysis": f"Error: {str(e)}",
                "priority": "error",
            }
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function for quick image analysis
async def analyze_drone_image(image: str | Path | Image.Image) -> ImageAnalysis:
    """
    Quick utility to analyze a single image.
    
    Args:
        image: Image source
        
    Returns:
        ImageAnalysis results
    """
    async with VILAClient() as client:
        return await client.analyze_image(image)
