"""
Nemotron Client for NVIDIA NIM API.

Handles natural language command parsing and report generation
using the Nemotron model via NVIDIA's NIM API.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import httpx

from ..config import settings


class CommandType(Enum):
    """Types of drone commands that can be parsed."""
    TAKEOFF = "takeoff"
    LAND = "land"
    GOTO = "goto"
    ROTATE = "rotate"
    HOVER = "hover"
    CAPTURE = "capture"
    SURVEY = "survey"
    INSPECT = "inspect"
    RETURN_HOME = "return_home"
    STATUS = "status"
    ARM = "arm"
    DISARM = "disarm"
    ABORT = "abort"
    UNKNOWN = "unknown"


@dataclass
class DroneCommand:
    """Parsed drone command from natural language."""
    command_type: CommandType
    parameters: dict = field(default_factory=dict)
    confidence: float = 0.0
    raw_text: str = ""
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        """Convert command to dictionary."""
        return {
            "command_type": self.command_type.value,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "reasoning": self.reasoning,
        }


class NemotronClient:
    """
    Client for NVIDIA Nemotron model via NIM API.
    
    Used for:
    - Parsing natural language into drone commands
    - Generating mission reports
    - Interpreting complex instructions
    """
    
    COMMAND_PARSING_PROMPT = """You are a drone command parser. Convert natural language instructions into structured drone commands.

Available commands:
- takeoff: Take off to altitude (params: altitude_meters)
- land: Land at current position
- goto: Navigate to position (params: x, y, z OR relative_x, relative_y, relative_z, speed)
- rotate: Rotate drone (params: degrees, relative=true/false)
- hover: Stop and hover in place
- capture: Take a photo
- survey: Survey an area (params: area_size, pattern)
- return_home: Return to starting position
- status: Report current status
- arm: Arm the motors
- disarm: Disarm the motors
- abort: Emergency stop

Respond with ONLY valid JSON in this exact format:
{
    "command_type": "<command>",
    "parameters": {<relevant parameters>},
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}

Examples:
User: "fly up to 10 meters"
{"command_type": "takeoff", "parameters": {"altitude_meters": 10}, "confidence": 0.95, "reasoning": "Clear takeoff command with altitude"}

User: "go forward 5 meters"
{"command_type": "goto", "parameters": {"relative_x": 5, "relative_y": 0, "relative_z": 0}, "confidence": 0.9, "reasoning": "Relative forward movement"}

User: "turn left 90 degrees"
{"command_type": "rotate", "parameters": {"degrees": -90, "relative": true}, "confidence": 0.95, "reasoning": "Relative left rotation"}

User: "take a picture"
{"command_type": "capture", "parameters": {}, "confidence": 0.98, "reasoning": "Photo capture request"}

Parse the following instruction:"""

    REPORT_GENERATION_PROMPT = """You are a professional drone mission report generator. 
Create clear, concise reports based on mission data provided.

Include:
- Mission summary
- Key observations
- Any anomalies or issues
- Recommendations

Keep the tone professional and factual."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize Nemotron client.
        
        Args:
            api_key: NVIDIA API key (defaults to settings)
            base_url: API base URL (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.nvidia_api_key
        self.base_url = base_url or settings.nvidia_base_url
        self.model = model or settings.nemotron_model
        
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
    
    async def _call_api(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """
        Make API call to Nemotron.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            Model response text
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def parse_command(self, user_input: str) -> DroneCommand:
        """
        Parse natural language into a drone command.
        
        Args:
            user_input: Natural language instruction
            
        Returns:
            Parsed DroneCommand
        """
        messages = [
            {"role": "system", "content": self.COMMAND_PARSING_PROMPT},
            {"role": "user", "content": user_input},
        ]
        
        try:
            response = await self._call_api(messages, temperature=0.1)
            
            if not response:
                return DroneCommand(
                    command_type=CommandType.UNKNOWN,
                    parameters={},
                    confidence=0.0,
                    raw_text=user_input,
                    reasoning="Empty response from API",
                )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response)
            
            # Create command - handle both "command_type" and "action" response formats
            cmd_type_str = parsed.get("command_type") or parsed.get("action", "unknown")
            try:
                command_type = CommandType(cmd_type_str)
            except ValueError:
                command_type = CommandType.UNKNOWN
            
            # Build parameters
            parameters = parsed.get("parameters", {})
            if "target" in parsed:
                parameters["target"] = parsed["target"]
            if "location" in parsed:
                parameters["location"] = parsed["location"]
            
            return DroneCommand(
                command_type=command_type,
                parameters=parameters,
                confidence=parsed.get("confidence", 0.5),
                raw_text=user_input,
                reasoning=parsed.get("reasoning", ""),
            )
            
        except (json.JSONDecodeError, httpx.HTTPError) as e:
            # Return unknown command on error
            return DroneCommand(
                command_type=CommandType.UNKNOWN,
                parameters={},
                confidence=0.0,
                raw_text=user_input,
                reasoning=f"Parse error: {str(e)}",
            )
    
    async def generate_report(
        self,
        mission_data: dict,
        observations: list[str],
        images_analyzed: int = 0,
    ) -> str:
        """
        Generate a mission report.
        
        Args:
            mission_data: Dictionary with mission details
            observations: List of observations/findings
            images_analyzed: Number of images processed
            
        Returns:
            Formatted mission report
        """
        context = f"""Mission Data:
- Start Time: {mission_data.get('start_time', 'N/A')}
- End Time: {mission_data.get('end_time', 'N/A')}
- Duration: {mission_data.get('duration', 'N/A')}
- Distance Traveled: {mission_data.get('distance', 'N/A')}
- Commands Executed: {mission_data.get('commands_count', 0)}
- Images Captured: {images_analyzed}
- Battery Used: {mission_data.get('battery_used', 'N/A')}%

Observations:
{chr(10).join(f'- {obs}' for obs in observations)}

Area Covered: {mission_data.get('area_covered', 'N/A')}
Weather Conditions: {mission_data.get('weather', 'N/A')}
"""
        
        messages = [
            {"role": "system", "content": self.REPORT_GENERATION_PROMPT},
            {"role": "user", "content": f"Generate a mission report based on this data:\n\n{context}"},
        ]
        
        try:
            report = await self._call_api(messages, temperature=0.3, max_tokens=1024)
            return report
        except httpx.HTTPError as e:
            return f"Report generation failed: {str(e)}"
    
    async def interpret_complex_instruction(
        self,
        instruction: str,
        context: Optional[dict] = None,
    ) -> list[DroneCommand]:
        """
        Interpret complex multi-step instructions.
        
        Args:
            instruction: Complex instruction that may require multiple commands
            context: Optional context about current mission state
            
        Returns:
            List of DroneCommands to execute in sequence
        """
        context_str = ""
        if context:
            context_str = f"\nCurrent context: {json.dumps(context)}"
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a drone mission planner. Break down complex instructions into a sequence of simple drone commands.

Available commands: takeoff, land, goto, rotate, hover, capture, survey, return_home, status, arm, disarm, abort

Respond with a JSON array of commands in execution order:
[
    {{"command_type": "...", "parameters": {{...}}, "confidence": 0.9, "reasoning": "..."}},
    ...
]
{context_str}""",
            },
            {"role": "user", "content": instruction},
        ]
        
        try:
            response = await self._call_api(messages, temperature=0.2, max_tokens=1024)
            
            # Extract JSON array
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                parsed_list = json.loads(json_match.group())
            else:
                parsed_list = json.loads(response)
            
            commands = []
            for item in parsed_list:
                try:
                    cmd = DroneCommand(
                        command_type=CommandType(item.get("command_type", "unknown")),
                        parameters=item.get("parameters", {}),
                        confidence=item.get("confidence", 0.7),
                        raw_text=instruction,
                        reasoning=item.get("reasoning", ""),
                    )
                    commands.append(cmd)
                except ValueError:
                    continue
            
            return commands if commands else [
                DroneCommand(
                    command_type=CommandType.UNKNOWN,
                    raw_text=instruction,
                    reasoning="Could not parse complex instruction",
                )
            ]
            
        except (json.JSONDecodeError, httpx.HTTPError) as e:
            return [
                DroneCommand(
                    command_type=CommandType.UNKNOWN,
                    raw_text=instruction,
                    reasoning=f"Parse error: {str(e)}",
                )
            ]
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function for quick command parsing
async def parse_drone_command(text: str) -> DroneCommand:
    """
    Quick utility to parse a single command.
    
    Args:
        text: Natural language command
        
    Returns:
        Parsed DroneCommand
    """
    async with NemotronClient() as client:
        return await client.parse_command(text)
