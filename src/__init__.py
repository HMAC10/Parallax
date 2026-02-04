"""
Drone AI Command System
=======================

A natural language drone control system using NVIDIA Nemotron, VILA, 
Isaac Sim, and LangGraph for the NVIDIA GTC 2026 competition.

Modules:
    - config: Application configuration and settings
    - drone: Drone interface and implementations
    - llm: Language model clients (Nemotron, VILA)
    - agents: LangGraph-based AI agents
    - optimization: Path and mission optimization
"""

__version__ = "0.1.0"
__author__ = "GTC 2026 Team"

from .config import settings

__all__ = ["settings", "__version__"]
