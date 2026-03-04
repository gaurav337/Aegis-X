"""Core data types for the Aegis-X system.

This module contains the unified interface and payload contracts for all internal
forensic tools and orchestration components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ToolResult:
    """Standardized output payload from any forensic tool in Aegis-X.
    
    This exact structure forms the sacred contract between the specialized forensic
    tools and the overarching ensemble evaluation logic.
    """
    tool_name: str
    success: bool
    score: float
    confidence: float
    details: Dict[str, Any]
    error: bool
    error_msg: Optional[str]
    execution_time: float
    evidence_summary: str
