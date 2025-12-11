"""
Shared Pydantic schemas used across agents and LLMs.
"""
from typing import Optional, Dict
from pydantic import BaseModel


class ActionModel(BaseModel):
    """Schema for a tool action in ReAct."""
    reason: str
    server: str
    tool: str
    arguments: Dict[str, str]


class ReActStep(BaseModel):
    """Schema for a ReAct reasoning step."""
    thought: str
    action: Optional[ActionModel] = None
    answer: Optional[str] = None
