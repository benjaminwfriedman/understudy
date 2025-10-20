"""
Understudy Client - Python SDK for Understudy LLM to SLM platform.

This package provides:
- Direct client for Understudy API
- LangChain LLM wrappers (if langchain is installed)
- LangGraph integration utilities (if langgraph is installed)
"""

from understudy.client import Understudy

__version__ = "1.0.0"
__all__ = ["Understudy"]

# Optional LangChain integration
try:
    from understudy.langchain_integration import UnderstudyLLM, UnderstudyChatModel
    __all__.extend(["UnderstudyLLM", "UnderstudyChatModel"])
except ImportError:
    pass

# Optional LangGraph integration  
try:
    from understudy.langgraph_integration import create_understudy_node
    __all__.extend(["create_understudy_node"])
except ImportError:
    pass