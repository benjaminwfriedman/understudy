from typing import TypedDict, Annotated, Dict, Any, Callable
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from understudy.langchain_integration import UnderstudyChatModel
import logging

logger = logging.getLogger(__name__)


class UnderstudyGraphState(TypedDict):
    """Base state for LangGraph integration with Understudy."""
    messages: Annotated[list[BaseMessage], "The messages in the conversation"]
    endpoint_id: str
    metadata: Dict[str, Any]


def create_understudy_node(
    endpoint_id: str,
    base_url: str = "http://localhost:8000",
    api_key: str = None,
    node_name: str = "understudy_node",
    **llm_kwargs
) -> Callable:
    """
    Create a LangGraph node powered by Understudy.
    
    Args:
        endpoint_id: The Understudy endpoint ID to use
        base_url: Base URL of the Understudy API
        api_key: Optional API key for authentication
        node_name: Name for this node (for logging)
        **llm_kwargs: Additional arguments for the LLM
    
    Returns:
        A function that can be used as a LangGraph node
    """
    # Create the Understudy chat model
    llm = UnderstudyChatModel(
        endpoint_id=endpoint_id,
        base_url=base_url,
        api_key=api_key,
        **llm_kwargs
    )
    
    def node_function(state: UnderstudyGraphState) -> Dict[str, Any]:
        """The actual node function that will be called by LangGraph."""
        try:
            messages = state.get("messages", [])
            if not messages:
                logger.warning(f"{node_name}: No messages in state")
                return {"messages": []}
            
            # Generate response using Understudy
            logger.info(f"{node_name}: Generating response for {len(messages)} messages")
            
            # Invoke the LLM
            result = llm.invoke(messages)
            
            # Extract the AI message
            if hasattr(result, 'content'):
                ai_message = AIMessage(content=result.content)
            else:
                ai_message = result
            
            # Return updated state
            updated_state = {
                "messages": [ai_message]
            }
            
            # Add metadata if available
            if hasattr(result, 'response_metadata'):
                updated_state["metadata"] = result.response_metadata
            
            logger.info(f"{node_name}: Generated response successfully")
            return updated_state
            
        except Exception as e:
            logger.error(f"{node_name}: Error generating response: {e}")
            # Return an error message
            error_message = AIMessage(content=f"Error: {str(e)}")
            return {"messages": [error_message]}
    
    return node_function


def create_understudy_chain_node(
    endpoint_id: str,
    prompt_template: str,
    base_url: str = "http://localhost:8000",
    api_key: str = None,
    node_name: str = "understudy_chain_node",
    **llm_kwargs
) -> Callable:
    """
    Create a LangGraph node that uses a prompt template with Understudy.
    
    Args:
        endpoint_id: The Understudy endpoint ID to use
        prompt_template: Template string with {variables} to format
        base_url: Base URL of the Understudy API
        api_key: Optional API key for authentication
        node_name: Name for this node (for logging)
        **llm_kwargs: Additional arguments for the LLM
    
    Returns:
        A function that can be used as a LangGraph node
    """
    from langchain_core.prompts import ChatPromptTemplate
    
    # Create the Understudy chat model
    llm = UnderstudyChatModel(
        endpoint_id=endpoint_id,
        base_url=base_url,
        api_key=api_key,
        **llm_kwargs
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Create chain
    chain = prompt | llm
    
    def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
        """The actual node function that will be called by LangGraph."""
        try:
            logger.info(f"{node_name}: Processing state with chain")
            
            # Invoke the chain with the state
            result = chain.invoke(state)
            
            # Extract the AI message
            if hasattr(result, 'content'):
                ai_message = AIMessage(content=result.content)
            else:
                ai_message = result
            
            # Return updated state
            updated_state = {"messages": [ai_message]}
            
            # Preserve other state variables
            for key, value in state.items():
                if key not in updated_state:
                    updated_state[key] = value
            
            logger.info(f"{node_name}: Chain processing completed successfully")
            return updated_state
            
        except Exception as e:
            logger.error(f"{node_name}: Error in chain processing: {e}")
            # Return an error message
            error_message = AIMessage(content=f"Error: {str(e)}")
            return {"messages": [error_message]}
    
    return node_function


def create_understudy_tool_node(
    endpoint_id: str,
    tools: list,
    base_url: str = "http://localhost:8000",
    api_key: str = None,
    node_name: str = "understudy_tool_node",
    **llm_kwargs
) -> Callable:
    """
    Create a LangGraph node that can use tools with Understudy.
    
    Args:
        endpoint_id: The Understudy endpoint ID to use
        tools: List of tools to bind to the LLM
        base_url: Base URL of the Understudy API
        api_key: Optional API key for authentication
        node_name: Name for this node (for logging)
        **llm_kwargs: Additional arguments for the LLM
    
    Returns:
        A function that can be used as a LangGraph node
    """
    # Create the Understudy chat model
    llm = UnderstudyChatModel(
        endpoint_id=endpoint_id,
        base_url=base_url,
        api_key=api_key,
        **llm_kwargs
    )
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
        """The actual node function that will be called by LangGraph."""
        try:
            messages = state.get("messages", [])
            if not messages:
                logger.warning(f"{node_name}: No messages in state")
                return {"messages": []}
            
            logger.info(f"{node_name}: Processing {len(messages)} messages with tools")
            
            # Invoke the LLM with tools
            result = llm_with_tools.invoke(messages)
            
            # Extract the AI message
            if hasattr(result, 'content'):
                ai_message = AIMessage(content=result.content)
            else:
                ai_message = result
            
            # Check for tool calls
            tool_calls = getattr(result, 'tool_calls', [])
            if tool_calls:
                logger.info(f"{node_name}: Generated {len(tool_calls)} tool calls")
                ai_message.tool_calls = tool_calls
            
            # Return updated state
            updated_state = {"messages": [ai_message]}
            
            # Preserve other state variables
            for key, value in state.items():
                if key not in updated_state:
                    updated_state[key] = value
            
            logger.info(f"{node_name}: Tool node processing completed successfully")
            return updated_state
            
        except Exception as e:
            logger.error(f"{node_name}: Error in tool node processing: {e}")
            # Return an error message
            error_message = AIMessage(content=f"Error: {str(e)}")
            return {"messages": [error_message]}
    
    return node_function


# Utility functions for common LangGraph patterns

def should_continue_to_tools(state: Dict[str, Any]) -> str:
    """
    Routing function to determine if we should continue to tools.
    Use this as a conditional edge in LangGraph.
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


def create_tool_execution_node(tools_dict: Dict[str, Callable]) -> Callable:
    """
    Create a node that executes tools based on tool calls in messages.
    
    Args:
        tools_dict: Dictionary mapping tool names to callable functions
        
    Returns:
        A function that can be used as a LangGraph node
    """
    def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools based on tool calls in the last message."""
        from langchain_core.messages import ToolMessage
        
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
        
        last_message = messages[-1]
        tool_calls = getattr(last_message, 'tool_calls', [])
        
        if not tool_calls:
            return {"messages": []}
        
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")
            
            if tool_name in tools_dict:
                try:
                    result = tools_dict[tool_name](**tool_args)
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id
                        )
                    )
                except Exception as e:
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error executing {tool_name}: {str(e)}",
                            tool_call_id=tool_id
                        )
                    )
            else:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool {tool_name} not found",
                        tool_call_id=tool_id
                    )
                )
        
        return {"messages": tool_messages}
    
    return tool_node