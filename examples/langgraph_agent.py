#!/usr/bin/env python3
"""
LangGraph multi-agent system with Understudy.

This example demonstrates how to:
1. Build a multi-agent system using LangGraph
2. Use Understudy for LLM components
3. Create tool-using agents
4. Train SLMs from agent interactions
"""

import asyncio
import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from understudy import Understudy, create_understudy_node


# Define the graph state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The messages in the conversation"]
    next_agent: str
    task_complete: bool


# Define tools that agents can use
@tool
def calculate_savings(current_cost: float, reduction_factor: float) -> str:
    """Calculate cost savings from using SLM instead of LLM."""
    savings = current_cost * (1 - 1/reduction_factor)
    return f"Monthly savings: ${savings:.2f} (${current_cost:.2f} -> ${current_cost/reduction_factor:.2f})"


@tool
def estimate_training_time(num_examples: int, model_size: str = "1B") -> str:
    """Estimate SLM training time based on number of examples."""
    if model_size == "1B":
        minutes_per_100 = 5
    else:
        minutes_per_100 = 10
    
    estimated_minutes = (num_examples / 100) * minutes_per_100
    hours = estimated_minutes / 60
    
    if hours < 1:
        return f"Estimated training time: {estimated_minutes:.0f} minutes"
    else:
        return f"Estimated training time: {hours:.1f} hours"


@tool
def carbon_calculator(energy_kwh: float, country: str = "USA") -> str:
    """Calculate carbon emissions from energy usage."""
    # Simplified carbon factors (kg CO2 per kWh)
    factors = {
        "USA": 0.4,
        "EU": 0.3,
        "CANADA": 0.15,
        "BRAZIL": 0.08
    }
    
    factor = factors.get(country.upper(), 0.4)
    emissions = energy_kwh * factor
    
    return f"Estimated CO2 emissions: {emissions:.4f} kg ({energy_kwh} kWh × {factor} kg/kWh)"


def create_router_node():
    """Route to the next appropriate agent."""
    def router(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not messages:
            return {"next_agent": "planner"}
        
        last_message = messages[-1]
        content = last_message.content.lower()
        
        if "calculate" in content and "cost" in content:
            return {"next_agent": "cost_analyst"}
        elif "training" in content or "time" in content:
            return {"next_agent": "training_specialist"}
        elif "carbon" in content or "emission" in content:
            return {"next_agent": "carbon_analyst"}
        else:
            return {"next_agent": "planner"}
    
    return router


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end."""
    if state.get("task_complete", False):
        return "end"
    
    messages = state.get("messages", [])
    if not messages:
        return "continue"
    
    last_message = messages[-1]
    if "task complete" in last_message.content.lower() or "finished" in last_message.content.lower():
        return "end"
    
    return "continue"


async def main():
    print("🤖 Multi-Agent LangGraph System with Understudy")
    print("=" * 60)
    
    # Initialize Understudy client
    client = Understudy(base_url="http://localhost:8000")
    
    # Create endpoints for different agent types
    print("\n🚀 Setting up agent endpoints...")
    
    endpoints = {}
    
    # Planner agent
    planner_endpoint = client.create_endpoint(
        name="planner-agent",
        description="Planning and coordination agent",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        training_batch_size=30
    )
    endpoints["planner"] = planner_endpoint["id"]
    
    # Cost analyst agent  
    cost_endpoint = client.create_endpoint(
        name="cost-analyst-agent",
        description="Cost analysis and savings calculation agent",
        llm_provider="openai", 
        llm_model="gpt-3.5-turbo",
        training_batch_size=20
    )
    endpoints["cost_analyst"] = cost_endpoint["id"]
    
    # Training specialist agent
    training_endpoint = client.create_endpoint(
        name="training-specialist-agent",
        description="ML training and optimization specialist",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo", 
        training_batch_size=20
    )
    endpoints["training_specialist"] = training_endpoint["id"]
    
    # Carbon analyst agent
    carbon_endpoint = client.create_endpoint(
        name="carbon-analyst-agent",
        description="Carbon footprint and sustainability analyst",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        training_batch_size=20
    )
    endpoints["carbon_analyst"] = carbon_endpoint["id"]
    
    print(f"✅ Created {len(endpoints)} agent endpoints")
    
    # Create agent nodes with specialized prompts
    tools = [calculate_savings, estimate_training_time, carbon_calculator]
    
    planner_node = create_understudy_node(
        endpoint_id=endpoints["planner"],
        base_url="http://localhost:8000",
        node_name="planner",
        temperature=0.7
    )
    
    cost_analyst_node = create_understudy_node(
        endpoint_id=endpoints["cost_analyst"],
        base_url="http://localhost:8000", 
        node_name="cost_analyst",
        temperature=0.3
    )
    
    training_specialist_node = create_understudy_node(
        endpoint_id=endpoints["training_specialist"],
        base_url="http://localhost:8000",
        node_name="training_specialist", 
        temperature=0.4
    )
    
    carbon_analyst_node = create_understudy_node(
        endpoint_id=endpoints["carbon_analyst"],
        base_url="http://localhost:8000",
        node_name="carbon_analyst",
        temperature=0.3
    )
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", create_router_node())
    workflow.add_node("planner", planner_node)
    workflow.add_node("cost_analyst", cost_analyst_node)
    workflow.add_node("training_specialist", training_specialist_node)
    workflow.add_node("carbon_analyst", carbon_analyst_node)
    
    # Add edges
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        lambda x: x["next_agent"],
        {
            "planner": "planner", 
            "cost_analyst": "cost_analyst",
            "training_specialist": "training_specialist",
            "carbon_analyst": "carbon_analyst"
        }
    )
    
    # All agents go back to router
    workflow.add_edge("planner", "router")
    workflow.add_edge("cost_analyst", "router") 
    workflow.add_edge("training_specialist", "router")
    workflow.add_edge("carbon_analyst", "router")
    
    # Conditional ending
    workflow.add_conditional_edges(
        "router",
        should_continue,
        {
            "continue": "router",
            "end": END
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    # Test scenarios
    scenarios = [
        {
            "query": "I'm spending $500/month on OpenAI API calls for my customer support bot. How much could I save with Understudy if it achieves 50x cost reduction?",
            "description": "Cost analysis scenario"
        },
        {
            "query": "I have 1000 customer support conversations collected. How long would it take to train an SLM, and what's the carbon footprint?",
            "description": "Training and carbon analysis scenario"
        },
        {
            "query": "My current LLM usage consumes about 2.5 kWh per month. What would be the carbon emissions if I'm based in Canada?",
            "description": "Carbon footprint analysis"
        }
    ]
    
    print("\n🎭 Running multi-agent scenarios...")
    print("=" * 80)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎬 Scenario {i}: {scenario['description']}")
        print(f"Query: {scenario['query']}")
        print("-" * 60)
        
        # Run the agent system
        initial_state = {
            "messages": [HumanMessage(content=scenario['query'])],
            "next_agent": "planner",
            "task_complete": False
        }
        
        try:
            # Stream the execution
            step_count = 0
            async for output in app.astream(initial_state):
                step_count += 1
                if step_count > 10:  # Prevent infinite loops
                    break
                    
                for key, value in output.items():
                    if "messages" in value and value["messages"]:
                        last_msg = value["messages"][-1]
                        if hasattr(last_msg, 'content'):
                            print(f"🤖 {key}: {last_msg.content[:200]}...")
            
        except Exception as e:
            print(f"❌ Error in scenario {i}: {e}")
        
        print("\n" + "=" * 60)
    
    # Check training status for all agents
    print("\n📊 Agent Training Status:")
    for agent_name, endpoint_id in endpoints.items():
        try:
            metrics = client.get_metrics(endpoint_id)
            print(f"\n🤖 {agent_name}:")
            print(f"   Interactions: {metrics['total_inferences']}")
            print(f"   LLM calls: {metrics['llm_inferences']}")
            
            if metrics['llm_inferences'] >= 5:
                print(f"   ✅ Ready for training")
                # Start training
                result = client.start_training(endpoint_id)
                print(f"   Training: {result['message']}")
            else:
                print(f"   ⏳ Need {5 - metrics['llm_inferences']} more interactions")
                
        except Exception as e:
            print(f"   ⚠️  Metrics not available yet")
    
    # Show overall carbon impact
    print(f"\n🌱 Overall Environmental Impact:")
    total_training_emissions = 0
    total_avoided_emissions = 0
    
    for agent_name, endpoint_id in endpoints.items():
        try:
            carbon = client.get_carbon_summary(endpoint_id)
            total_training_emissions += carbon['total_training_emissions_kg']
            total_avoided_emissions += carbon['avoided_emissions_kg']
        except:
            pass
    
    net_savings = total_avoided_emissions - total_training_emissions
    print(f"Total training emissions: {total_training_emissions:.6f} kg CO₂")
    print(f"Total avoided emissions: {total_avoided_emissions:.6f} kg CO₂")
    print(f"Net environmental benefit: {net_savings:.6f} kg CO₂")
    
    print(f"\n🎉 Multi-agent demo complete!")
    print(f"📊 Monitor all agents: http://localhost:3000/dashboard")
    
    print(f"\n💡 Next steps:")
    print("   1. Each agent will train its own specialized SLM")
    print("   2. Activate SLMs when similarity thresholds are reached")
    print("   3. Enjoy cost savings across your entire agent fleet!")


if __name__ == "__main__":
    asyncio.run(main())