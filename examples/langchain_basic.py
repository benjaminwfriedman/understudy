#!/usr/bin/env python3
"""
Basic LangChain integration example with Understudy.

This example demonstrates how to:
1. Create an Understudy endpoint
2. Use it with LangChain LLM wrapper
3. Build a simple chain
4. Monitor training progress
"""

import asyncio
import time
from understudy import Understudy, UnderstudyLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


async def main():
    # Initialize Understudy client
    print("🚀 Initializing Understudy client...")
    client = Understudy(base_url="http://localhost:8000")
    
    # Check health
    try:
        health = client.health_check()
        print(f"✅ Understudy API is {health['status']}")
    except Exception as e:
        print(f"❌ Failed to connect to Understudy API: {e}")
        return
    
    # Create endpoint
    print("\n📡 Creating Understudy endpoint...")
    endpoint = client.create_endpoint(
        name="customer-support-demo-2",
        description="Demo endpoint for customer support automation",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        training_batch_size=50,
        similarity_threshold=0.93,
        auto_switchover=False  # Manual activation for demo
    )
    
    endpoint_id = endpoint["id"]
    print(f"✅ Created endpoint: {endpoint['name']} (ID: {endpoint_id})")
    
    # Create LangChain LLM wrapper
    print("\n🦜 Setting up LangChain integration...")
    llm = UnderstudyLLM(
        endpoint_id=endpoint_id,
        base_url="http://localhost:8000",
        max_tokens=200,
        temperature=0.7
    )
    
    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""You are a helpful customer support agent. 
        
Context: {context}

Customer Question: {question}

Please provide a helpful, professional response:"""
    )
    
    # Create LangChain chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Example customer support scenarios
    scenarios = [
        {
            "context": "Our product is a cloud-based project management tool",
            "question": "How do I reset my password?"
        },
        {
            "context": "We offer 24/7 technical support via chat and email",
            "question": "What are your support hours?"
        },
        {
            "context": "Free trial includes access to all features for 14 days",
            "question": "What's included in the free trial?"
        },
        {
            "context": "Pro plan costs $29/month, Team plan costs $49/month",
            "question": "What are your pricing plans?"
        },
        {
            "context": "We integrate with Slack, Teams, Jira, and GitHub",
            "question": "Do you integrate with Slack?"
        },
        {
            "context": "Data is encrypted at rest and in transit with AES-256",
            "question": "How secure is my data?"
        },
        {
            "context": "You can export data in CSV, JSON, or PDF format",
            "question": "Can I export my project data?"
        },
        {
            "context": "Mobile apps available for iOS and Android",
            "question": "Do you have a mobile app?"
        }
    ]
    
    print(f"\n💬 Running {len(scenarios)} customer support scenarios...")
    print("=" * 80)
    
    # Run scenarios to generate training data
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📞 Scenario {i}/{len(scenarios)}")
        print(f"Question: {scenario['question']}")
        
        # Generate response using the chain
        response = chain.run(**scenario)
        print(f"Response: {response}")
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    print("\n" + "=" * 80)
    
    # Check endpoint status
    print("\n📊 Checking endpoint status...")
    endpoint_status = client.get_endpoint(endpoint_id)
    print(f"Status: {endpoint_status['status']}")
    
    # Get metrics
    try:
        metrics = client.get_metrics(endpoint_id)
        print(f"Total inferences: {metrics['total_inferences']}")
        print(f"LLM inferences: {metrics['llm_inferences']}")
        print(f"SLM inferences: {metrics['slm_inferences']}")
        print(f"Cost saved: ${metrics['total_cost_saved']:.4f}")
    except Exception as e:
        print(f"Metrics not yet available: {e}")
    
    # Start training if we have enough data
    print("\n🏋️ Starting training...")
    try:
        training_result = client.start_training(endpoint_id)
        print(f"Training status: {training_result['status']}")
        print(f"Message: {training_result['message']}")
        
        if training_result['status'] == 'scheduled':
            print("\n⏳ Training started! You can monitor progress in the dashboard.")
            print("Once training completes and similarity threshold is met,")
            print("you can activate the SLM for cost savings.")
            
            # In a real scenario, you would poll for completion
            print("\n💡 To check training progress:")
            print(f"   runs = client.get_training_runs('{endpoint_id}')")
            print("   print(runs[0]['status'])  # 'running', 'completed', or 'failed'")
            
    except Exception as e:
        print(f"Training failed to start: {e}")
    
    # Show carbon impact
    try:
        carbon = client.get_carbon_summary(endpoint_id)
        print(f"\n🌱 Carbon Impact:")
        print(f"Training emissions: {carbon['total_training_emissions_kg']:.6f} kg CO₂")
        print(f"Avoided emissions: {carbon['avoided_emissions_kg']:.6f} kg CO₂")
        print(f"Net savings: {carbon['net_emissions_saved_kg']:.6f} kg CO₂")
        
        if carbon['carbon_payback_achieved']:
            print("✅ Carbon payback achieved!")
        else:
            print("⏳ Carbon payback in progress...")
    except Exception as e:
        print(f"Carbon data not yet available: {e}")
    
    print(f"\n🎉 Demo complete! Visit the dashboard to see your endpoint:")
    print(f"   http://localhost:3000/endpoints/{endpoint_id}")


if __name__ == "__main__":
    asyncio.run(main())