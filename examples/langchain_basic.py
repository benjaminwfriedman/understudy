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
    
    endpoint_id = "1c9ce60c-db11-4134-b8da-578a1efe3aa9"
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
    
    # Extended customer support scenarios - 100+ for extensive testing
    base_scenarios = [
        # Authentication & Security (15 scenarios)
        {"context": "Our product is a cloud-based project management tool", "question": "How do I reset my password?"},
        {"context": "Two-factor authentication is required for enterprise accounts", "question": "How do I enable 2FA?"},
        {"context": "Sessions expire after 24 hours of inactivity", "question": "Why was I logged out?"},
        {"context": "We support SSO with Google, Microsoft, and SAML", "question": "Can I use SSO to login?"},
        {"context": "Account lockout occurs after 5 failed login attempts", "question": "My account is locked, what do I do?"},
        {"context": "Password must be 12+ chars with special characters", "question": "What are the password requirements?"},
        {"context": "API keys can be generated in account settings", "question": "How do I create an API key?"},
        {"context": "Data is encrypted at rest and in transit with AES-256", "question": "How secure is my data?"},
        {"context": "We are SOC 2 Type II compliant", "question": "Do you have security certifications?"},
        {"context": "GDPR compliance includes right to deletion", "question": "Can I delete all my data?"},
        {"context": "We log all user activities for audit trails", "question": "Do you track what I do in the app?"},
        {"context": "IP allowlisting is available for enterprise plans", "question": "Can I restrict access by IP?"},
        {"context": "Backup codes are provided when enabling 2FA", "question": "What if I lose my 2FA device?"},
        {"context": "Account recovery requires email verification", "question": "How do I recover a lost account?"},
        {"context": "We support hardware security keys like YubiKey", "question": "Can I use a hardware key?"},
        
        # Pricing & Plans (20 scenarios)
        {"context": "Free trial includes access to all features for 14 days", "question": "What's included in the free trial?"},
        {"context": "Pro plan costs $29/month, Team plan costs $49/month", "question": "What are your pricing plans?"},
        {"context": "Annual billing offers 20% discount", "question": "Do you offer yearly discounts?"},
        {"context": "Enterprise plans start at $99/month for 50+ users", "question": "What is enterprise pricing?"},
        {"context": "Free plan supports up to 3 projects and 5 users", "question": "What are free plan limits?"},
        {"context": "Billing is prorated when upgrading mid-cycle", "question": "How does mid-cycle upgrading work?"},
        {"context": "Cancellation takes effect at end of billing period", "question": "When does cancellation take effect?"},
        {"context": "Refunds are available within 30 days", "question": "Can I get a refund?"},
        {"context": "Team plan includes priority support", "question": "What support do I get with Team plan?"},
        {"context": "Storage limits are 100GB per user on Pro plan", "question": "How much storage do I get?"},
        {"context": "Additional storage costs $5 per 10GB per month", "question": "How much does extra storage cost?"},
        {"context": "Custom integrations available on Enterprise plan", "question": "Can you build custom integrations?"},
        {"context": "Non-profit organizations get 50% discount", "question": "Do you offer non-profit discounts?"},
        {"context": "Student discounts are 80% off Pro plan", "question": "Do you have student pricing?"},
        {"context": "Payment methods include cards, ACH, and wire transfer", "question": "What payment methods do you accept?"},
        {"context": "Invoice billing available for Enterprise customers", "question": "Can I pay by invoice?"},
        {"context": "VAT is added for EU customers", "question": "Do you charge VAT?"},
        {"context": "Currency options include USD, EUR, GBP", "question": "Can I pay in my local currency?"},
        {"context": "Volume discounts start at 100 users", "question": "Do you offer volume discounts?"},
        {"context": "Price increases are communicated 60 days in advance", "question": "Will my price change?"},
        
        # Features & Functionality (25 scenarios)
        {"context": "We integrate with Slack, Teams, Jira, and GitHub", "question": "Do you integrate with Slack?"},
        {"context": "Gantt charts are available on Pro plans and above", "question": "Do you have Gantt charts?"},
        {"context": "Time tracking includes manual and automatic options", "question": "How does time tracking work?"},
        {"context": "Custom fields can be added to projects and tasks", "question": "Can I add custom fields?"},
        {"context": "Automated workflows trigger based on task status", "question": "Do you have workflow automation?"},
        {"context": "Reports include burndown, velocity, and resource allocation", "question": "What reports are available?"},
        {"context": "Real-time collaboration allows simultaneous editing", "question": "Can multiple people work simultaneously?"},
        {"context": "Version history keeps 30 days of changes", "question": "Can I see previous versions?"},
        {"context": "Templates are available for common project types", "question": "Do you have project templates?"},
        {"context": "Dependencies can be set between tasks", "question": "Can tasks depend on other tasks?"},
        {"context": "Calendar view shows deadlines and milestones", "question": "Do you have a calendar view?"},
        {"context": "Notifications can be customized per user", "question": "Can I control notifications?"},
        {"context": "Bulk operations allow mass task updates", "question": "Can I update multiple tasks at once?"},
        {"context": "Search includes full-text across all content", "question": "How powerful is your search?"},
        {"context": "Kanban boards support custom columns", "question": "Can I customize Kanban boards?"},
        {"context": "Guest access allows external collaboration", "question": "Can I invite external users?"},
        {"context": "File attachments support up to 100MB per file", "question": "What's the file size limit?"},
        {"context": "Comments support @mentions and threading", "question": "How do comments work?"},
        {"context": "Project portfolios group related projects", "question": "Can I organize projects into portfolios?"},
        {"context": "Resource management shows team capacity", "question": "Do you track team capacity?"},
        {"context": "Milestone tracking shows progress against goals", "question": "How do milestones work?"},
        {"context": "Risk management includes issue tracking", "question": "Can I track project risks?"},
        {"context": "Budget tracking compares actual vs planned costs", "question": "Do you support budget tracking?"},
        {"context": "Goal setting links to OKRs and KPIs", "question": "Can I set and track goals?"},
        {"context": "Dashboard widgets are customizable", "question": "Can I customize my dashboard?"},
        
        # Data & Export (15 scenarios)
        {"context": "You can export data in CSV, JSON, or PDF format", "question": "Can I export my project data?"},
        {"context": "API access allows custom integrations", "question": "Do you have an API?"},
        {"context": "Data import supports CSV and Excel files", "question": "How do I import existing data?"},
        {"context": "Backup exports run automatically weekly", "question": "Do you backup my data?"},
        {"context": "Data retention policy keeps deleted items for 30 days", "question": "What happens to deleted data?"},
        {"context": "Export includes all project history and attachments", "question": "What's included in exports?"},
        {"context": "Migration tools help move from other platforms", "question": "Can you help migrate from other tools?"},
        {"context": "Data portability ensures you own your data", "question": "Do I own my data?"},
        {"context": "Archive feature preserves completed projects", "question": "Can I archive old projects?"},
        {"context": "Bulk delete operations require confirmation", "question": "How do I delete multiple items?"},
        {"context": "Restore functionality recovers deleted items", "question": "Can I restore deleted content?"},
        {"context": "Data sync ensures consistency across devices", "question": "How do you keep data in sync?"},
        {"context": "Offline access caches recent data locally", "question": "Can I work offline?"},
        {"context": "Export scheduling allows automated backups", "question": "Can I schedule automatic exports?"},
        {"context": "Data anonymization removes PII on request", "question": "Can you anonymize my data?"},
        
        # Support & Training (10 scenarios)  
        {"context": "We offer 24/7 technical support via chat and email", "question": "What are your support hours?"},
        {"context": "Phone support is available for Enterprise customers", "question": "Do you offer phone support?"},
        {"context": "Knowledge base includes tutorials and FAQs", "question": "Do you have documentation?"},
        {"context": "Video tutorials cover all major features", "question": "Are there video guides?"},
        {"context": "Onboarding sessions are included with Team plans", "question": "Do you help with setup?"},
        {"context": "Training webinars run monthly for all users", "question": "Do you offer training?"},
        {"context": "Community forum allows user discussions", "question": "Is there a user community?"},
        {"context": "Feature requests are tracked and prioritized", "question": "How do I request new features?"},
        {"context": "Status page shows real-time system health", "question": "Where can I check system status?"},
        {"context": "Response times are 4 hours for email, 1 hour for chat", "question": "How fast do you respond?"},
        
        # Mobile & Access (10 scenarios)
        {"context": "Mobile apps available for iOS and Android", "question": "Do you have a mobile app?"},
        {"context": "Web app works on all modern browsers", "question": "What browsers do you support?"},
        {"context": "PWA support enables offline mobile access", "question": "Can I install as an app?"},
        {"context": "Push notifications keep you updated on mobile", "question": "Do you send mobile notifications?"},
        {"context": "Touch interface optimized for tablet use", "question": "Does it work well on tablets?"},
        {"context": "Responsive design adapts to any screen size", "question": "Does it work on small screens?"},
        {"context": "Dark mode available in user preferences", "question": "Do you support dark mode?"},
        {"context": "Accessibility features support screen readers", "question": "Is your app accessible?"},
        {"context": "Multi-language support includes 12 languages", "question": "Is it available in other languages?"},
        {"context": "Keyboard shortcuts speed up common actions", "question": "Are there keyboard shortcuts?"},
        
        # Company & Product (5 scenarios)
        {"context": "App was first created 2 weeks ago", "question": "How long ago was the app created"},
        {"context": "We have consumer and enterprise subscriptions", "question": "Do you have a consumer subscription?"},
        {"context": "Company founded in 2018 with 50+ employees", "question": "How old is your company?"},
        {"context": "Servers hosted in US, EU, and Asia regions", "question": "Where are your servers located?"},
        {"context": "99.9% uptime SLA with status page monitoring", "question": "What's your uptime guarantee?"}
    ]
    
    # Generate additional variations with different parameters
    scenarios = []
    
    # Add base scenarios
    scenarios.extend(base_scenarios)
    
    # Add stress test variations with different contexts
    stress_contexts = [
        "Emergency response system for first responders",
        "Medical record management for hospitals", 
        "Financial trading platform for investment firms",
        "Educational platform for online learning",
        "E-commerce system for retail operations"
    ]
    
    stress_questions = [
        "How do I get started?", "What are the main features?", "Is this secure?",
        "How much does it cost?", "Do you offer training?", "Can I integrate with my existing tools?",
        "What's your uptime?", "How do I contact support?", "Is there a mobile app?",
        "Can I export my data?"
    ]
    
    for context in stress_contexts:
        for question in stress_questions:
            scenarios.append({"context": context, "question": question})
    
    # Add edge case scenarios  
    edge_scenarios = [
        {"context": "Very long context " + "that repeats " * 50 + "to test handling", "question": "Short question?"},
        {"context": "Short context", "question": "Very long question " + "with lots of detail " * 20 + "to test limits?"},
        {"context": "", "question": "What happens with empty context?"},
        {"context": "Normal context", "question": ""},
        {"context": "Context with émojis 🚀 and spëcial çharacters", "question": "Does this work with ünïcödé?"},
        {"context": "Context\nwith\nmultiple\nlines", "question": "How about newlines?"},
        {"context": "Context with \"quotes\" and 'apostrophes'", "question": "What about punctuation marks?"}
    ]
    
    scenarios.extend(edge_scenarios)
    
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
    
    # # Start training if we have enough data
    # print("\n🏋️ Starting training...")
    # try:
    #     training_result = client.start_training(endpoint_id)
    #     print(f"Training status: {training_result['status']}")
    #     print(f"Message: {training_result['message']}")
        
    #     if training_result['status'] == 'scheduled':
    #         print("\n⏳ Training started! You can monitor progress in the dashboard.")
    #         print("Once training completes and similarity threshold is met,")
    #         print("you can activate the SLM for cost savings.")
            
    #         # In a real scenario, you would poll for completion
    #         print("\n💡 To check training progress:")
    #         print(f"   runs = client.get_training_runs('{endpoint_id}')")
    #         print("   print(runs[0]['status'])  # 'running', 'completed', or 'failed'")
            
    # except Exception as e:
    #     print(f"Training failed to start: {e}")
    
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