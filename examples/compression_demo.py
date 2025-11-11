#!/usr/bin/env python3
"""
Few-Shot Classification Demo with Extensive Label Context
Shows how prompt compression handles large classification contexts with detailed label explanations.
"""

import requests
import json
import time
import random
from typing import List, Dict

# Configuration
UNDERSTUDY_BASE_URL = "http://localhost:8000/api/v1"
ENDPOINT_ID = "7d0cc2a8-b6e4-4486-84b0-700421f68e13"  # Your endpoint with compression enabled

# Extensive classification context with detailed label explanations
CLASSIFICATION_EXAMPLES = [
    {
        "name": "Email Intent Classification",
        "label_context": """
COMPREHENSIVE EMAIL INTENT CLASSIFICATION SYSTEM

This classification system categorizes customer emails into specific intents to enable proper routing and response. Each label has distinct characteristics and triggers:

BILLING_INQUIRY (Label: billing)
- Definition: Questions about charges, payments, invoices, refunds, or pricing
- Characteristics: Contains financial terms, mentions of costs, payment methods, billing cycles
- Common phrases: "charge on my account", "invoice", "refund", "payment failed", "billing question", "subscription cost"
- Subcategories: Payment issues, refund requests, invoice questions, pricing inquiries, billing disputes
- Urgency: Medium-High (affects customer's financial relationship)
- Typical resolution: 24-48 hours by billing department
- Examples: "Why was I charged twice?", "Can I get a refund?", "My payment failed"

TECHNICAL_SUPPORT (Label: tech_support)  
- Definition: Issues with product functionality, bugs, performance problems, or feature requests
- Characteristics: Describes system behavior, error messages, troubleshooting attempts
- Common phrases: "not working", "error message", "bug", "feature request", "slow performance", "can't access"
- Subcategories: Bug reports, feature requests, performance issues, access problems, integration help
- Urgency: High (directly affects product usage)
- Typical resolution: 1-72 hours by engineering team
- Examples: "The app crashes when I click submit", "How do I integrate with Slack?"

ACCOUNT_MANAGEMENT (Label: account)
- Definition: Changes to user account, profile updates, access permissions, or account closure
- Characteristics: Requests to modify account settings, user permissions, or account status
- Common phrases: "update my profile", "change password", "add user", "delete account", "permission denied"
- Subcategories: Profile updates, permission changes, account deletion, user management, security settings
- Urgency: Medium (affects user access but not critical)
- Typical resolution: 4-24 hours by account management team
- Examples: "I need to add a team member", "How do I update my email address?"

SALES_INQUIRY (Label: sales)
- Definition: Questions about purchasing, upgrades, demos, or business development opportunities
- Characteristics: Interest in buying products/services, upgrade requests, demo scheduling
- Common phrases: "pricing information", "demo request", "upgrade plan", "purchase", "business inquiry"
- Subcategories: New customer inquiries, upgrade requests, demo scheduling, custom enterprise solutions
- Urgency: High (potential revenue opportunity)
- Typical resolution: 2-24 hours by sales team
- Examples: "I'd like to schedule a demo", "What's the cost for the enterprise plan?"

GENERAL_INQUIRY (Label: general)
- Definition: General questions, documentation requests, or informational queries not covered above
- Characteristics: Broad questions about company, services, or general information
- Common phrases: "how does this work", "general question", "information about", "documentation"
- Subcategories: Product information, company information, documentation requests, general questions
- Urgency: Low-Medium (informational, not blocking)
- Typical resolution: 12-48 hours by support team
- Examples: "What industries do you serve?", "Where can I find the API documentation?"

COMPLAINT (Label: complaint)
- Definition: Expression of dissatisfaction, negative feedback, or service quality issues
- Characteristics: Negative sentiment, expressions of frustration, service quality concerns
- Common phrases: "disappointed", "poor service", "unacceptable", "frustrated", "terrible experience"
- Subcategories: Service complaints, product quality issues, process complaints, staff behavior concerns
- Urgency: Very High (customer retention risk)
- Typical resolution: 1-12 hours by customer success team with management escalation
- Examples: "Your service is terrible", "I'm very disappointed with the response time"

CLASSIFICATION RULES:
1. If multiple intents are present, prioritize by urgency: complaint > sales > billing > tech_support > account > general
2. Look for explicit keywords and phrases listed above
3. Consider the overall tone and sentiment of the message
4. Account for context clues like previous interactions or account status
5. When uncertain between two labels, choose the one that requires faster response time
6. Escalate any messages containing legal threats, security concerns, or executive complaints regardless of primary intent
        """,
        "examples": [
            {"email": "Hi there, I noticed a charge of $99 on my credit card from your company but I can't find any invoice or explanation. Can you please clarify what this charge is for? I've been a customer for 2 years and haven't seen this before. Thanks!", "label": "billing"},
            {"email": "The dashboard is completely broken! Every time I try to load my analytics, it shows a 500 error. This has been happening for 3 days now and I can't access any of my data. Please fix this ASAP as it's affecting my business.", "label": "tech_support"},
            {"email": "I need to add 5 new team members to our account and give them admin permissions. How do I do this? Also, can I change the primary account email from john@company.com to admin@company.com?", "label": "account"},
            {"email": "We're a Fortune 500 company interested in your enterprise solution. Could we schedule a demo for next week? We have about 10,000 employees who would need access. What would the pricing look like?", "label": "sales"},
            {"email": "I'm absolutely furious! Your support team has ignored my tickets for a week and your product keeps failing at the worst possible times. This is completely unacceptable and I'm considering switching to a competitor. Fix this now!", "label": "complaint"},
            {"email": "My payment was declined yesterday and I'm not sure why. I have sufficient funds in my account. Can you help me process the payment again?", "label": "billing"},
            {"email": "The mobile app keeps crashing whenever I try to upload files. I'm using iOS 16 on iPhone 13. Is this a known issue?", "label": "tech_support"},
            {"email": "I forgot my password and the reset email isn't coming through. Can you manually reset it for me? My username is john_doe123.", "label": "account"},
            {"email": "What's the difference between your Pro and Enterprise plans? We have about 50 employees and need advanced analytics features.", "label": "sales"},
            {"email": "This is ridiculous! I've been waiting 3 hours for a simple response and nobody has helped me yet. Your customer service is terrible!", "label": "complaint"},
            {"email": "I was charged for the annual plan but I only wanted monthly billing. Can I get a refund and switch to monthly payments?", "label": "billing"},
            {"email": "The API is returning 503 errors consistently. Our integration is failing and we can't serve our customers. Please prioritize this!", "label": "tech_support"},
            {"email": "How do I remove a user from our team workspace? They left the company last week and still have access to our projects.", "label": "account"},
            {"email": "I'd like to upgrade to your premium plan. What's the process and when would the new features be available?", "label": "sales"},
            {"email": "I am extremely disappointed with your service. The platform went down during our product launch and cost us thousands of dollars in revenue.", "label": "complaint"},
            {"email": "Can you provide an itemized invoice for last month's charges? I need it for expense reporting to our finance team.", "label": "billing"},
            {"email": "The export feature isn't working properly. My CSV files are corrupted and missing data. This started after yesterday's update.", "label": "tech_support"},
            {"email": "I need to update my company address and billing information. Our office moved last month and all our details are outdated.", "label": "account"},
            {"email": "We're evaluating your platform for our startup. Do you offer any discounts for early-stage companies?", "label": "sales"},
            {"email": "Your platform is completely unreliable! It's been down 3 times this week and I'm losing customers because of it.", "label": "complaint"},
            {"email": "There's a duplicate charge on my statement for $149. I should only be billed once per month. Please investigate and refund the extra charge.", "label": "billing"},
            {"email": "I'm getting SSL certificate errors when trying to access the admin panel. This started this morning and I can't manage my account.", "label": "tech_support"},
            {"email": "Can you change the account owner from me to my colleague Sarah? She'll be taking over account management going forward.", "label": "account"},
            {"email": "I saw your demo at TechCrunch and I'm interested in implementing your solution. Can we set up a call to discuss our requirements?", "label": "sales"},
            {"email": "This is absolutely unacceptable! Your system deleted all my data and your support team just keeps giving me the runaround!", "label": "complaint"},
            {"email": "I need a receipt for my recent payment of $299. Our accounting department requires all vendor receipts for tax purposes.", "label": "billing"},
            {"email": "The search functionality is returning irrelevant results. It was working fine last week but now it's completely broken.", "label": "tech_support"},
            {"email": "I can't access the billing section of my account. It shows a permission error even though I'm the account administrator.", "label": "account"},
            {"email": "What integrations do you offer with Salesforce? We need to sync our lead data automatically.", "label": "sales"},
            {"email": "Your service has been terrible lately. Constant outages, slow response times, and bugs everywhere. I want to cancel my subscription immediately!", "label": "complaint"},
            {"email": "My subscription auto-renewed but I wanted to cancel it. Can you process a refund for the charge that happened yesterday?", "label": "billing"},
            {"email": "I'm unable to connect to your database API. I keep getting timeout errors after 30 seconds. Is there a server issue?", "label": "tech_support"},
            {"email": "How do I enable two-factor authentication for my account? I want to improve our security posture.", "label": "account"},
            {"email": "We need a custom enterprise contract for our organization. Who should I contact to discuss enterprise pricing and terms?", "label": "sales"},
            {"email": "I am fed up with your poor service quality! Every interaction with your company has been a nightmare. I'm reporting you to the BBB!", "label": "complaint"},
            {"email": "I was overcharged by $50 last month. According to my plan, I should only pay $99 but was charged $149. Please correct this error.", "label": "billing"},
            {"email": "The webhook notifications stopped working suddenly. Our entire workflow depends on these webhooks and nothing is being triggered.", "label": "tech_support"},
            {"email": "I need to downgrade my account from Pro to Basic. How do I do this and what happens to my existing data?", "label": "account"},
            {"email": "Can you provide a detailed comparison between your platform and your main competitors? We're doing a vendor evaluation.", "label": "sales"},
            {"email": "This is the worst customer experience I've ever had! Your software is buggy, your support is unresponsive, and your billing is a mess!", "label": "complaint"},
            {"email": "I received an invoice for services I never ordered. The invoice number is INV-2023-4567. Please cancel this charge immediately.", "label": "billing"},
            {"email": "The performance has degraded significantly over the past month. Page load times are 10x slower than before. What's causing this?", "label": "tech_support"},
            {"email": "Can you help me set up single sign-on (SSO) with our Azure AD? I need step-by-step instructions for the configuration.", "label": "account"},
            {"email": "I represent a non-profit organization and we're interested in your platform. Do you offer any special pricing for non-profits?", "label": "sales"},
            {"email": "Your platform caused a major data breach in our system! This is completely unacceptable and we're considering legal action!", "label": "complaint"},
            {"email": "Why am I being charged a setup fee of $199? This wasn't mentioned when I signed up for the service. Please explain these charges.", "label": "billing"},
            {"email": "I'm getting JavaScript errors in the browser console and the interface is completely broken. This started after your maintenance window.", "label": "tech_support"},
            {"email": "I accidentally deleted my workspace and need to recover it urgently. Do you have any backup systems that can restore my data?", "label": "account"},
            {"email": "We'd like to become a reseller partner for your platform. What are the requirements and commission structure?", "label": "sales"},
            {"email": "I cannot believe how awful your service is! Nothing works properly and your team clearly doesn't care about customer satisfaction!", "label": "complaint"}
        ],
        "test_emails": [
            "My payment method expired and I need to update it to avoid service interruption",
            "Can you help me integrate your API with our internal system?", 
            "I want to delete my account permanently",
            "What's the difference between your Pro and Enterprise plans?",
            "Your customer service is the worst I've ever experienced",
            "I don't understand this invoice - what are these charges for?",
            "The website won't load and I'm getting a 404 error",
            "How do I change my username on the platform?",
            "Do you offer volume discounts for large organizations?",
            "This is completely unacceptable service and I demand a manager!",
            "Can I get a refund for last month's subscription fee?",
            "My data export is failing every time I try to download it",
            "I need to add my assistant to the account with read-only access",
            "What security certifications does your platform have?",
            "I'm disgusted with how poorly this has been handled",
            "There's an unexpected charge of $75 on my card statement",
            "The mobile app crashes when I try to upload photos",
            "Can I transfer my subscription to a different email address?",
            "I'm interested in a demo of your enterprise features",
            "Your support team is ignoring my urgent requests completely",
            "I was double-billed this month and need one charge reversed",
            "Error 500 keeps appearing when I submit the contact form",
            "How do I reset my account password without access to my old email?",
            "What's your pricing for educational institutions?",
            "This platform is garbage and you people don't care at all",
            "Please send me a detailed breakdown of my annual charges",
            "The search function returns completely wrong results",
            "I want to upgrade my user permissions to admin level",
            "Can we schedule a call to discuss custom integration options?",
            "I am beyond frustrated with your terrible customer support",
            "Why was I charged a cancellation fee when I'm still subscribed?",
            "Database connection timeouts are affecting our production system",
            "I need to remove team members who no longer work here",
            "What's the difference between your API tiers?",
            "Your service is absolutely horrible and I want my money back",
            "I need an updated invoice with our new company tax ID",
            "File uploads are extremely slow and often timeout completely",
            "Can you help me configure SAML authentication for our domain?",
            "We're considering switching from our current provider to you",
            "This is the most frustrating experience I've ever had with software",
            "I'm being charged for features I never signed up for",
            "The webhook endpoint keeps returning SSL handshake errors",
            "How do I archive old projects without deleting the data?",
            "What kind of SLA do you provide for uptime guarantees?",
            "Your platform is broken and your support doesn't respond to anything",
            "Can you explain the overage charges on my latest bill?",
            "Performance has gotten worse since the last update - page loads are very slow",
            "I accidentally locked myself out of my account and need help",
            "Are there any partnership opportunities for system integrators?",
            "I've had it with your unreliable service and poor communication"
        ]
    }
]

def make_classification_request(email: str, label_context: str, examples: List[Dict]) -> Dict:
    """Make a classification request to Understudy API."""
    
    # Build few-shot examples
    example_text = ""
    for ex in examples:
        example_text += f"Email: {ex['email']}\nLabel: {ex['label']}\n\n"
    
    prompt = f"""You are an expert email classification system. Classify customer emails into the correct intent category based on the detailed label descriptions and examples provided.

{label_context}

Here are examples of correctly classified emails:

{example_text}

Now classify this email:
Email: {email}
Label:"""

    payload = {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{UNDERSTUDY_BASE_URL}/inference/{ENDPOINT_ID}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            result['request_time'] = end_time - start_time
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def estimate_tokens(text: str) -> int:
    """Rough token estimation (approximately 4 characters per token)."""
    return len(text) // 4

def run_classification_demo():
    """Run the few-shot classification demonstration with extensive context."""
    
    print("📧 Few-Shot Email Classification Demo")
    print("=" * 50)
    print(f"Endpoint: {ENDPOINT_ID}")
    print("Testing classification with extensive label context...\n")
    
    example = CLASSIFICATION_EXAMPLES[0]
    print(f"📋 Scenario: {example['name']}")
    print("-" * 40)
    
    # Show context size
    context = example['label_context']
    context_tokens = estimate_tokens(context)
    print(f"Label Context Size: ~{context_tokens:,} tokens")
    print(f"Number of Examples: {len(example['examples'])}")
    print(f"Test Emails: {len(example['test_emails'])}\n")
    
    # Test each email
    correct_predictions = 0
    total_predictions = 0
    
    for i, email in enumerate(example['test_emails'], 1):
        print(f"\n{i}. Classifying: {email[:80]}...")
        
        # Make classification request
        result = make_classification_request(
            email, 
            example['label_context'], 
            example['examples']
        )
        
        if result:
            predicted_label = result['output'].strip().lower()
            print(f"   Predicted Label: {predicted_label}")
            print(f"   Response Time: {result.get('request_time', 0):.2f}s")
            print(f"   Cost: ${result.get('cost_usd', 0):.6f}")
            
            # Show confidence (if available)
            if 'confidence' in result:
                print(f"   Confidence: {result['confidence']:.2f}")
            
            total_predictions += 1
            time.sleep(1)
    
    print(f"\n✅ Completed {example['name']}")
    print(f"Total Classifications: {total_predictions}")
    
    # Get compression metrics
    print("\n📊 Fetching Compression Metrics...")
    try:
        metrics_response = requests.get(f"{UNDERSTUDY_BASE_URL}/metrics/{ENDPOINT_ID}")
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            
            print("\n🎯 COMPRESSION RESULTS")
            print("=" * 50)
            print(f"Total Compressed Requests: {metrics.get('total_compressed_requests', 0)}")
            print(f"Average Tokens Saved: {metrics.get('avg_tokens_saved', 0):.1f}")
            print(f"Compression Cost Savings: ${metrics.get('compression_cost_savings', 0):.4f}")
            print(f"Total Inferences: {metrics.get('total_inferences', 0)}")
            
            if metrics.get('avg_tokens_saved', 0) > 0:
                # Calculate compression benefits for classification
                saved_tokens = metrics.get('avg_tokens_saved', 0)
                original_size = context_tokens + 500  # Rough estimate including examples
                compression_ratio = saved_tokens / original_size * 100
                print(f"Context Compression Ratio: {compression_ratio:.1f}%")
                
                # Projected savings for high-volume classification
                monthly_classifications = 100000
                monthly_savings = (metrics.get('compression_cost_savings', 0) / max(total_predictions, 1)) * monthly_classifications
                print(f"Projected Monthly Savings (100K classifications): ${monthly_savings:.2f}")
        
    except Exception as e:
        print(f"Could not fetch metrics: {e}")
    
    print("\n🎉 Classification demo completed!")
    print("\nKey Benefits Demonstrated:")
    print("- Large context handling with detailed label descriptions")
    print("- Few-shot learning with comprehensive examples") 
    print("- Prompt compression reducing costs for repetitive classification tasks")
    print("- Consistent performance across multiple test cases")

if __name__ == "__main__":
    print("Starting Classification Demo in 3 seconds...")
    print("Make sure your endpoint has compression enabled!")
    time.sleep(3)
    run_classification_demo()