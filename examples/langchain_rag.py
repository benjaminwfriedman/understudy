#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) example with Understudy.

This example demonstrates how to:
1. Build a RAG system using LangChain
2. Use Understudy for the LLM component
3. Process documents and answer questions
4. Train an SLM from the RAG interactions
"""

import asyncio
from pathlib import Path
from understudy import Understudy, UnderstudyLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Sample document content for demo
SAMPLE_DOCS = {
    "product_info.txt": """
Understudy Product Information

Understudy is an AI platform that helps organizations reduce their AI inference costs by automatically training small language models (SLMs) from large language model (LLM) interactions.

Key Features:
- Automatic SLM training from LLM interactions
- LangChain and LangGraph compatibility  
- Real-time carbon emissions tracking with CodeCarbon
- 10-100x cost reduction potential
- Seamless switchover when SLM reaches accuracy threshold
- Built-in metrics and analytics dashboard

Pricing:
- Starter Plan: $29/month - Up to 10,000 inferences
- Professional Plan: $99/month - Up to 100,000 inferences
- Enterprise Plan: Custom pricing for unlimited usage

Support:
- 24/7 chat support for all plans
- Email support with 4-hour response time
- Phone support for Enterprise customers
- Comprehensive documentation and tutorials
""",
    
    "technical_specs.txt": """
Technical Specifications

Architecture:
- Backend: FastAPI with Python 3.11+
- Frontend: React 18 with TypeScript
- Database: SQLite (development) / PostgreSQL (production)
- ML Framework: Hugging Face Transformers + PEFT (LoRA/QLoRA)
- Base Model: Llama 3.2 1B (CPU-friendly)

Deployment Options:
- Docker Compose for local development
- Kubernetes for production scaling
- Cloud providers: AWS, GCP, Azure support
- On-premises deployment available

Security:
- AES-256 encryption at rest and in transit
- OAuth 2.0 and JWT authentication
- SOC 2 Type II compliant
- GDPR and CCPA compliant data handling

Integrations:
- OpenAI API (GPT-3.5, GPT-4)
- Anthropic Claude (Claude-3 family)
- LangChain framework
- LangGraph for complex workflows
- CodeCarbon for emissions tracking
""",
    
    "getting_started.txt": """
Getting Started with Understudy

Step 1: Installation
pip install understudy-client

Step 2: Set up environment variables
export OPENAI_API_KEY="your-openai-key"
export UNDERSTUDY_API_URL="http://localhost:8000"

Step 3: Create your first endpoint
from understudy import Understudy

client = Understudy()
endpoint = client.create_endpoint(
    name="my-assistant",
    llm_provider="openai", 
    llm_model="gpt-3.5-turbo"
)

Step 4: Use with LangChain
from understudy import UnderstudyLLM
from langchain.chains import LLMChain

llm = UnderstudyLLM(endpoint_id=endpoint["id"])
chain = prompt | llm | output_parser

Step 5: Monitor and activate SLM
- Use the dashboard to monitor training progress
- Activate SLM when similarity threshold is reached
- Enjoy 10-100x cost savings!

Common Issues:
- Insufficient training data: Need at least 10 LLM interactions
- API key errors: Ensure correct environment variables
- Connection issues: Check Understudy service is running
""",

    "pricing_billing.txt": """
Pricing and Billing Details

Plan Comparison:
- Starter Plan: $29/month
  * Up to 10,000 inferences per month
  * Email support
  * Basic analytics dashboard
  * 1 endpoint included
  
- Professional Plan: $99/month  
  * Up to 100,000 inferences per month
  * Priority email + chat support
  * Advanced analytics and metrics
  * Up to 5 endpoints
  * Custom model fine-tuning
  * API access
  
- Enterprise Plan: Custom pricing
  * Unlimited inferences
  * Dedicated support manager
  * Phone support with SLA
  * Unlimited endpoints
  * On-premises deployment
  * Custom integrations
  * Volume discounts available

Payment & Billing:
- Accepted payments: Credit cards, ACH, wire transfer
- Billing cycle: Monthly or annual (10% discount for annual)
- Free trial: 14 days with 1,000 free inferences
- Overages: $0.001 per additional inference
- Refund policy: 30-day money-back guarantee
- VAT handling: Automatic calculation for EU customers
- Setup fees: None for any plan
- Cancellation: Cancel anytime, no penalties

Volume Discounts:
- 20% off for 1M+ inferences/month
- 30% off for 10M+ inferences/month
- Non-profit organizations: 40% discount available
- Educational institutions: 50% discount with verification
""",

    "api_reference.txt": """
API Reference and Integration Guide

Authentication:
- API Key authentication required
- Include in header: Authorization: Bearer <api-key>
- Rate limits: 1000 requests/minute (Starter), 10000/min (Pro), unlimited (Enterprise)
- Request timeout: 30 seconds default, configurable up to 120 seconds

REST API Endpoints:
POST /api/v1/endpoints - Create new endpoint
GET /api/v1/endpoints - List all endpoints  
GET /api/v1/endpoints/{id} - Get endpoint details
DELETE /api/v1/endpoints/{id} - Delete endpoint
POST /api/v1/endpoints/{id}/inference - Run inference
POST /api/v1/endpoints/{id}/training - Start training
GET /api/v1/endpoints/{id}/metrics - Get performance metrics

Request/Response Format:
- Content-Type: application/json
- Response codes: 200 (success), 400 (bad request), 401 (unauthorized), 429 (rate limited), 500 (server error)
- Error format: {"error": {"code": "...", "message": "..."}}
- Request IDs included for debugging

Python SDK Methods:
client.create_endpoint(name, llm_provider, llm_model)
client.inference(endpoint_id, prompt, max_tokens, temperature)
client.start_training(endpoint_id, training_data_size)
client.get_metrics(endpoint_id)
client.activate_slm(endpoint_id)

Webhook Events:
- training.started - Training job initiated
- training.completed - Training finished successfully  
- training.failed - Training encountered error
- endpoint.activated - SLM activated for endpoint
- usage.threshold - Approaching plan limits

Error Handling Best Practices:
- Implement exponential backoff for retries
- Handle rate limiting with 429 response codes
- Check endpoint health before inference calls
- Use request IDs for debugging failed requests
""",

    "training_guide.txt": """
SLM Training Guide

Training Process:
1. Data Collection: Understudy automatically collects LLM interaction data
2. Minimum Requirements: 10 LLM calls before training can begin
3. Optimal Data: 100+ interactions for best performance
4. Training Time: 5-30 minutes depending on data size
5. Evaluation: Automatic similarity testing against LLM outputs
6. Activation: Manual or automatic when threshold reached (default 85%)

Similarity Threshold:
- Measures how closely SLM outputs match LLM outputs
- Default activation threshold: 85% similarity
- Configurable range: 70-95% (higher = more conservative)
- Calculated using semantic similarity (sentence transformers)
- Updated in real-time during training

Training Parameters:
- Learning rate: 5e-4 (automatically tuned)
- Batch size: 4-16 (based on available GPU memory)  
- Epochs: 3-10 (early stopping when convergence detected)
- LoRA rank: 16 (configurable 8-64)
- LoRA alpha: 32 (scales LoRA impact)
- Dropout: 0.1 (prevents overfitting)

Model Architecture:
- Base model: Llama 3.2 1B (optimized for CPU inference)
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- Quantization: 4-bit for memory efficiency
- Context length: 2048 tokens
- Parameter count: ~1B (base) + ~10M (LoRA adapters)

Training Monitoring:
- Real-time loss graphs in dashboard
- Training/validation accuracy metrics
- GPU utilization and memory usage
- Estimated completion time
- Carbon emissions tracking

Training Failures:
Common causes and solutions:
- Insufficient data: Need minimum 10 interactions
- GPU memory issues: Reduce batch size or use CPU training
- Connection timeouts: Check network stability
- Invalid training data: Ensure proper prompt/response format
- Model convergence issues: Try different learning rates

Recovery procedures:
- Automatic retry with adjusted parameters
- Manual restart with different configuration
- Data cleaning and reprocessing
- Escalation to support team for complex issues
""",

    "deployment_guide.txt": """
Deployment and Infrastructure Guide

Local Development:
- Docker Compose setup for full stack
- Minimum requirements: 8GB RAM, 4 CPU cores
- Recommended: 16GB RAM, 8 CPU cores, GPU optional
- Storage: 10GB for base models and training data
- Network: Stable internet for model downloads

Kubernetes Production:
- Minimum cluster: 3 nodes, 16GB RAM each
- Recommended: 5+ nodes with auto-scaling
- Persistent storage for model artifacts
- Load balancer for high availability
- Monitoring with Prometheus/Grafana

Cloud Provider Setup:
AWS:
- EKS cluster with m5.xlarge+ instances
- EFS for shared model storage  
- RDS PostgreSQL for production database
- ALB for load balancing
- CloudWatch for monitoring

GCP:
- GKE cluster with n1-standard-4+ machines
- Filestore for model artifacts
- Cloud SQL PostgreSQL
- Cloud Load Balancing
- Operations Suite for monitoring

Azure:
- AKS cluster with Standard_D4s_v3+ VMs
- Azure Files for storage
- Azure Database for PostgreSQL
- Application Gateway
- Azure Monitor

On-Premises:
- VMware vSphere or bare metal Kubernetes
- Ceph or NFS for shared storage
- External PostgreSQL cluster
- HAProxy or NGINX for load balancing
- Custom monitoring stack

Security Configuration:
- TLS 1.3 for all communications
- Network policies for pod-to-pod security
- Secrets management with Kubernetes secrets or Vault
- Regular security scanning and updates
- Firewall rules for external access
- VPN access for management interfaces

Backup and Recovery:
- Daily database backups with 30-day retention
- Model artifact backups to object storage
- Configuration backup in git repositories
- Disaster recovery procedures documented
- RTO: 4 hours, RPO: 1 hour for production
""",

    "troubleshooting.txt": """
Troubleshooting Guide

Common Issues:

API Key Errors:
- Symptoms: 401 Unauthorized responses
- Causes: Invalid/expired API key, incorrect header format
- Solutions: Verify key in dashboard, check Authorization header format
- Prevention: Rotate keys regularly, use environment variables

Connection Timeouts:
- Symptoms: Request timeouts, 504 Gateway Timeout
- Causes: Network issues, server overload, long-running inference
- Solutions: Check network connectivity, retry with backoff, increase timeout
- Prevention: Monitor endpoint health, implement circuit breakers

High Memory Usage:
- Symptoms: Out of memory errors, slow training
- Causes: Large batch sizes, insufficient RAM, memory leaks
- Solutions: Reduce batch size, restart services, upgrade resources
- Prevention: Monitor memory usage, set appropriate resource limits

Slow Inference:
- Symptoms: Response times >10 seconds
- Causes: Cold start, model loading, resource constraints
- Solutions: Keep endpoints warm, optimize model size, scale resources
- Prevention: Use SLM for faster inference, implement caching

Training Failures:
- Symptoms: Training stops with error, no progress updates
- Causes: Insufficient data, GPU errors, network issues
- Solutions: Check training data quality, restart with CPU, verify connectivity
- Prevention: Validate data before training, monitor GPU health

SSL Certificate Errors:
- Symptoms: Certificate verification failures
- Causes: Expired certificates, wrong hostname, self-signed certs
- Solutions: Update certificates, verify hostname, configure cert validation
- Prevention: Use cert-manager for auto-renewal, monitor expiration

Model Version Conflicts:
- Symptoms: Inconsistent responses, loading errors
- Causes: Multiple model versions, cache issues, deployment race conditions
- Solutions: Clear cache, redeploy with specific version, check deployment logs
- Prevention: Use semantic versioning, implement blue-green deployments

Debug Tools:
- Enable debug logging: Set LOG_LEVEL=DEBUG
- Check service health: GET /health endpoint
- Monitor metrics: Dashboard or API endpoints
- Trace requests: Use request IDs for correlation
- Performance profiling: Built-in timing metrics

Support Escalation:
- Collect logs and error messages
- Note reproduction steps
- Include environment details
- Check status page for known issues
- Contact support with incident ID
"""
}


async def setup_documents():
    """Create sample documents for the demo."""
    docs_dir = Path("./demo_docs")
    docs_dir.mkdir(exist_ok=True)
    
    for filename, content in SAMPLE_DOCS.items():
        doc_path = docs_dir / filename
        with open(doc_path, 'w') as f:
            f.write(content)
    
    return docs_dir


async def main():
    print("🔍 RAG System with Understudy Demo")
    print("=" * 50)
    
    # Initialize Understudy client
    print("\n🚀 Setting up Understudy...")
    client = Understudy(base_url="http://localhost:8000")
    
    # Create endpoint for RAG

    endpoint_id = "230e7f6c-185c-4307-8ebd-cbde608f2343"
    
    print(f"✅ Created RAG endpoint: {endpoint_id}")
    
    # Set up documents
    print("\n📄 Setting up document corpus...")
    docs_dir = await setup_documents()
    
    # Load documents into memory (simplified approach)
    documents_text = {}
    for doc_file in docs_dir.glob("*.txt"):
        with open(doc_file, 'r') as f:
            documents_text[doc_file.name] = f.read()
    
    print(f"✅ Loaded {len(documents_text)} documents")
    
    # Create an improved search function
    def advanced_search(question: str, docs: dict, k: int = 5) -> str:
        """Advanced search with better keyword matching and scoring."""
        question_lower = question.lower()
        question_words = [word.strip('?.,!') for word in question_lower.split()]
        
        # Enhanced keyword mappings for better matching
        keyword_mappings = {
            'cost': ['price', 'pricing', 'billing', 'payment', 'plan', 'fee', 'discount'],
            'price': ['cost', 'pricing', 'billing', 'payment', 'plan', 'fee', 'discount'],
            'api': ['endpoint', 'authentication', 'request', 'response', 'integration'],
            'training': ['slm', 'model', 'learning', 'similarity', 'threshold', 'fine-tuning'],
            'deploy': ['deployment', 'kubernetes', 'docker', 'cloud', 'infrastructure'],
            'error': ['troubleshooting', 'issue', 'problem', 'debug', 'failure'],
            'setup': ['installation', 'configuration', 'getting started', 'environment'],
            'support': ['help', 'contact', 'documentation', 'assistance'],
            'llm': ['large language model', 'gpt', 'claude', 'openai', 'anthropic'],
            'slm': ['small language model', 'fine-tuned', 'trained', 'lora'],
            'inference': ['prediction', 'generation', 'response', 'completion'],
            'authentication': ['api key', 'oauth', 'jwt', 'authorization', 'login'],
            'plan': ['starter', 'professional', 'enterprise', 'subscription'],
            'memory': ['ram', 'gpu', 'cpu', 'resources', 'usage'],
            'timeout': ['connection', 'network', 'latency', 'slow'],
            'ssl': ['certificate', 'https', 'tls', 'security'],
            'webhook': ['event', 'notification', 'callback'],
            'langchain': ['integration', 'framework', 'chain']
        }
        
        # Expand search terms
        expanded_terms = set(question_words)
        for word in question_words:
            if word in keyword_mappings:
                expanded_terms.update(keyword_mappings[word])
        
        scored_chunks = []
        
        for filename, content in docs.items():
            # Split into sentences and paragraphs for better granularity
            sections = content.split('\n\n')
            for section in sections:
                if not section.strip():
                    continue
                    
                section_lower = section.lower()
                score = 0
                
                # Score based on exact matches
                for term in expanded_terms:
                    if term in section_lower:
                        # Higher score for exact question word matches
                        if term in question_words:
                            score += 3
                        else:
                            score += 1
                        
                        # Bonus for multiple occurrences
                        score += section_lower.count(term) * 0.5
                
                # Bonus for sections containing multiple search terms
                matching_terms = sum(1 for term in expanded_terms if term in section_lower)
                if matching_terms > 1:
                    score += matching_terms * 2
                
                # Penalty for very short sections
                if len(section) < 100:
                    score *= 0.5
                
                if score > 0:
                    scored_chunks.append((score, section, filename))
        
        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Format results with source attribution
        results = []
        for score, chunk, filename in scored_chunks[:k]:
            results.append(f"[From {filename}]\n{chunk}")
        
        return '\n\n---\n\n'.join(results) if results else ""
    
    print("✅ Search function created")
    
    # Create Understudy LLM (same as langchain_basic)
    llm = UnderstudyLLM(
        endpoint_id=endpoint_id,
        base_url="http://localhost:8000",
        max_tokens=300,
        temperature=0.3  # Lower temperature for factual Q&A
    )
    
    # Create enhanced prompt for RAG
    prompt_template = """You are a helpful assistant for Understudy, an AI platform. Use the provided context to answer questions accurately.

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain relevant information, say "I don't have information about that in the available documentation"
- Be specific and cite information from the context when possible
- If the context is partially relevant, answer what you can and note what's not covered

Context:
{context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create simple Q&A chain
    qa_chain = LLMChain(llm=llm, prompt=PROMPT)
    
    # Extended Q&A scenarios - 50+ for extensive testing
    basic_questions = [
        "What is Understudy and what does it do?",
        "How much does the Professional plan cost?", 
        "What base model does Understudy use for training?",
        "How do I install the Understudy client?",
        "What security features does Understudy have?",
        "What deployment options are available?",
        "How do I create my first endpoint?",
        "What support options are available?",
        "What cloud providers does Understudy support?",
        "How do I troubleshoot API key errors?"
    ]
    
    # Additional technical questions
    technical_questions = [
        "What Python version is required for Understudy?",
        "How do I set up environment variables?",
        "What's the difference between LLM and SLM training?",
        "How does the similarity threshold work?",
        "What machine learning frameworks does Understudy use?",
        "How does the cost reduction calculation work?",
        "What's the maximum file attachment size?",
        "How do I configure OAuth 2.0 authentication?",
        "What's the API rate limit?",
        "How do I enable debug logging?",
        "What Docker images are available?",
        "How do I configure Kubernetes deployments?",
        "What's the database schema structure?",
        "How does data encryption work?",
        "What monitoring tools are integrated?"
    ]
    
    # Business and pricing questions
    business_questions = [
        "What's included in the Starter plan?",
        "How does Enterprise pricing work?",
        "What payment methods are accepted?",
        "Is there a free trial period?",
        "What's your refund policy?",
        "Do you offer volume discounts?",
        "What's the cancellation process?",
        "Are there setup fees?",
        "How is billing calculated for API usage?",
        "Do you offer non-profit discounts?",
        "What happens if I exceed my plan limits?",
        "Can I change plans mid-cycle?",
        "How do you handle VAT for international customers?",
        "What's your SLA for uptime?",
        "Do you offer professional services?"
    ]
    
    # Integration and compatibility questions
    integration_questions = [
        "How do I integrate with LangChain?",
        "Does Understudy work with LangGraph?",
        "What OpenAI models are supported?",
        "Can I use Anthropic Claude models?",
        "How do I integrate with existing applications?",
        "What webhook events are available?",
        "How do I use the Python SDK?",
        "Is there a REST API available?",
        "How do I authenticate API requests?",
        "What response formats are supported?",
        "Can I use custom prompts?",
        "How do I handle API errors?",
        "What's the request/response structure?",
        "How do I implement retry logic?",
        "Can I batch API requests?"
    ]
    
    # Edge case and troubleshooting questions
    edge_questions = [
        "What happens if training fails?",
        "How do I handle connection timeouts?",
        "What if my endpoint becomes unresponsive?",
        "How do I recover from a failed deployment?",
        "What should I do if metrics aren't updating?",
        "How do I debug slow inference responses?",
        "What causes high memory usage during training?",
        "How do I fix SSL certificate errors?",
        "What if my API key gets compromised?",
        "How do I handle model version conflicts?"
    ]
    
    # Combine all question sets
    questions = (basic_questions + 
                technical_questions + 
                business_questions + 
                integration_questions + 
                edge_questions)
    
    print("\n💬 Running Q&A scenarios...")
    print("=" * 80)
    
    # Process questions
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}/{len(questions)}: {question}")
        
        # Retrieve relevant context
        context = advanced_search(question, documents_text, k=5)
        
        # Get answer from LLM with context
        answer = qa_chain.run(context=context, question=question)
        
        print(f"🤖 Answer: {answer}")
        
        # Show if we found relevant context
        if context:
            print(f"📚 Found relevant context ({len(context)} chars)")
        
        print("-" * 40)
    
    print("\n" + "=" * 80)
    
    # Check training progress
    print("\n📊 Checking training status...")
    try:
        metrics = client.get_metrics(endpoint_id)
        print(f"Total Q&A interactions: {metrics['total_inferences']}")
        print(f"LLM calls: {metrics['llm_inferences']}")
        
        # Start training if we have enough data
        if metrics['llm_inferences'] >= 10:
            print("\n🏋️ Starting SLM training...")
            training_result = client.start_training(endpoint_id)
            print(f"Training: {training_result['message']}")
        else:
            print(f"Need {10 - metrics['llm_inferences']} more interactions for training")
            
    except Exception as e:
        print(f"Starting training with current data...")
        training_result = client.start_training(endpoint_id)
        print(f"Training: {training_result['message']}")
    
    # Show carbon impact
    try:
        carbon = client.get_carbon_summary(endpoint_id)
        print(f"\n🌱 Environmental Impact:")
        print(f"Training emissions: {carbon['total_training_emissions_kg']:.6f} kg CO₂")
        print(f"Estimated savings: {carbon['avoided_emissions_kg']:.6f} kg CO₂")
    except Exception as e:
        print("Carbon tracking will be available after training completes")
    
    print(f"\n🎉 RAG Demo complete!")
    print(f"📊 Dashboard: http://localhost:3000/endpoints/{endpoint_id}")
    print("\n💡 Next steps:")
    print("   1. Monitor training progress in the dashboard")
    print("   2. Activate SLM when similarity threshold is reached")
    print("   3. Continue using the RAG system with cost savings!")
    
    # Cleanup
    import shutil
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
        print("\n🧹 Cleaned up demo documents")


if __name__ == "__main__":
    asyncio.run(main())