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
    endpoint = client.create_endpoint(
        name="documentation-qa",
        description="RAG system for product documentation Q&A", 
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        training_batch_size=20,
        similarity_threshold=0.90
    )
    endpoint_id = endpoint["id"]
    print(f"✅ Created RAG endpoint: {endpoint['name']}")
    
    # Set up documents
    print("\n📄 Setting up document corpus...")
    docs_dir = await setup_documents()
    
    # Load documents into memory (simplified approach)
    documents_text = {}
    for doc_file in docs_dir.glob("*.txt"):
        with open(doc_file, 'r') as f:
            documents_text[doc_file.name] = f.read()
    
    print(f"✅ Loaded {len(documents_text)} documents")
    
    # Create a simple in-memory "vector store" (for demo purposes)
    # In production, you would use real embeddings and vector search
    def simple_search(question: str, docs: dict, k: int = 3) -> str:
        """Simple keyword-based search for demo."""
        question_lower = question.lower()
        relevant_chunks = []
        
        for filename, content in docs.items():
            # Split into paragraphs
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if any(word in para.lower() for word in question_lower.split()):
                    relevant_chunks.append(para)
        
        # Return top k chunks
        return '\n\n'.join(relevant_chunks[:k])
    
    print("✅ Search function created")
    
    # Create Understudy LLM (same as langchain_basic)
    llm = UnderstudyLLM(
        endpoint_id=endpoint_id,
        base_url="http://localhost:8000",
        max_tokens=300,
        temperature=0.3  # Lower temperature for factual Q&A
    )
    
    # Create custom prompt for RAG
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

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
    
    # Sample questions for demo
    questions = [
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
    
    print("\n💬 Running Q&A scenarios...")
    print("=" * 80)
    
    # Process questions
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}/{len(questions)}: {question}")
        
        # Retrieve relevant context
        context = simple_search(question, documents_text, k=3)
        
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