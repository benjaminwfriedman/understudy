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
from understudy import Understudy, UnderstudyChatModel
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
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
    
    # Load and process documents
    documents = []
    for doc_file in docs_dir.glob("*.txt"):
        loader = TextFileLoader(str(doc_file))
        docs = loader.load()
        documents.extend(docs)
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(splits)} chunks")
    
    # Create embeddings and vector store
    print("\n🔍 Creating vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("✅ Vector store created")
    
    # Create Understudy chat model
    llm = UnderstudyChatModel(
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
    
    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
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
        
        # Get answer from RAG system
        result = qa_chain({"query": question})
        answer = result["result"]
        sources = result["source_documents"]
        
        print(f"🤖 Answer: {answer}")
        
        # Show sources
        if sources:
            print(f"📚 Sources: {len(sources)} documents")
        
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