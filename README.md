# Understudy 🎭

**LLM to SLM Training & Transition Platform with LangChain Integration**

Understudy automatically captures your LLM interactions and progressively trains Small Language Models (SLMs) to mimic your Large Language Model's behavior. Once the SLM reaches your accuracy threshold, Understudy seamlessly switches to the SLM - delivering **10-100x cost reduction** and **significant carbon savings**.

## 🌟 Key Features

- 💰 **Massive Cost Reduction**: 10-100x cheaper inference with trained SLMs
- 🌱 **Carbon Tracking**: Real-time emissions monitoring with CodeCarbon
- 🔗 **LangChain Compatible**: Drop-in replacement for LangChain LLMs
- 🕸️ **LangGraph Support**: Full compatibility with complex agent workflows
- 📊 **Visual Dashboard**: Monitor training progress and carbon impact
- 🔄 **Seamless Switchover**: Automatic transition when SLM reaches threshold
- 🤖 **Multi-Model Support**: OpenAI, Anthropic, and more

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/understudy.git
cd understudy

# Start with Docker Compose
docker-compose up -d

# Or install Python client
pip install understudy-client
```

### 2. Basic Usage

```python
from understudy import Understudy, UnderstudyLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Create Understudy client
client = Understudy(base_url="http://localhost:8000")

# Create endpoint
endpoint = client.create_endpoint(
    name="my-assistant",
    llm_provider="openai",
    llm_model="gpt-3.5-turbo",
    similarity_threshold=0.95
)

# Use with LangChain (automatically logs interactions)
llm = UnderstudyLLM(endpoint_id=endpoint["id"])
chain = prompt | llm | output_parser

# Your LLM calls are now training an SLM!
response = chain.invoke({"input": "Hello, world!"})
```

### 3. Monitor & Activate

```python
# Check training progress
metrics = client.get_metrics(endpoint["id"])
print(f"Similarity: {metrics['avg_similarity']:.2%}")

# Activate SLM when ready
if metrics['avg_similarity'] > 0.95:
    client.activate_slm(endpoint["id"])
    print("🎉 SLM activated! Enjoying 10-100x cost savings!")
```

## 📊 Dashboard

Visit `http://localhost:3000` to see:

- **Endpoint Management**: Create and configure LLM-to-SLM endpoints
- **Training Progress**: Real-time similarity metrics and training status
- **Carbon Impact**: Emissions tracking and environmental savings
- **Cost Analytics**: Before/after cost comparisons
- **LangChain Integration**: Setup guides and examples

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               LangChain/LangGraph Application                │
│                                                             │
│    from understudy import UnderstudyLLM                    │
│    llm = UnderstudyLLM(endpoint_id="my-endpoint")          │
│    chain = prompt | llm | output_parser                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Understudy Platform                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Inference  │  │   Training   │  │  Metrics &   │      │
│  │   Router     │  │   Scheduler  │  │  Analytics   │      │
│  │              │  │ +CodeCarbon  │  │  +Carbon     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬────────────────────┬─────────────────┘
                       │                    │
        ┌──────────────┴─────────┬──────────┴─────────┐
        ▼                        ▼                     ▼
┌──────────────┐        ┌──────────────┐      ┌──────────────┐
│ LLM Provider │        │   Training   │      │   React      │
│  (OpenAI)    │        │   Worker     │      │  Dashboard   │
└──────────────┘        └──────────────┘      └──────────────┘
                                │
                        ┌───────┴────────┐
                        ▼                ▼
                ┌──────────────┐  ┌──────────────┐
                │ Llama 3.2 1B │  │    SQLite    │
                │ + LoRA       │  │   Database   │
                └──────────────┘  └──────────────┘
```

## 📚 Examples

### LangChain Integration

```python
# examples/langchain_basic.py
from understudy import UnderstudyLLM
from langchain.chains import LLMChain

llm = UnderstudyLLM(endpoint_id="customer-support")
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(question="How do I reset my password?")
```

### RAG System

```python
# examples/langchain_rag.py
from understudy import UnderstudyChatModel
from langchain.chains import RetrievalQA

llm = UnderstudyChatModel(endpoint_id="documentation-qa")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
```

### LangGraph Agents

```python
# examples/langgraph_agent.py
from understudy import create_understudy_node
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("agent", create_understudy_node(
    endpoint_id="planning-agent"
))
```

## 🌱 Carbon Impact

Understudy tracks carbon emissions in real-time using CodeCarbon:

- **Training Emissions**: One-time cost to train your SLM
- **Inference Savings**: Ongoing carbon reduction from SLM usage  
- **Net Impact**: Automatic payback calculation
- **Equivalencies**: Miles driven, trees planted, LED hours

Example impact after 1000 inferences:
- Training: 0.045 kg CO₂
- Saved: 0.892 kg CO₂  
- **Net Savings: 0.847 kg CO₂** ✅

## 🚢 Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

## 📖 Documentation

- [**Getting Started Guide**](docs/getting-started.md)
- [**LangChain Integration**](docs/langchain-integration.md)
- [**LangGraph Support**](docs/langgraph-support.md)
- [**API Reference**](docs/api-reference.md)
- [**Deployment Guide**](docs/deployment.md)
- [**Carbon Tracking**](docs/carbon-tracking.md)

## 🛠️ Configuration

### Environment Variables

```bash
# Backend
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DATABASE_URL=sqlite+aiosqlite:///./understudy.db
CARBON_TRACKING_ENABLED=true

# Training
DEFAULT_SIMILARITY_THRESHOLD=0.95
DEFAULT_BATCH_SIZE=100
BASE_MODEL_PATH=meta-llama/Llama-3.2-1B

# Carbon Tracking
COUNTRY_ISO_CODE=USA
CARBON_DATA_DIR=./carbon_data
```

### Endpoint Configuration

```python
endpoint = client.create_endpoint(
    name="my-endpoint",
    llm_provider="openai",
    llm_model="gpt-3.5-turbo",
    config={
        "training_batch_size": 100,
        "similarity_threshold": 0.95,
        "auto_switchover": False,
        "lora_r": 8,
        "lora_alpha": 16,
        "learning_rate": 3e-4,
        "track_carbon": True
    }
)
```

## 🧪 Development

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose

### Setup
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend  
cd frontend
npm install
npm run dev

# Client SDK
cd client
pip install -e .
```

### Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test

# Integration tests
python examples/langchain_basic.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.understudy.ai](https://docs.understudy.ai)
- **Issues**: [GitHub Issues](https://github.com/your-org/understudy/issues)
- **Discord**: [Understudy Community](https://discord.gg/understudy)
- **Email**: support@understudy.ai

## 🗺️ Roadmap

- [ ] **Q1 2024**: Anthropic Claude integration
- [ ] **Q2 2024**: Multi-GPU training support
- [ ] **Q3 2024**: Kubernetes operator
- [ ] **Q4 2024**: Advanced LoRA techniques (QLoRA, AdaLoRA)
- [ ] **2025**: Custom base model support

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/understudy&type=Date)](https://star-history.com/#your-org/understudy&Date)

---

**Built with ❤️ by the Understudy team**

*Reduce AI costs, reduce carbon footprint, one SLM at a time.* 🌍