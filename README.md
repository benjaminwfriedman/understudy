# Understudy 🎭

**LLM to SLM Training & Transition Platform with LangChain Integration**

Understudy automatically captures your LLM interactions and progressively trains Small Language Models (SLMs) to mimic your Large Language Model's behavior. Once the SLM reaches your accuracy threshold, Understudy seamlessly switches to the SLM - delivering **10-100x cost reduction** and **significant carbon savings**.

## 🌟 Key Features

- 💰 **Massive Cost Reduction**: 10-100x cheaper inference with trained SLMs
- 🌱 **Carbon Tracking**: Real-time emissions monitoring with CodeCarbon
- 🔗 **LangChain Compatible**: Drop-in replacement for LangChain LLMs
- 🕸️ **LangGraph Support**: Full compatibility with complex agent workflows (**FUTURE**)
- 📊 **Visual Dashboard**: Monitor training progress and carbon impact
- 🔄 **Seamless Switchover**: Automatic transition when SLM reaches threshold
- 🤖 **Multi-Model Support**: OpenAI, Anthropic, and more

## 🚀 Quick Start

### 1. Installation

A. Clone the repo:

```bash
# Clone the repository
git clone https://github.com/your-org/understudy.git
cd understudy
```
B. Run runpod_config.sh 

```bash
./runpod_config.sh

# follow instructions from console to add it to your runpod account
```

C. Update docker-compose with your secrets

    * OPENAI_API_KEY for initial LLM inferance
    * HF_TOKEN for access to Llama 3.2 1B
    * RUNPOD_API_KEY for remote SLM fine-tuning
    * RUNPOD_SSH_PUBLIC_KEY from runpod_config.sh
    * RUNPOD_SSH_PRIVATE_KEY_PATH path to private key generated from runpod_config.sh

D. Build and start containers 
```bash
docker-compose up
```

### 2. Basic Usage
```bash
pip install -r requiremets.txt # includes the local install of the understudy-client
```

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


## 📊 Dashboard

Visit `http://localhost:3000` to see:

- **Endpoint Management**: Create and configure LLM-to-SLM endpoints
- **Training Progress**: Real-time similarity metrics and training status (**FUTURE**)
- **Carbon Impact**: Emissions tracking and environmental savings (**Buggy**)
- **Cost Analytics**: Before/after cost comparisons (**FUTURE**)

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

### Debug
```bash
./debug.sh ## triggers debug settings
```
#### vscode debug launch.json configuration
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach to Docker Backend",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}/backend",
                    "remoteRoot": "/app"
                }
            ],
            "justMyCode": false,
            "redirectOutput": true,
            "logToFile": true
        },
        {
            "name": "Containers: Python - Fastapi",
            "type": "docker",
            "request": "launch",
            "preLaunchTask": "docker-run: debug",
            "python": {
                "pathMappings": [
                    {
                        "localRoot": "${workspaceFolder}",
                        "remoteRoot": "/app"
                    }
                ],
                "projectType": "fastapi"
            }
        }
    ]
}
```


### Kubernetes (**FUTURE**)
```bash
kubectl apply -f k8s/
```


*Reduce AI costs, reduce carbon footprint, one SLM at a time.* 🌍