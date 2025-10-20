# Understudy Python Client

The official Python client for Understudy - LLM to SLM Training Platform.

## Installation

```bash
pip install understudy-client
```

## Quick Start

```python
from understudy import Understudy, UnderstudyLLM

# Initialize client
client = Understudy(base_url="http://localhost:8000")

# Create endpoint
endpoint = client.create_endpoint(
    name="my-assistant",
    llm_provider="openai",
    llm_model="gpt-3.5-turbo"
)

# Use with LangChain
llm = UnderstudyLLM(endpoint_id=endpoint["id"])
response = llm.invoke("Hello, world!")
```

## LangChain Integration

### Basic LLM

```python
from understudy import UnderstudyLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = UnderstudyLLM(endpoint_id="your-endpoint-id")
prompt = PromptTemplate.from_template("Answer: {question}")
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(question="What is AI?")
```

### Chat Model

```python
from understudy import UnderstudyChatModel
from langchain_core.messages import HumanMessage

chat = UnderstudyChatModel(endpoint_id="your-endpoint-id")
response = chat.invoke([HumanMessage(content="Hello!")])
```

## LangGraph Integration

```python
from understudy import create_understudy_node
from langgraph.graph import StateGraph

# Create Understudy-powered node
node = create_understudy_node(
    endpoint_id="your-endpoint-id",
    base_url="http://localhost:8000"
)

# Add to graph
workflow = StateGraph(YourState)
workflow.add_node("assistant", node)
```

## API Reference

### Understudy Client

#### Methods

- `create_endpoint()` - Create a new endpoint
- `list_endpoints()` - List all endpoints  
- `get_endpoint()` - Get endpoint details
- `delete_endpoint()` - Delete an endpoint
- `activate_endpoint()` - Activate SLM for endpoint
- `generate()` - Generate text
- `start_training()` - Start SLM training
- `get_metrics()` - Get endpoint metrics
- `get_carbon_summary()` - Get carbon emissions data
- `health_check()` - Check API health

### UnderstudyLLM

LangChain-compatible LLM wrapper.

#### Parameters

- `endpoint_id` (str): Understudy endpoint ID
- `base_url` (str): API base URL  
- `api_key` (str, optional): Authentication key
- `max_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature

### UnderstudyChatModel

LangChain-compatible chat model wrapper.

#### Parameters

Same as UnderstudyLLM, but supports message-based conversations.

## Examples

See the `/examples` directory for complete examples:

- `langchain_basic.py` - Basic LangChain integration
- `langchain_rag.py` - RAG system with document Q&A
- `langgraph_agent.py` - Multi-agent system with LangGraph

## License

MIT License