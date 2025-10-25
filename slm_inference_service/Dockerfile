FROM python:3.11-slim
WORKDIR /app
RUN pip install torch transformers vllm fastapi uvicorn
COPY . .
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--host", "0.0.0.0", "--port", "8000"]
