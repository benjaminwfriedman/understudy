# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Base stage
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies including Azure CLI
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    lsb-release \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -sLS https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /etc/apt/keyrings/microsoft.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/microsoft.gpg \
    && echo "deb [arch=`dpkg --print-architecture` signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/azure-cli/ bookworm main" | tee /etc/apt/sources.list.d/azure-cli.list \
    && apt-get update \
    && apt-get install -y azure-cli \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Create necessary directories including HuggingFace cache and Azure CLI config
RUN mkdir -p /app/models /app/checkpoints /app/carbon_data /app/logs /home/appuser/.cache /home/appuser/.azure && \
    chown -R appuser:appuser /app /home/appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

### Debug ###
FROM base as debug
# Switch back to root to install debugpy
USER root
RUN /opt/venv/bin/pip install debugpy
# Switch back to appuser for running
USER appuser

CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

## PROD ##
FROM base as prod
# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1



# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]