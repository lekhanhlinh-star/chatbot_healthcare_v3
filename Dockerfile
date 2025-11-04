
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    default-mysql-client \
    ffmpeg \
    git \
    python3-pip \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    software-properties-common && \
    ln -sf python3.10 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /chatbot_healthcare

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy application files
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create directory for Ollama models
RUN mkdir -p /root/.ollama

# Set environment variables for Ollama
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/root/.ollama/models

# Download Whisper large-v2 model
RUN python -c "import whisper; whisper.load_model('medium')"

# Download HuggingFace reranker model
RUN python -c "from langchain_community.cross_encoders import HuggingFaceCrossEncoder; HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-v2-m3')"

# Start Ollama service temporarily and pull models during build
RUN ollama serve & \
    OLLAMA_PID=$! && \
    echo "Waiting for Ollama server to start..." && \
    while ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do \
        sleep 2; \
    done && \
    echo "Ollama server is ready! Pulling models..." && \
    ollama pull qwen3:14b && \
    ollama pull bge-m3 && \
    echo "Models pulled successfully!" && \
    kill $OLLAMA_PID && \
    wait $OLLAMA_PID 2>/dev/null || true
# EXPOSE 11434 80

ENTRYPOINT ["./entrypoint.sh"]