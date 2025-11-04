#!/bin/bash
# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama server is ready
echo "Waiting for Ollama server to start..."
while ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; do
    sleep 2
done
echo "Ollama server is ready!"

# Pull models sequentially
ollama pull qwen3:14b
ollama pull bge-m3

# Run Python app
python app.py