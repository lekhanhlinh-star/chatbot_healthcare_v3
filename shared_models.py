"""
Shared models initialization for all RAG inference modules.
This file initializes common models once to save memory and improve performance.
"""

import os

print("Initializing shared models...")

try:
    from langchain_ollama import OllamaEmbeddings
    from langchain_ollama import ChatOllama
    from langchain_classic.retrievers.document_compressors import  CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # Shared Embeddings Model
    print("Loading embeddings model: bge-m3...")
    shared_embeddings = OllamaEmbeddings(model="bge-m3")

    # Shared Chat Model  
    print("Loading chat model: qwen3:14b...")
    shared_chat_model = ChatOllama(
        model="qwen3:14b",
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.2,
    )

    # Shared Reranker Model
    print("Loading reranker model: BAAI/bge-reranker-v2-m3...")
    shared_reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    # shared_reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    shared_compressor = CrossEncoderReranker(model=shared_reranker, top_n=5)

    print("✅ All shared models loaded successfully!")

except ImportError as e:
    print(f"❌ Error importing dependencies: {e}")
    print("📝 Please install required packages:")
    print("pip install -r requirements.txt")
    # Create dummy objects to prevent import errors
    shared_embeddings = None
    shared_chat_model = None
    shared_compressor = None
    shared_reranker = None