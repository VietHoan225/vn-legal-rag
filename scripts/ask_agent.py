"""Test script to chat with the Legal AI Agent."""

import argparse
import sys
from pathlib import Path
import logging

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import QdrantVectorStore
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from src.agent.tools import LegalSearchTool
from src.agent.agent import LegalAgent

logging.basicConfig(level=logging.INFO)  # Show info to see tool calls

def main():
    print("Khởi tạo AI Agent...")
    
    # Khởi tạo các thành phần
    embedder = Embedder()
    vector_store = QdrantVectorStore()
    retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
    search_tool = LegalSearchTool(retriever=retriever, reranker=Reranker())
    
    # Model có hỗ trợ tool-use của OpenRouter
    agent = LegalAgent(search_tool=search_tool)
    print("\n" + "="*50)
    print("⚖️ TƯ VẤN PHÁP LUẬT VN (Agent Mode)")
    print("Nhập 'quit' hoặc 'exit' để thoát.")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nBạn: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            if not user_input.strip():
                continue
                
            print("\nAgent đang suy nghĩ...")
            response = agent.chat(user_input)
            print("\nAI:", response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nLỗi: {e}")

if __name__ == "__main__":
    main()
