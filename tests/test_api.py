import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app
from src.retrieval.vector_store import SearchResult
from src.generation.generator import GenerationResult

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running"}

@patch("src.api.main.get_embedder")
@patch("src.api.main.get_vector_store")
@patch("src.api.main.get_generator")
def test_ask_question_success(mock_get_generator, mock_get_vector_store, mock_get_embedder):
    # Setup mocks
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 384
    mock_get_embedder.return_value = mock_embed
    
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SearchResult(
            chunk_id="1", 
            text="Điều 1: Luật này...", 
            score=0.9, 
            doc_id=100,
            article_id="art_1",
            article_number=1,
            title="Điều 1",
            category="general"
        )
    ]
    mock_get_vector_store.return_value = mock_search

    mock_generate = MagicMock()
    mock_generate.generate_answer.return_value = GenerationResult(
        answer="Theo Điều 1, luật này...",
        prompt_used="test prompt"
    )
    mock_get_generator.return_value = mock_generate

    # Make request
    response = client.post("/ask", json={"question": "Luật này nói gì?"})

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Theo Điều 1, luật này..."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["id"] == "1"
    assert data["sources"][0]["content"] == "Điều 1: Luật này..."

@patch("src.api.main.get_embedder")
@patch("src.api.main.get_vector_store")
def test_ask_question_no_results(mock_get_vector_store, mock_get_embedder):
    # Setup mocks
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 384
    mock_get_embedder.return_value = mock_embed
    
    mock_search = MagicMock()
    mock_search.search.return_value = []
    mock_get_vector_store.return_value = mock_search

    # Make request
    response = client.post("/ask", json={"question": "Câu hỏi không có trong luật?"})

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "không tìm thấy" in data["answer"].lower()
    assert len(data["sources"]) == 0

def test_ask_question_empty_query():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()
