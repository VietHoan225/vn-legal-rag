"""LLM Generator using Groq API."""

import logging
from groq import Groq
from pydantic import BaseModel

from src.config import settings
from src.retrieval.vector_store import SearchResult
from src.generation.prompts import RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    """Structured result from LLM generation."""
    answer: str
    prompt_used: str


class GroqGenerator:
    """Class to interact with Groq API (Llama models) for text generation."""

    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        """Initialize the Groq client."""
        if not settings.groq_api_key:
            logger.warning("groq_api_key is empty. Generation will fail if called.")
        
        self.client = Groq(api_key=settings.groq_api_key, timeout=30.0)
        self.model_name = model_name

    def format_context(self, search_results: list[SearchResult]) -> str:
        """Combine retrieved search results into a single context string."""
        context_parts = []
        for rank, res in enumerate(search_results, start=1):
            source = f"[{rank}] {res.title} (Điều {res.article_number})" if res.article_number else f"[{rank}] {res.title}"
            text = f"Nguồn: {source}\nNội dung: {res.text}"
            context_parts.append(text)
            
        return "\n\n---\n\n".join(context_parts)

    def generate_answer(
        self, 
        question: str, 
        search_results: list[SearchResult]
    ) -> GenerationResult:
        """Generate an answer using Groq given the question and retrieved contexts."""
        context_str = self.format_context(search_results)
        
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=question
        )
        
        logger.info("Calling Groq (%s) for generation...", self.model_name)
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
        )
        
        answer = response.choices[0].message.content or ""
        return GenerationResult(
            answer=answer,
            prompt_used=prompt
        )
