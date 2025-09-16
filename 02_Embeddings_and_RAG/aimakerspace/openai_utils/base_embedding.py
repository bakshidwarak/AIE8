from abc import ABC, abstractmethod
from typing import List
import asyncio

class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding models"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        pass
    
    @abstractmethod
    async def async_get_embedding(self, text: str) -> List[float]:
        """Async version of get_embedding"""
        pass
    
    @abstractmethod
    async def async_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of get_embeddings"""
        pass