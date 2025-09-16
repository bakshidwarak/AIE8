"""
Enhanced Vector Database with support for multiple embedding models

This module extends the original VectorDatabase to work with any embedding model
that implements the BaseEmbeddingModel interface.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Union
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.openai_utils.alternative_embeddings import BaseEmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the euclidean distance between two vectors."""
    return np.linalg.norm(vector_a - vector_b)


def dot_product_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the dot product similarity between two vectors."""
    return np.dot(vector_a, vector_b)


class EnhancedVectorDatabase:
    """
    Enhanced vector database that supports multiple embedding models
    and distance metrics.
    """
    
    def __init__(self, embedding_model: Union[EmbeddingModel, BaseEmbeddingModel] = None):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)  # Store additional metadata for each vector
        self.embedding_model = embedding_model or EmbeddingModel()
    
    def insert(self, key: str, vector: np.array, metadata: dict = None) -> None:
        """Insert a vector with optional metadata"""
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata
    
    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors using the specified distance measure"""
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    
    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        """Search using text query"""
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results
    
    def search_with_metadata(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        metadata_filter: dict = None
    ) -> List[Tuple[str, float, dict]]:
        """Search with optional metadata filtering"""
        scores = []
        for key, vector in self.vectors.items():
            # Apply metadata filter if provided
            if metadata_filter:
                if not self._matches_metadata_filter(key, metadata_filter):
                    continue
            
            score = distance_measure(query_vector, vector)
            metadata = self.metadata.get(key, {})
            scores.append((key, score, metadata))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    
    def _matches_metadata_filter(self, key: str, filter_dict: dict) -> bool:
        """Check if vector metadata matches the filter criteria"""
        vector_metadata = self.metadata.get(key, {})
        for filter_key, filter_value in filter_dict.items():
            if filter_key not in vector_metadata:
                return False
            if vector_metadata[filter_key] != filter_value:
                return False
        return True
    
    def retrieve_from_key(self, key: str) -> np.array:
        """Retrieve vector by key"""
        return self.vectors.get(key, None)
    
    def retrieve_metadata(self, key: str) -> dict:
        """Retrieve metadata by key"""
        return self.metadata.get(key, {})
    
    def get_all_keys(self) -> List[str]:
        """Get all vector keys"""
        return list(self.vectors.keys())
    
    def get_vector_count(self) -> int:
        """Get total number of vectors"""
        return len(self.vectors)
    
    def clear(self) -> None:
        """Clear all vectors and metadata"""
        self.vectors.clear()
        self.metadata.clear()
    
    async def abuild_from_list(
        self, 
        list_of_text: List[str], 
        metadata_list: List[dict] = None
    ) -> "EnhancedVectorDatabase":
        """Build vector database from list of texts with optional metadata"""
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            self.insert(text, np.array(embedding), metadata)
        
        return self
    
    def build_from_list(
        self, 
        list_of_text: List[str], 
        metadata_list: List[dict] = None
    ) -> "EnhancedVectorDatabase":
        """Synchronous version of abuild_from_list"""
        embeddings = self.embedding_model.get_embeddings(list_of_text)
        
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            self.insert(text, np.array(embedding), metadata)
        
        return self
    
    def get_distance_measures(self) -> dict:
        """Get available distance measures"""
        return {
            "cosine_similarity": cosine_similarity,
            "euclidean_distance": euclidean_distance,
            "dot_product_similarity": dot_product_similarity
        }


# Backward compatibility - alias for the original VectorDatabase
VectorDatabase = EnhancedVectorDatabase


if __name__ == "__main__":
    # Test with different embedding models
    from aimakerspace.openai_utils.alternative_embeddings import EmbeddingModelFactory
    
    # Test data
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]
    
    # Test with Sentence Transformers
    try:
        print("Testing with Sentence Transformers...")
        embedding_model = EmbeddingModelFactory.create_model("sentence_transformer", model_name="all-MiniLM-L6-v2")
        vector_db = EnhancedVectorDatabase(embedding_model)
        vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
        
        # Test search
        results = vector_db.search_by_text("I think fruit is awesome!", k=2)
        print(f"Search results: {results}")
        
        # Test different distance measures
        distance_measures = vector_db.get_distance_measures()
        for name, measure in distance_measures.items():
            results = vector_db.search_by_text("I think fruit is awesome!", k=2, distance_measure=measure)
            print(f"Results with {name}: {results}")
        
        print("✅ Sentence Transformers test passed!")
        
    except ImportError as e:
        print(f"❌ Sentence Transformers not available: {e}")
    except Exception as e:
        print(f"❌ Sentence Transformers test failed: {e}")
    
    # Test with metadata
    try:
        print("\nTesting with metadata...")
        metadata_list = [
            {"source": "food", "category": "vegetables"},
            {"source": "food", "category": "smoothie"},
            {"source": "animals", "category": "pets"},
            {"source": "animals", "category": "pets"},
            {"source": "animals", "category": "pets"}
        ]
        
        vector_db = EnhancedVectorDatabase(embedding_model)
        vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text, metadata_list))
        
        # Test metadata filtering
        query_vector = embedding_model.get_embedding("I love animals!")
        results = vector_db.search_with_metadata(
            query_vector, 
            k=3, 
            metadata_filter={"source": "animals"}
        )
        print(f"Filtered results (animals only): {results}")
        
        print("✅ Metadata test passed!")
        
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")
