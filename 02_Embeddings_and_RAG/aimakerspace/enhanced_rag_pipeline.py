"""
Enhanced RAG Pipeline with support for multiple embedding models
"""

import asyncio
from typing import List, Dict, Any, Union, Optional
from aimakerspace.openai_utils.base_embedding import BaseEmbeddingModel
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.openai_utils.alternative_embeddings import EmbeddingModelFactory
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.prompts import SystemRolePrompt, UserRolePrompt
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter


class EnhancedRAGPipeline:
    """
    Enhanced RAG Pipeline that supports multiple embedding models
    and provides flexible configuration options.
    """
    
    def __init__(
        self,
        embedding_model_type: str = "openai",
        embedding_model_kwargs: Dict[str, Any] = None,
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        response_style: str = "detailed",
        include_scores: bool = False,
        distance_metric: str = "cosine_similarity"
    ):
        """
        Initialize the Enhanced RAG Pipeline
        
        Args:
            embedding_model_type: Type of embedding model to use
                Options: "openai", "sentence_transformer", "cohere", "huggingface"
            embedding_model_kwargs: Additional arguments for the embedding model
            llm_model: LLM model name for generation
            chunk_size: Size of text chunks for document splitting
            chunk_overlap: Overlap between chunks
            response_style: Style of response ("detailed", "concise", "brief")
            include_scores: Whether to include similarity scores in output
            distance_metric: Distance metric for similarity search
        """
        # Initialize embedding model
        embedding_kwargs = embedding_model_kwargs or {}
        self.embedding_model = EmbeddingModelFactory.create_model(
            embedding_model_type, 
            **embedding_kwargs
        )
        
        # Initialize vector database
        self.vector_db = VectorDatabase(self.embedding_model)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=llm_model)
        
        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.response_style = response_style
        self.include_scores = include_scores
        self.distance_metric = distance_metric
        
        # Initialize text splitter
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Setup RAG prompts
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup the RAG system and user prompts"""
        self.rag_system_template = """You are a knowledgeable assistant that answers questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Only provide answers when you are confident the context supports your response.
- When citing sources, use the format [Source X] where X is the source number."""

        self.rag_user_template = """Context Information:
{context}

Number of relevant sources found: {context_count}
{similarity_scores}

Question: {user_query}

Please provide your answer based solely on the context above."""

        self.system_prompt = SystemRolePrompt(
            self.rag_system_template,
            strict=True,
            defaults={
                "response_style": self.response_style,
                "response_length": "detailed"
            }
        )

        self.user_prompt = UserRolePrompt(
            self.rag_user_template,
            strict=True,
            defaults={
                "context_count": "",
                "similarity_scores": ""
            }
        )
    
    async def load_documents(self, file_path: str) -> List[str]:
        """Load documents from a file"""
        print(f"Loading documents from {file_path}...")
        text_loader = TextFileLoader(file_path)
        documents = text_loader.load_documents()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    async def build_knowledge_base(
        self, 
        documents: List[str], 
        metadata: List[Dict[str, Any]] = None
    ) -> None:
        """Build the knowledge base from documents"""
        print(f"Building knowledge base with {len(documents)} documents...")
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_texts(documents)
        print(f"Split into {len(split_documents)} chunks")
        
        # Build vector database
        await self.vector_db.abuild_from_list(split_documents)
        print(f"Vector database built with {self.vector_db.get_vector_count()} vectors")
        print(f"Using embedding model: {type(self.embedding_model).__name__}")
    
    def query(
        self, 
        question: str, 
        k: int = 4, 
        response_length: str = "detailed",
        **system_kwargs
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            k: Number of relevant chunks to retrieve
            response_length: Length of response ("brief", "detailed", "comprehensive")
            **system_kwargs: Additional system prompt parameters
            
        Returns:
            Dictionary containing response, context, and metadata
        """
        # Retrieve relevant contexts
        context_list = self.vector_db.search_by_text(question, k=k)
        
        # Format context
        context_prompt = ""
        similarity_scores = []
        
        for i, (context, score) in enumerate(context_list, 1):
            context_prompt += f"[Source {i}]: {context}\n\n"
            similarity_scores.append(f"Source {i}: {score:.3f}")
        
        # Create system message with parameters
        system_params = {
            "response_style": self.response_style,
            "response_length": response_length,
            **system_kwargs
        }
        
        formatted_system_prompt = self.system_prompt.create_message(**system_params)
        
        # Create user message
        user_params = {
            "user_query": question,
            "context": context_prompt.strip(),
            "context_count": len(context_list),
            "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
        }
        
        formatted_user_prompt = self.user_prompt.create_message(**user_params)
        
        # Generate response
        response = self.llm.run([formatted_system_prompt, formatted_user_prompt])
        
        return {
            "response": response,
            "context": context_list,
            "context_count": len(context_list),
            "similarity_scores": similarity_scores if self.include_scores else None,
            "embedding_model": type(self.embedding_model).__name__,
            "llm_model": self.llm.model_name,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current models"""
        return {
            "embedding_model": type(self.embedding_model).__name__,
            "embedding_model_name": getattr(self.embedding_model, 'model_name', 'Unknown'),
            "llm_model": self.llm.model_name,
            "vector_count": self.vector_db.get_vector_count(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "response_style": self.response_style
        }