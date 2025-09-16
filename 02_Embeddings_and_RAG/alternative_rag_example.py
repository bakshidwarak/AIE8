"""
Complete RAG Application with Alternative Embedding Models

This example demonstrates how to build a RAG application using different
embedding models instead of OpenAI's text-embedding-3-small.
"""

import asyncio
import os
from typing import List, Dict, Any
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.alternative_vectordatabase import EnhancedVectorDatabase
from aimakerspace.openai_utils.alternative_embeddings import EmbeddingModelFactory
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.prompts import SystemRolePrompt, UserRolePrompt


class AlternativeRAGPipeline:
    """
    RAG Pipeline that supports multiple embedding models
    """
    
    def __init__(
        self, 
        embedding_model_type: str = "sentence_transformer",
        embedding_model_kwargs: Dict[str, Any] = None,
        llm_model: str = "gpt-4o-mini",
        response_style: str = "detailed",
        include_scores: bool = False
    ):
        # Initialize embedding model
        embedding_kwargs = embedding_model_kwargs or {}
        self.embedding_model = EmbeddingModelFactory.create_model(
            embedding_model_type, 
            **embedding_kwargs
        )
        
        # Initialize vector database
        self.vector_db = EnhancedVectorDatabase(self.embedding_model)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=llm_model)
        
        # Configuration
        self.response_style = response_style
        self.include_scores = include_scores
        
        # RAG prompts
        self.rag_system_template = """You are a knowledgeable assistant that answers questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Only provide answers when you are confident the context supports your response."""

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
                "response_length": "brief"
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
    
    async def build_knowledge_base(self, documents: List[str], chunk_size: int = 1000) -> None:
        """Build the knowledge base from documents"""
        print(f"Building knowledge base with {len(documents)} documents...")
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size)
        split_documents = text_splitter.split_texts(documents)
        
        print(f"Split into {len(split_documents)} chunks")
        
        # Build vector database
        await self.vector_db.abuild_from_list(split_documents)
        print(f"Vector database built with {self.vector_db.get_vector_count()} vectors")
    
    def query(self, question: str, k: int = 4, **system_kwargs) -> Dict[str, Any]:
        """Query the RAG system"""
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
            "response_length": system_kwargs.get("response_length", "detailed")
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
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }
    
    def compare_models(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Compare different embedding models on the same question"""
        results = {}
        
        # Test different models
        models_to_test = [
            ("sentence_transformer", {"model_name": "all-MiniLM-L6-v2"}),
            ("sentence_transformer", {"model_name": "all-mpnet-base-v2"}),
        ]
        
        for model_type, model_kwargs in models_to_test:
            try:
                # Create temporary embedding model
                temp_embedding_model = EmbeddingModelFactory.create_model(model_type, **model_kwargs)
                temp_vector_db = EnhancedVectorDatabase(temp_embedding_model)
                
                # Rebuild vector database with new model
                print(f"Testing {model_type} with {model_kwargs}...")
                
                # This would require rebuilding the vector database
                # For demo purposes, we'll just show the concept
                results[model_type] = {
                    "status": "would_require_rebuild",
                    "model_kwargs": model_kwargs
                }
                
            except Exception as e:
                results[model_type] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results


async def main():
    """Main function demonstrating the RAG application"""
    
    # Load documents
    print("Loading documents...")
    text_loader = TextFileLoader("data/PMarcaBlogs.txt")
    documents = text_loader.load_documents()
    print(f"Loaded {len(documents)} documents")
    
    # Test different embedding models
    embedding_configs = [
        {
            "type": "sentence_transformer",
            "kwargs": {"model_name": "all-MiniLM-L6-v2"},
            "name": "Sentence Transformers (MiniLM-L6-v2)"
        },
        {
            "type": "sentence_transformer", 
            "kwargs": {"model_name": "all-mpnet-base-v2"},
            "name": "Sentence Transformers (MPNet-base-v2)"
        },
        {
            "type": "huggingface",
            "kwargs": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
            "name": "Hugging Face (MiniLM-L6-v2)"
        }
    ]
    
    test_questions = [
        "What is the Michael Eisner Memorial Weak Executive Problem?",
        "What are the key principles of good management?",
        "How should companies handle competition?"
    ]
    
    for config in embedding_configs:
        try:
            print(f"\n{'='*60}")
            print(f"Testing: {config['name']}")
            print(f"{'='*60}")
            
            # Initialize RAG pipeline
            rag_pipeline = AlternativeRAGPipeline(
                embedding_model_type=config["type"],
                embedding_model_kwargs=config["kwargs"],
                include_scores=True
            )
            
            # Build knowledge base
            await rag_pipeline.build_knowledge_base(documents)
            
            # Test questions
            for question in test_questions:
                print(f"\nQuestion: {question}")
                print("-" * 50)
                
                result = rag_pipeline.query(question, k=3)
                
                print(f"Answer: {result['response']}")
                print(f"Sources found: {result['context_count']}")
                if result['similarity_scores']:
                    print(f"Similarity scores: {result['similarity_scores']}")
                
        except ImportError as e:
            print(f"❌ {config['name']} not available: {e}")
        except Exception as e:
            print(f"❌ {config['name']} failed: {e}")
    
    # Demonstrate model comparison
    print(f"\n{'='*60}")
    print("Model Comparison Demo")
    print(f"{'='*60}")
    
    # Use the first working model for comparison demo
    try:
        rag_pipeline = AlternativeRAGPipeline(
            embedding_model_type="sentence_transformer",
            embedding_model_kwargs={"model_name": "all-MiniLM-L6-v2"},
            include_scores=True
        )
        
        await rag_pipeline.build_knowledge_base(documents)
        
        question = "What is the Michael Eisner Memorial Weak Executive Problem?"
        result = rag_pipeline.query(question, k=3)
        
        print(f"Question: {question}")
        print(f"Answer: {result['response']}")
        print(f"Embedding Model: {result['embedding_model']}")
        print(f"Sources: {result['context_count']}")
        
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    # Set up OpenAI API key if not already set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        print("You can do this by running: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Run the main function
    asyncio.run(main())
