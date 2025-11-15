"""
Embedding service for generating text embeddings
"""
from sentence_transformers import SentenceTransformer
from typing import List, Optional


class EmbeddingService:
    """Service for generating text embeddings using SentenceTransformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
    
    def initialize(self) -> None:
        """Load the embedding model (lazy initialization)"""
        if self.model is None:
            print(f"🔄 Loading embedding model '{self.model_name}' (this takes ~10 seconds on first run)...")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Embedding model loaded successfully!")
            print(f"📊 Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
    
    def encode_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            normalize: Whether to normalize embeddings
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.initialize()
        
        embeddings = self.model.encode(texts, normalize_embeddings=normalize)
        return embeddings.tolist()
    
    def encode_single(self, text: str, normalize: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            normalize: Whether to normalize embedding
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            self.initialize()
        
        embedding = self.model.encode(text, normalize_embeddings=normalize)
        return embedding.tolist()
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {
                "model": self.model_name,
                "loaded": False,
                "dimensions": None
            }
        
        return {
            "model": self.model_name,
            "loaded": True,
            "dimensions": self.model.get_sentence_embedding_dimension()
        }

