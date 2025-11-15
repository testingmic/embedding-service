"""
Handler for embedding endpoints
"""
import json
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.embedding_service import EmbeddingService
    from utils.memory_tracker import get_memory_usage


class EmbeddingHandler:
    """Handler for embedding-related endpoints"""
    
    def __init__(self, embedding_service: 'EmbeddingService', memory_tracker):
        """
        Initialize the embedding handler.
        
        Args:
            embedding_service: EmbeddingService instance
            memory_tracker: Function to get memory usage
        """
        self.embedding_service = embedding_service
        self.get_memory_usage = memory_tracker
    
    def handle_embed_batch(self, handler: BaseHTTPRequestHandler) -> None:
        """Handle POST /embed - batch embeddings"""
        try:
            content_length = int(handler.headers['Content-Length'])
            post_data = handler.rfile.read(content_length)
            
            data = json.loads(post_data.decode('utf-8'))
            texts = data.get('texts', [])
            
            if not texts:
                handler.send_error(400, "No texts provided")
                return
            
            # Track memory before processing
            memory_before = self.get_memory_usage()
            
            # Generate embeddings
            embeddings_list = self.embedding_service.encode_batch(texts, normalize=True)
            
            # Track memory after processing
            memory_after = self.get_memory_usage()
            memory_delta = memory_after['process_memory_mb'] - memory_before['process_memory_mb']
            
            # Send response
            handler.send_response(200)
            handler.send_header('Content-type', 'application/json')
            handler.end_headers()
            
            response = {
                'embeddings': embeddings_list,
                'dimensions': len(embeddings_list[0]),
                'count': len(embeddings_list),
                'memory': {
                    'process_memory_mb': memory_after['process_memory_mb'],
                    'memory_delta_mb': round(memory_delta, 2),
                    'system_memory_percent': memory_after['system_memory_percent']
                }
            }
            
            handler.wfile.write(json.dumps(response).encode('utf-8'))
            print(f"✅ Successfully generated {len(embeddings_list)} embeddings")
            print(f"📊 Memory: {memory_after['process_memory_mb']} MB (Δ {round(memory_delta, 2)} MB)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            handler.send_error(500, f"Error: {str(e)}")
    
    def handle_embed_single(self, handler: BaseHTTPRequestHandler) -> None:
        """Handle POST /embed_single - single embedding"""
        try:
            content_length = int(handler.headers['Content-Length'])
            post_data = handler.rfile.read(content_length)
            
            data = json.loads(post_data.decode('utf-8'))
            text = data.get('text', '')
            
            if not text:
                handler.send_error(400, "No text provided")
                return
            
            # Track memory before processing
            memory_before = self.get_memory_usage()
            
            # Generate embedding
            embedding = self.embedding_service.encode_single(text, normalize=True)
            
            # Track memory after processing
            memory_after = self.get_memory_usage()
            memory_delta = memory_after['process_memory_mb'] - memory_before['process_memory_mb']
            
            handler.send_response(200)
            handler.send_header('Content-type', 'application/json')
            handler.end_headers()
            
            response = {
                'embedding': embedding,
                'dimensions': len(embedding),
                'memory': {
                    'process_memory_mb': memory_after['process_memory_mb'],
                    'memory_delta_mb': round(memory_delta, 2),
                    'system_memory_percent': memory_after['system_memory_percent']
                }
            }
            
            handler.wfile.write(json.dumps(response).encode('utf-8'))
            print(f"✅ Successfully generated embedding")
            print(f"📊 Memory: {memory_after['process_memory_mb']} MB (Δ {round(memory_delta, 2)} MB)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            handler.send_error(500, f"Error: {str(e)}")

