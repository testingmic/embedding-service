#!/usr/bin/env python3
"""
Local Embedding Service for Mac
Simple HTTP server that generates text embeddings
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from sentence_transformers import SentenceTransformer
import json
import sys
import tempfile
import os
import cgi
from Transcriber.transcriber import transcribe

class EmbeddingHandler(BaseHTTPRequestHandler):
    model = None
    
    @classmethod
    def initialize_model(cls):
        if cls.model is None:
            print("🔄 Loading model (this takes ~10 seconds on first run)...")
            cls.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model loaded successfully!")
            print(f"📊 Embedding dimensions: {cls.model.get_sentence_embedding_dimension()}")
    
    def log_message(self, format, *args):
        # Suppress default logs, only show our messages
        pass
    
    def do_POST(self):
        if self.path == '/embed':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                texts = data.get('texts', [])
                
                if not texts:
                    self.send_error(400, "No texts provided")
                    return
                
                # Generate embeddings
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                embeddings_list = embeddings.tolist()
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    'embeddings': embeddings_list,
                    'dimensions': len(embeddings_list[0]),
                    'count': len(embeddings_list)
                }
                
                self.wfile.write(json.dumps(response).encode('utf-8'))
                print(f"✅ Successfully generated {len(embeddings_list)} embeddings")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                self.send_error(500, f"Error: {str(e)}")
        
        elif self.path == '/embed_single':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                text = data.get('text', '')
                
                if not text:
                    self.send_error(400, "No text provided")
                    return
                
                # Generate embedding
                embedding = self.model.encode(text, normalize_embeddings=True)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    'embedding': embedding.tolist(),
                    'dimensions': len(embedding)
                }
                
                self.wfile.write(json.dumps(response).encode('utf-8'))
                print(f"✅ Successfully generated embedding")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                self.send_error(500, f"Error: {str(e)}")
        
        elif self.path == '/transcribe':
            try:
                # Check content type
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data'):
                    self.send_error(400, "Content-Type must be multipart/form-data")
                    return
                
                # Parse multipart form data
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST'}
                )
                
                # Get the uploaded file
                if 'audio' not in form:
                    self.send_error(400, "No 'audio' file provided in form data")
                    return
                
                file_item = form['audio']
                if not file_item.filename:
                    self.send_error(400, "No file uploaded")
                    return
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_item.filename)[1]) as tmp_file:
                    tmp_file.write(file_item.file.read())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Create temporary output directory for transcription
                    with tempfile.TemporaryDirectory() as output_dir:
                        # Transcribe the audio file
                        print(f"🎤 Transcribing audio file: {file_item.filename}")
                        transcribe(
                            urls_or_paths=[tmp_file_path],
                            output_dir=output_dir,
                            output_formats=["txt"],
                            language="en"
                        )
                        
                        # Read the transcription result
                        # The output file will be named based on the input file
                        base_name = os.path.splitext(os.path.basename(tmp_file_path))[0]
                        output_file = os.path.join(output_dir, f"{base_name}.txt")
                        
                        if os.path.exists(output_file):
                            with open(output_file, 'r', encoding='utf-8') as f:
                                transcription_text = f.read()
                        else:
                            # Try to find any .txt file in the output directory
                            txt_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
                            if txt_files:
                                with open(os.path.join(output_dir, txt_files[0]), 'r', encoding='utf-8') as f:
                                    transcription_text = f.read()
                            else:
                                raise Exception("Transcription output file not found")
                        
                        # Send response
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        response = {
                            'transcription': transcription_text,
                            'filename': file_item.filename
                        }
                        
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                        print(f"✅ Successfully transcribed audio file")
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                self.send_error(500, f"Error: {str(e)}")
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'status': 'healthy',
                'model': 'all-MiniLM-L6-v2',
                'dimensions': 384
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
            print("💚 Health check passed")
        else:
            self.send_error(404, "Endpoint not found")

def run_server(port=9876):
    print("=" * 60)
    print("🚀 Local Embedding Service")
    print("=" * 60)
    
    # Initialize model before starting server
    EmbeddingHandler.initialize_model()
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, EmbeddingHandler)
    
    print(f"\n✅ Server running on http://localhost:{port}")
    print(f"\n📚 Available endpoints:")
    print(f"   GET  /health        - Health check")
    print(f"   POST /embed_single  - Generate single embedding")
    print(f"   POST /embed         - Generate batch embeddings")
    print(f"   POST /transcribe    - Transcribe audio file (multipart/form-data, field: 'audio')")
    print(f"\n💡 Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()