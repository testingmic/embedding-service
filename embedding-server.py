#!/usr/bin/env python3
"""
Local Embedding Service for Mac
Simple HTTP server that generates text embeddings
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from sentence_transformers import SentenceTransformer
import json
import tempfile
import os
import re

# Try to import Transcriber, fallback to openai-whisper if not available
TRANSCRIBER_AVAILABLE = False
WHISPER_AVAILABLE = False
transcribe_func = None

try:
    from Transcriber.transcriber import transcribe
    transcribe_func = transcribe
    TRANSCRIBER_AVAILABLE = True
    print("✅ Transcriber package available")
except ImportError:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        TRANSCRIBER_AVAILABLE = True  # Mark as available since we have whisper
        print("✅ Using openai-whisper for transcription (Transcriber package not available)")
    except ImportError:
        print("⚠️  Warning: Neither Transcriber nor openai-whisper available. /transcribe endpoint will not work.")

class EmbeddingHandler(BaseHTTPRequestHandler):
    model = None
    whisper_model = None
    
    @classmethod
    def initialize_model(cls):
        if cls.model is None:
            print("🔄 Loading embedding model (this takes ~10 seconds on first run)...")
            cls.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully!")
            print(f"📊 Embedding dimensions: {cls.model.get_sentence_embedding_dimension()}")
    
    @classmethod
    def initialize_whisper(cls):
        if WHISPER_AVAILABLE and not transcribe_func and cls.whisper_model is None:
            print("🔄 Loading Whisper model (this may take a minute on first run)...")
            import whisper
            cls.whisper_model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully!")
    
    def log_message(self, format, *args):
        # Suppress default logs, only show our messages
        pass
    
    def parse_multipart_form_data(self, content_type, content_length):
        """Parse multipart/form-data request body."""
        boundary = None
        if 'boundary=' in content_type:
            boundary = content_type.split('boundary=')[1].strip()
            if boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]
        
        if not boundary:
            raise ValueError("No boundary found in Content-Type")
        
        # Read the request body
        body = self.rfile.read(content_length)
        
        # Split by boundary
        parts = body.split(f'--{boundary}'.encode())
        
        form_data = {}
        for part in parts:
            if not part.strip() or part.strip() == b'--':
                continue
            
            # Split headers and body
            if b'\r\n\r\n' in part:
                headers_data, body_data = part.split(b'\r\n\r\n', 1)
            elif b'\n\n' in part:
                headers_data, body_data = part.split(b'\n\n', 1)
            else:
                continue
            
            # Parse headers
            content_disposition = headers_data.decode('utf-8', errors='ignore')
            
            # Extract field name and filename
            field_name = None
            filename = None
            
            if 'Content-Disposition: form-data' in content_disposition:
                name_match = re.search(r'name="([^"]+)"', content_disposition)
                if name_match:
                    field_name = name_match.group(1)
                
                filename_match = re.search(r'filename="([^"]+)"', content_disposition)
                if filename_match:
                    filename = filename_match.group(1)
            
            if field_name:
                # Remove trailing boundary markers and newlines from body
                body_data = body_data.rstrip(b'\r\n--')
                
                form_data[field_name] = {
                    'filename': filename,
                    'data': body_data
                }
        
        return form_data
    
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
            if not TRANSCRIBER_AVAILABLE:
                self.send_error(503, "Transcriber service not available. Please install Transcriber or openai-whisper package.")
                return
            
            try:
                # Check content type
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data'):
                    self.send_error(400, "Content-Type must be multipart/form-data")
                    return
                
                # Get content length
                content_length_str = self.headers.get('Content-Length')
                if not content_length_str:
                    self.send_error(400, "Content-Length header required")
                    return
                
                content_length = int(content_length_str)
                if content_length == 0:
                    self.send_error(400, "No content provided")
                    return
                
                # Parse multipart form data
                form = self.parse_multipart_form_data(content_type, content_length)
                
                # Get the uploaded file
                if 'audio' not in form:
                    self.send_error(400, "No 'audio' file provided in form data")
                    return
                
                file_item = form['audio']
                if not file_item.get('filename'):
                    self.send_error(400, "No file uploaded")
                    return
                
                filename = file_item['filename']
                file_data = file_item['data']
                
                # Save uploaded file to temporary location
                file_ext = os.path.splitext(filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(file_data)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Transcribe the audio file
                    print(f"🎤 Transcribing audio file: {filename}")
                    
                    if WHISPER_AVAILABLE and not transcribe_func:
                        # Use openai-whisper directly
                        if self.whisper_model is None:
                            self.initialize_whisper()
                        result = self.whisper_model.transcribe(tmp_file_path, language="en")
                        transcription_text = result["text"].strip()
                    else:
                        # Use Transcriber package
                        with tempfile.TemporaryDirectory() as output_dir:
                            transcribe_func(
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
                        'filename': filename
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
    
    # Initialize models before starting server
    EmbeddingHandler.initialize_model()
    if WHISPER_AVAILABLE and not transcribe_func:
        EmbeddingHandler.initialize_whisper()
    
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