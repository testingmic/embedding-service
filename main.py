#!/usr/bin/env python3
"""
Local Transcription Service - Main Entry Point
HTTP server that provides audio transcription only
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

from services.transcription_service import TranscriptionService
from handlers.transcription_handler import TranscriptionHandler
from utils.memory_tracker import get_memory_usage
from utils.multipart_parser import parse_multipart_form_data


class APIHandler(BaseHTTPRequestHandler):
    """Main HTTP request handler that routes requests to appropriate handlers"""
    
    def __init__(self, *args, transcription_handler, **kwargs):
        """Initialize the handler with service instances"""
        self.transcription_handler = transcription_handler
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Suppress default logs, only show our messages"""
        pass
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/transcribe':
            self.transcription_handler.handle_transcribe(self)
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self._handle_health()
        else:
            self.send_error(404, "Endpoint not found")
    
    def _handle_health(self):
        """Handle GET /health - health check endpoint"""
        try:
            memory_stats = get_memory_usage()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'status': 'healthy',
                'service': 'transcription',
                'transcription_available': self.transcription_handler.transcription_service.is_available(),
                'memory': {
                    'process_memory_mb': memory_stats['process_memory_mb'],
                    'system_memory_percent': memory_stats['system_memory_percent']
                }
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
            print("[OK] Health check passed")
            
        except Exception as e:
            print(f"[ERROR] Health check error: {str(e)}")
            self.send_error(500, f"Error: {str(e)}")


def create_handler(transcription_handler):
    """Factory function to create handler with dependencies"""
    def handler(*args, **kwargs):
        return APIHandler(*args, 
                         transcription_handler=transcription_handler,
                         **kwargs)
    return handler


def run_server(port=9876):
    """Start the HTTP server"""
    print("=" * 60)
    print("Local Transcription Service")
    print("=" * 60)
    
    # Initialize services
    print("\n[INIT] Initializing transcription service...")
    transcription_service = TranscriptionService()
    if transcription_service.is_available():
        transcription_service.load_model()
    else:
        print("[ERROR] Transcription service not available!")
        print("[ERROR] Please install: pip install faster-whisper")
        return
    
    # Initialize handlers
    transcription_handler = TranscriptionHandler(
        transcription_service,
        parse_multipart_form_data,
        get_memory_usage
    )
    
    # Create server with handler factory
    handler_class = create_handler(transcription_handler)
    server_address = ('', port)
    httpd = HTTPServer(server_address, handler_class)
    
    # Get initial memory stats
    initial_memory = get_memory_usage()
    
    print(f"\n[OK] Server running on http://localhost:{port}")
    print(f"\n[INFO] Available endpoints:")
    print(f"   GET  /health        - Health check")
    print(f"   POST /transcribe    - Transcribe audio file (multipart/form-data, field: 'audio')")
    print(f"\n[MEMORY] Initial memory usage: {initial_memory['process_memory_mb']} MB")
    print(f"[INFO] Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Shutting down server...")
        final_memory = get_memory_usage()
        print(f"[MEMORY] Final memory usage: {final_memory['process_memory_mb']} MB")
        httpd.shutdown()


if __name__ == '__main__':
    run_server()