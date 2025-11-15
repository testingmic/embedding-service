#!/usr/bin/env python3
"""
Passenger WSGI adapter for the transcription service (transcription only)
"""
import os
import sys
import time
from io import BytesIO

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'

# Increase Passenger timeouts
os.environ['PASSENGER_STARTUP_TIMEOUT'] = '300'
os.environ['PASSENGER_MAX_REQUEST_TIME'] = '300'

# Add the application directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Log startup
print("[STARTUP] Starting transcription service via Passenger", flush=True)
start_time = time.time()

try:
    from services.transcription_service import TranscriptionService
    from handlers.transcription_handler import TranscriptionHandler
    from utils.memory_tracker import get_memory_usage
    from utils.multipart_parser import parse_multipart_form_data
    
    print("[STARTUP] Imports successful", flush=True)
    
    # Initialize transcription service only
    print("[STARTUP] Initializing transcription service...", flush=True)
    print("[INFO] Using 'tiny' model with lazy loading (loads on first request)...", flush=True)
    transcription_service = TranscriptionService(model_size="tiny")
    
    # DO NOT pre-load the model - let it load on first request
    # if transcription_service.is_available():
    #     transcription_service.load_model()  # <-- REMOVED
    
    # Initialize handler
    transcription_handler = TranscriptionHandler(
        transcription_service,
        parse_multipart_form_data,
        get_memory_usage
    )
    
    elapsed = time.time() - start_time
    print(f"[STARTUP] Services initialized in {elapsed:.2f}s", flush=True)
    
    # Get initial memory
    memory = get_memory_usage()
    print(f"[STARTUP] Memory usage: {memory['process_memory_mb']} MB", flush=True)
    
except Exception as e:
    print(f"[ERROR] Failed to initialize services: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise


class WSGIRequestHandler:
    """Adapter to convert WSGI environ to our handler format"""
    
    def __init__(self, environ):
        self.environ = environ
        self.path = environ.get('PATH_INFO', '/')
        self.command = environ.get('REQUEST_METHOD', 'GET')
        
        # Create a file-like object for reading the request body
        try:
            content_length = int(environ.get('CONTENT_LENGTH', 0))
        except ValueError:
            content_length = 0
        
        self.rfile = environ['wsgi.input']
        self.content_length = content_length
        
        # Headers dict
        self.headers = {}
        for key, value in environ.items():
            if key.startswith('HTTP_'):
                header_name = key[5:].replace('_', '-').title()
                self.headers[header_name] = value
            elif key in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
                header_name = key.replace('_', '-').title()
                self.headers[header_name] = value
        
        # Response data
        self.response_status = '200 OK'
        self.response_headers = []
        self.response_body = BytesIO()
    
    def send_response(self, code):
        """Set response status code"""
        status_messages = {
            200: 'OK',
            400: 'Bad Request',
            404: 'Not Found',
            500: 'Internal Server Error',
            503: 'Service Unavailable'
        }
        message = status_messages.get(code, 'Unknown')
        self.response_status = f'{code} {message}'
    
    def send_header(self, key, value):
        """Add a response header"""
        self.response_headers.append((key, str(value)))
    
    def end_headers(self):
        """Finalize headers (no-op for WSGI)"""
        pass
    
    def send_error(self, code, message):
        """Send an error response"""
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(f'Error {code}: {message}'.encode('utf-8'))
    
    @property
    def wfile(self):
        """File-like object for writing response"""
        return self.response_body


def application(environ, start_response):
    """WSGI application entry point"""
    request_start = time.time()
    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')
    
    print(f"[REQUEST] {method} {path}", flush=True)
    
    try:
        # Create our request handler wrapper
        handler = WSGIRequestHandler(environ)
        
        # Route the request
        if method == 'GET' and path == '/health':
            # Health check
            try:
                memory_stats = get_memory_usage()
                
                import json
                response_data = {
                    'status': 'healthy',
                    'service': 'transcription',
                    'transcription_available': transcription_service.is_available(),
                    'memory': {
                        'process_memory_mb': memory_stats['process_memory_mb'],
                        'system_memory_percent': memory_stats['system_memory_percent']
                    }
                }
                
                handler.send_response(200)
                handler.send_header('Content-Type', 'application/json')
                handler.end_headers()
                handler.wfile.write(json.dumps(response_data).encode('utf-8'))
                
            except Exception as e:
                print(f"[ERROR] Health check failed: {e}", flush=True)
                handler.send_error(500, str(e))
            
        elif method == 'POST' and path == '/transcribe':
            print("[REQUEST] Processing /transcribe request...", flush=True)
            transcription_handler.handle_transcribe(handler)
            
        else:
            handler.send_error(404, f"Endpoint not found: {method} {path}")
        
        # Get response
        status = handler.response_status
        headers = handler.response_headers
        body = handler.response_body.getvalue()
        
        # Log request completion
        elapsed = time.time() - request_start
        print(f"[REQUEST] Completed {method} {path} in {elapsed:.2f}s", flush=True)
        
        # Send WSGI response
        start_response(status, headers)
        return [body]
    
    except Exception as e:
        print(f"[ERROR] Request failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        # Return error response
        start_response('500 Internal Server Error', [('Content-Type', 'text/plain')])
        return [f'Internal Server Error: {str(e)}'.encode('utf-8')]