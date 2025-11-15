#!/home/syywrhsb/virtualenv/public/transcriber/3.11/bin/python
"""
WSGI adapter for cPanel/Passenger deployment
"""
import sys
import os

# Get the directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add cPanel virtual environment to path
VENV_SITE_PACKAGES = '/home/syywrhsb/virtualenv/public/transcriber/3.11/lib/python3.11/site-packages'
if VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, VENV_SITE_PACKAGES)

# Add application directory
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from io import BytesIO
from urllib.parse import parse_qs
import json

# Import your services and handlers
from services.embedding_service import EmbeddingService
from services.transcription_service import TranscriptionService
from handlers.embedding_handler import EmbeddingHandler
from handlers.transcription_handler import TranscriptionHandler
from utils.memory_tracker import get_memory_usage
from utils.multipart_parser import parse_multipart_form_data

# Initialize services globally (loaded once when app starts)
print("Initializing services...")
embedding_service = EmbeddingService()
embedding_service.initialize()

transcription_service = TranscriptionService()
if transcription_service.is_available():
    transcription_service.load_model()

# Initialize handlers
embedding_handler = EmbeddingHandler(embedding_service, get_memory_usage)
transcription_handler = TranscriptionHandler(
    transcription_service,
    parse_multipart_form_data,
    get_memory_usage
)
print("Services initialized successfully!")


class WSGIRequest:
    """Mock request object that mimics BaseHTTPRequestHandler interface"""
    def __init__(self, environ):
        self.environ = environ
        self.headers = {}
        self.path = environ.get('PATH_INFO', '/')
        self.response_status = None
        self.response_headers = []
        self.response_body = BytesIO()
        
        # Extract headers
        for key, value in environ.items():
            if key.startswith('HTTP_'):
                header_name = key[5:].replace('_', '-').title()
                self.headers[header_name] = value
            elif key == 'CONTENT_TYPE':
                self.headers['Content-Type'] = value
            elif key == 'CONTENT_LENGTH':
                self.headers['Content-Length'] = value
        
        # Setup input stream
        self.rfile = environ['wsgi.input']
    
    def send_response(self, code):
        self.response_status = code
    
    def send_header(self, key, value):
        self.response_headers.append((key, value))
    
    def end_headers(self):
        pass
    
    def send_error(self, code, message):
        self.response_status = code
        self.response_headers = [('Content-Type', 'application/json')]
        error_response = json.dumps({'error': message})
        self.response_body.write(error_response.encode('utf-8'))
    
    @property
    def wfile(self):
        return self.response_body


def application(environ, start_response):
    """
    WSGI application entry point
    This is what Passenger calls to handle each request
    """
    request = WSGIRequest(environ)
    method = environ.get('REQUEST_METHOD', 'GET')
    
    try:
        # Route requests based on method and path
        if method == 'POST':
            if request.path == '/embed':
                embedding_handler.handle_embed_batch(request)
            elif request.path == '/embed_single':
                embedding_handler.handle_embed_single(request)
            elif request.path == '/transcribe':
                transcription_handler.handle_transcribe(request)
            else:
                request.send_error(404, "Endpoint not found")
        
        elif method == 'GET':
            if request.path == '/' or request.path == '/health':
                handle_health(request)
            else:
                request.send_error(404, "Endpoint not found")
        else:
            request.send_error(405, "Method not allowed")
        
        # Prepare response
        status_code = request.response_status or 200
        status_text = {
            200: 'OK',
            400: 'Bad Request',
            404: 'Not Found',
            405: 'Method Not Allowed',
            500: 'Internal Server Error',
            503: 'Service Unavailable'
        }.get(status_code, 'Unknown')
        
        status = f'{status_code} {status_text}'
        headers = request.response_headers or [('Content-Type', 'application/json')]
        
        start_response(status, headers)
        return [request.response_body.getvalue()]
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        status = '500 Internal Server Error'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        error_response = json.dumps({'error': str(e)})
        return [error_response.encode('utf-8')]


def handle_health(request):
    """Handle GET /health - health check endpoint"""
    try:
        model_info = embedding_service.get_model_info()
        memory_stats = get_memory_usage()
        
        request.send_response(200)
        request.send_header('Content-Type', 'application/json')
        request.end_headers()
        
        response = {
            'status': 'healthy',
            'model': model_info['model'],
            'dimensions': model_info['dimensions'],
            'transcription_available': transcription_service.is_available(),
            'memory': {
                'process_memory_mb': memory_stats['process_memory_mb'],
                'system_memory_percent': memory_stats['system_memory_percent']
            }
        }
        
        request.wfile.write(json.dumps(response).encode('utf-8'))
        print("Health check passed")
        
    except Exception as e:
        print(f"Health check error: {str(e)}")
        request.send_error(500, f"Error: {str(e)}")