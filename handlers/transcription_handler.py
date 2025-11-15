"""
Handler for transcription endpoints
"""
import json
import tempfile
import os
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.transcription_service import TranscriptionService
    from utils.multipart_parser import parse_multipart_form_data
    from utils.memory_tracker import get_memory_usage


class TranscriptionHandler:
    """Handler for transcription-related endpoints"""
    
    def __init__(self, transcription_service: 'TranscriptionService', 
                 multipart_parser, memory_tracker):
        """
        Initialize the transcription handler.
        
        Args:
            transcription_service: TranscriptionService instance
            multipart_parser: Function to parse multipart form data
            memory_tracker: Function to get memory usage
        """
        self.transcription_service = transcription_service
        self.parse_multipart = multipart_parser
        self.get_memory_usage = memory_tracker
    
    def handle_transcribe(self, handler: BaseHTTPRequestHandler) -> None:
        """Handle POST /transcribe - transcribe audio file"""
        if not self.transcription_service.is_available():
            handler.send_error(503, "Transcriber service not available. Please install faster-whisper: pip install faster-whisper")
            return
        
        try:
            # Check content type
            content_type = handler.headers.get('Content-Type', '')
            if not content_type.startswith('multipart/form-data'):
                handler.send_error(400, "Content-Type must be multipart/form-data")
                return
            
            # Get content length
            content_length_str = handler.headers.get('Content-Length')
            if not content_length_str:
                handler.send_error(400, "Content-Length header required")
                return
            
            content_length = int(content_length_str)
            if content_length == 0:
                handler.send_error(400, "No content provided")
                return
            
            # Parse multipart form data
            form = self.parse_multipart(handler.rfile, content_type, content_length)
            
            # Get the uploaded file
            if 'audio' not in form:
                handler.send_error(400, "No 'audio' file provided in form data")
                return
            
            file_item = form['audio']
            if not file_item.get('filename'):
                handler.send_error(400, "No file uploaded")
                return
            
            filename = file_item['filename']
            file_data = file_item['data']
            
            # Save uploaded file to temporary location
            file_ext = os.path.splitext(filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(file_data)
                tmp_file_path = tmp_file.name
            
            try:
                # Track memory before processing
                memory_before = self.get_memory_usage()
                
                # Transcribe the audio file
                print(f"[TRANSCRIBE] Processing audio file: {filename}")
                transcription_text = self.transcription_service.transcribe(tmp_file_path, language="en")
                
                # Track memory after processing
                memory_after = self.get_memory_usage()
                memory_delta = memory_after['process_memory_mb'] - memory_before['process_memory_mb']
                
                # Send response
                handler.send_response(200)
                handler.send_header('Content-type', 'application/json')
                handler.end_headers()
                
                response = {
                    'transcription': transcription_text,
                    'filename': filename,
                    'memory': {
                        'process_memory_mb': memory_after['process_memory_mb'],
                        'memory_delta_mb': round(memory_delta, 2),
                        'system_memory_percent': memory_after['system_memory_percent']
                    }
                }
                
                handler.wfile.write(json.dumps(response).encode('utf-8'))
                print(f"[OK] Successfully transcribed audio file")
                print(f"[MEMORY] {memory_after['process_memory_mb']} MB (delta: {round(memory_delta, 2)} MB)")
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            handler.send_error(500, f"Error: {str(e)}")