"""
Multipart form data parser for handling file uploads
"""
import re
from typing import Dict, Optional


def parse_multipart_form_data(rfile, content_type: str, content_length: int) -> Dict[str, Dict]:
    """
    Parse multipart/form-data request body.
    
    Args:
        rfile: File-like object to read from
        content_type: Content-Type header value
        content_length: Content-Length header value
        
    Returns:
        Dictionary mapping field names to file data:
        {
            'field_name': {
                'filename': 'example.mp3',
                'data': b'...'
            }
        }
    """
    boundary = None
    if 'boundary=' in content_type:
        boundary = content_type.split('boundary=')[1].strip()
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]
    
    if not boundary:
        raise ValueError("No boundary found in Content-Type")
    
    # Read the request body
    body = rfile.read(content_length)
    
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

