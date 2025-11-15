# Transcription Service

A local HTTP service that provides audio transcription capabilities.

## Project Structure

```
transcription-service/
├── main.py                 # Entry point - starts the HTTP server
├── handlers/               # Request handlers
│   ├── __init__.py
│   └── transcription_handler.py # Handles /transcribe endpoint
├── services/               # Business logic services
│   ├── __init__.py
│   └── transcription_service.py # Transcription model management
└── utils/                  # Utility functions
    ├── __init__.py
    ├── memory_tracker.py        # Memory usage tracking
    └── multipart_parser.py      # Multipart form data parsing
```

## Features

- **Audio Transcription**: Transcribe audio files using Whisper models
- **Memory Tracking**: All responses include memory usage statistics
- **Clean Architecture**: Separated concerns with handlers, services, and utilities

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv embeddings_env
source embeddings_env/bin/activate  # On Windows: embeddings_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt

# Or manually:
pip install psutil openai-whisper  # or faster-whisper
```

## Usage

Run the server:
```bash
python main.py
```

The server will start on `http://localhost:9876`

## API Endpoints

### GET /health
Health check endpoint that returns server status and memory usage.

**Response:**
```json
{
  "status": "healthy",
  "transcription_available": true,
  "memory": {
    "process_memory_mb": 245.67,
    "system_memory_percent": 45.2
  }
}
```

### POST /transcribe
Transcribe an audio file.

**Request:**
- Content-Type: `multipart/form-data`
- Field name: `audio`
- File: Audio file (mp3, wav, etc.)

**Response:**
```json
{
  "transcription": "The transcribed text here...",
  "filename": "audio.mp3",
  "memory": {
    "process_memory_mb": 320.45,
    "memory_delta_mb": 65.22,
    "system_memory_percent": 48.3
  }
}
```

## Memory Tracking

All endpoints include memory usage information in their responses:
- `process_memory_mb`: Current process memory usage in MB
- `memory_delta_mb`: Change in memory from before to after processing
- `system_memory_percent`: Overall system memory usage percentage

## Architecture

The project follows a clean architecture pattern:

- **Handlers**: Handle HTTP requests/responses and route to services
- **Services**: Contain business logic and model management
- **Utils**: Reusable utility functions (memory tracking, parsing, etc.)
- **Main**: Entry point that initializes services and starts the server

This separation makes the code:
- Easier to test
- Easier to maintain
- Easier to extend with new features

