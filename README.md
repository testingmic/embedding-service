# Embedding Service

A local HTTP service that provides text embeddings and audio transcription capabilities.

## Project Structure

```
embedding-service/
├── main.py                 # Entry point - starts the HTTP server
├── handlers/               # Request handlers
│   ├── __init__.py
│   ├── embedding_handler.py    # Handles /embed and /embed_single endpoints
│   └── transcription_handler.py # Handles /transcribe endpoint
├── services/               # Business logic services
│   ├── __init__.py
│   ├── embedding_service.py    # Embedding model management
│   └── transcription_service.py # Transcription model management
└── utils/                  # Utility functions
    ├── __init__.py
    ├── memory_tracker.py        # Memory usage tracking
    └── multipart_parser.py      # Multipart form data parsing
```

## Features

- **Text Embeddings**: Generate embeddings for single texts or batches
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
pip install sentence-transformers torch numpy psutil

# For transcription (optional):
pip install openai-whisper  # or faster-whisper
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
  "model": "all-MiniLM-L6-v2",
  "dimensions": 384,
  "transcription_available": true,
  "memory": {
    "process_memory_mb": 245.67,
    "system_memory_percent": 45.2
  }
}
```

### POST /embed_single
Generate embedding for a single text.

**Request:**
```json
{
  "text": "Hello, world!"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...],
  "dimensions": 384,
  "memory": {
    "process_memory_mb": 250.12,
    "memory_delta_mb": 4.45,
    "system_memory_percent": 45.5
  }
}
```

### POST /embed
Generate embeddings for multiple texts.

**Request:**
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"]
}
```

**Response:**
```json
{
  "embeddings": [[0.123, ...], [0.456, ...], [0.789, ...]],
  "dimensions": 384,
  "count": 3,
  "memory": {
    "process_memory_mb": 255.23,
    "memory_delta_mb": 5.11,
    "system_memory_percent": 45.8
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

