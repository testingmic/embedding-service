"""
Transcription service for audio transcription
"""
import tempfile
import os
from typing import Optional, Tuple


class TranscriptionService:
    """Service for transcribing audio files"""
    
    def __init__(self):
        """Initialize the transcription service"""
        self.whisper_model = None
        self.model_type: Optional[str] = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Try to initialize available transcription models"""
        # Try faster-whisper first (best compatibility)
        try:
            from faster_whisper import WhisperModel
            self.model_type = "faster_whisper"
            print("✅ Using faster-whisper for transcription")
            return
        except ImportError:
            pass
        
        # Try Transcriber package
        try:
            from Transcriber.transcriber import transcribe as transcriber_transcribe
            self.transcriber_transcribe = transcriber_transcribe
            self.model_type = "transcriber"
            print("✅ Using Transcriber package for transcription")
            return
        except ImportError:
            pass
        
        # Try openai-whisper as last resort
        try:
            import whisper
            self.whisper_module = whisper
            self.model_type = "openai_whisper"
            print("✅ Using openai-whisper for transcription")
            return
        except ImportError:
            pass
        
        self.model_type = None
        print("⚠️  Warning: No transcription library available.")
        print("   Note: Python 3.14 has compatibility issues with transcription libraries.")
        print("   Options:")
        print("   1. Use Python 3.11 or 3.12: python3.11 -m venv venv311")
        print("   2. Try: pip install faster-whisper (may still fail due to onnxruntime)")
        print("   3. Use an API-based transcription service")
    
    def load_model(self) -> None:
        """Load the transcription model (lazy initialization)"""
        if self.model_type == "faster_whisper" and self.whisper_model is None:
            print("🔄 Loading faster-whisper model (this may take a minute on first run)...")
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("✅ faster-whisper model loaded successfully!")
        elif self.model_type == "openai_whisper" and self.whisper_model is None:
            print("🔄 Loading Whisper model (this may take a minute on first run)...")
            self.whisper_model = self.whisper_module.load_model("base")
            print("✅ Whisper model loaded successfully!")
    
    def transcribe(self, audio_file_path: str, language: str = "en") -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code (default: "en")
            
        Returns:
            Transcribed text
            
        Raises:
            Exception: If no transcription model is available or transcription fails
        """
        if self.model_type is None:
            raise Exception("No transcription model available. Please install a transcription library.")
        
        if self.model_type == "faster_whisper":
            if self.whisper_model is None:
                self.load_model()
            segments, info = self.whisper_model.transcribe(audio_file_path, language=language)
            return " ".join([segment.text for segment in segments]).strip()
        
        elif self.model_type == "openai_whisper":
            if self.whisper_model is None:
                self.load_model()
            result = self.whisper_model.transcribe(audio_file_path, language=language)
            return result["text"].strip()
        
        elif self.model_type == "transcriber":
            with tempfile.TemporaryDirectory() as output_dir:
                self.transcriber_transcribe(
                    urls_or_paths=[audio_file_path],
                    output_dir=output_dir,
                    output_formats=["txt"],
                    language=language
                )
                
                # Read the transcription result
                base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
                output_file = os.path.join(output_dir, f"{base_name}.txt")
                
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    # Try to find any .txt file in the output directory
                    txt_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
                    if txt_files:
                        with open(os.path.join(output_dir, txt_files[0]), 'r', encoding='utf-8') as f:
                            return f.read()
                    else:
                        raise Exception("Transcription output file not found")
        
        else:
            raise Exception("No transcription model available")
    
    def is_available(self) -> bool:
        """Check if transcription service is available"""
        return self.model_type is not None

