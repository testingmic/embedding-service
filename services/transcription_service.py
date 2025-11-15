"""
Transcription service for audio transcription with LAZY loading
Model is only loaded when first transcription is requested
"""
import tempfile
import os
from typing import Optional


class TranscriptionService:
    """Service for transcribing audio files with lazy loading"""
    
    def __init__(self, model_size: str = "tiny"):
        """
        Initialize the transcription service.
        
        Args:
            model_size: Whisper model size - 'tiny', 'base', 'small', 'medium', 'large'
                       tiny: ~40MB, fastest, lowest memory (~150MB total)
                       base: ~75MB, good balance (~250MB total)
                       small: ~244MB, better accuracy (~400MB total)
        """
        self.whisper_model = None
        self.model_size = model_size
        self.model_type: Optional[str] = None
        self._check_available_libraries()
    
    # Keep backward compatibility for old code
    @classmethod
    def create_default(cls):
        """Create service with default settings (for backward compatibility)"""
        return cls(model_size="tiny")
    
    def _check_available_libraries(self) -> None:
        """Check which transcription libraries are available (without loading them)"""
        # Try faster-whisper first (best compatibility and lowest memory)
        try:
            import faster_whisper
            self.model_type = "faster_whisper"
            print(f"[OK] faster-whisper available (will use '{self.model_size}' model)")
            return
        except ImportError:
            pass
        
        # Try openai-whisper as fallback
        try:
            import whisper
            self.model_type = "openai_whisper"
            print(f"[WARNING] Using openai-whisper (heavier). Consider: pip install faster-whisper")
            return
        except ImportError:
            pass
        
        self.model_type = None
        print("[WARNING] No transcription library available.")
        print("   Install with: pip install faster-whisper")
    
    def load_model(self) -> None:
        """
        Load the transcription model (lazy loading - only called when needed)
        """
        if self.whisper_model is not None:
            return  # Already loaded
            
        if self.model_type == "faster_whisper":
            print(f"[INIT] Loading faster-whisper '{self.model_size}' model (first transcription)...")
            from faster_whisper import WhisperModel
            
            # Use int8 quantization for lower memory usage
            self.whisper_model = WhisperModel(
                self.model_size, 
                device="cpu", 
                compute_type="int8",
                download_root=None,  # Use default cache
                num_workers=1  # Minimize memory
            )
            print(f"[OK] Model loaded successfully!")
            
        elif self.model_type == "openai_whisper":
            print(f"[INIT] Loading openai-whisper '{self.model_size}' model...")
            import whisper
            self.whisper_model = whisper.load_model(self.model_size)
            print("[OK] Model loaded successfully!")
    
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
            raise Exception("No transcription model available. Please install faster-whisper")
        
        # Lazy load the model on first use
        if self.whisper_model is None:
            self.load_model()
        
        if self.model_type == "faster_whisper":
            segments, info = self.whisper_model.transcribe(
                audio_file_path, 
                language=language,
                beam_size=1,  # Lower beam size = less memory
                vad_filter=True,  # Filter silence
            )
            return " ".join([segment.text for segment in segments]).strip()
        
        elif self.model_type == "openai_whisper":
            result = self.whisper_model.transcribe(audio_file_path, language=language)
            return result["text"].strip()
        
        else:
            raise Exception("No transcription model available")
    
    def is_available(self) -> bool:
        """Check if transcription service is available"""
        return self.model_type is not None
    
    def unload_model(self) -> None:
        """Unload the model to free memory (optional optimization)"""
        if self.whisper_model is not None:
            print("[INFO] Unloading transcription model to free memory...")
            self.whisper_model = None
            
            # Force garbage collection
            import gc
            gc.collect()