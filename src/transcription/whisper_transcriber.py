"""Real-time transcription using Whisper."""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass
import sys


@dataclass
class TranscriptionResult:
    """Result from transcription."""

    text: str  # Transcribed text
    start_time: float  # Start time in audio
    end_time: float  # End time in audio
    confidence: Optional[float] = None  # Confidence score if available
    language: Optional[str] = None  # Detected language


class WhisperTranscriber:
    """
    Handles real-time transcription using Whisper.

    Supports multiple backends:
    - mlx-whisper: Optimized for Apple Silicon (Mac Metal GPU)
    - openai-whisper: Standard implementation
    - transformers: HuggingFace implementation
    """

    def __init__(
        self,
        model_name: str = "small",
        language: str = "en",
        device: Optional[str] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            language: Language code for transcription
            device: Device to use (auto-detected if None)
        """
        self.model_name = model_name
        self.language = language
        self.device = device

        self._model = None
        self._backend: Optional[str] = None
        self._processor = None  # For transformers backend

        # Buffer for accumulating transcript
        self._transcript_buffer: List[str] = []
        self._full_transcript: str = ""

    def load_model(self) -> None:
        """Load the Whisper model using the best available backend."""
        if self._model is not None:
            return

        # Try backends in order of preference
        backends = [
            ("mlx_whisper", self._load_mlx_whisper),
            ("whisper", self._load_openai_whisper),
            ("transformers", self._load_transformers_whisper),
        ]

        for backend_name, loader in backends:
            try:
                loader()
                self._backend = backend_name
                print(f"Loaded Whisper using {backend_name} backend")
                return
            except ImportError:
                continue
            except Exception as e:
                print(f"Failed to load {backend_name}: {e}")
                continue

        raise RuntimeError(
            "Could not load Whisper model. Please install one of: "
            "mlx-whisper (Mac), openai-whisper, or transformers"
        )

    def _load_mlx_whisper(self) -> None:
        """Load mlx-whisper (Apple Silicon optimized)."""
        import mlx_whisper

        # mlx_whisper loads model on first transcription
        self._model = mlx_whisper
        self._backend = "mlx_whisper"

    def _load_openai_whisper(self) -> None:
        """Load OpenAI's whisper."""
        import whisper

        device = self.device
        if device is None:
            # Auto-detect device
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"  # Mac Metal
            else:
                device = "cpu"

        self._model = whisper.load_model(self.model_name, device=device)
        self._backend = "whisper"

    def _load_transformers_whisper(self) -> None:
        """Load Whisper using HuggingFace transformers."""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch

        model_id = f"openai/whisper-{self.model_name}"

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._processor = WhisperProcessor.from_pretrained(model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_id).to(
            device
        )
        self.device = device
        self._backend = "transformers"

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio chunk.

        Args:
            audio: Audio data as numpy array (float32, normalized to [-1, 1])
            sample_rate: Sample rate of the audio
            start_time: Start time of this chunk in the full audio
            end_time: End time of this chunk (calculated if None)

        Returns:
            TranscriptionResult with transcribed text
        """
        if self._model is None:
            self.load_model()

        if end_time is None:
            end_time = start_time + len(audio) / sample_rate

        # Ensure audio is float32 and normalized
        audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))

        # Transcribe based on backend
        if self._backend == "mlx_whisper":
            text = self._transcribe_mlx(audio)
        elif self._backend == "whisper":
            text = self._transcribe_openai(audio)
        elif self._backend == "transformers":
            text = self._transcribe_transformers(audio, sample_rate)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        # Clean up text
        text = text.strip()

        return TranscriptionResult(
            text=text,
            start_time=start_time,
            end_time=end_time,
            language=self.language,
        )

    def _transcribe_mlx(self, audio: np.ndarray) -> str:
        """Transcribe using mlx-whisper."""
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=f"mlx-community/whisper-{self.model_name}-mlx",
            language=self.language,
        )
        return result.get("text", "")

    def _transcribe_openai(self, audio: np.ndarray) -> str:
        """Transcribe using OpenAI whisper."""
        result = self._model.transcribe(
            audio, language=self.language, fp16=False, verbose=False
        )
        return result.get("text", "")

    def _transcribe_transformers(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe using HuggingFace transformers."""
        import torch

        # Prepare input features
        input_features = self._processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self._model.generate(input_features)

        # Decode
        transcription = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription[0] if transcription else ""

    def transcribe_stream(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        start_time: float = 0.0,
    ) -> TranscriptionResult:
        """
        Transcribe audio and accumulate to the full transcript.

        This is useful for streaming transcription where you want to
        maintain a running transcript.

        Args:
            audio: Audio chunk to transcribe
            sample_rate: Sample rate
            start_time: Start time of this chunk

        Returns:
            TranscriptionResult with the new text
        """
        result = self.transcribe(audio, sample_rate, start_time)

        if result.text:
            self._transcript_buffer.append(result.text)
            self._full_transcript = " ".join(self._transcript_buffer)

        return result

    @property
    def full_transcript(self) -> str:
        """Get the accumulated full transcript."""
        return self._full_transcript

    def reset_transcript(self) -> None:
        """Reset the accumulated transcript."""
        self._transcript_buffer = []
        self._full_transcript = ""

    def remove_duplicates(self, new_text: str, overlap_threshold: float = 0.5) -> str:
        """
        Remove duplicate text that may appear due to chunk overlaps.

        Args:
            new_text: Newly transcribed text
            overlap_threshold: Threshold for considering text as duplicate

        Returns:
            Text with duplicates removed
        """
        if not self._transcript_buffer:
            return new_text

        last_text = self._transcript_buffer[-1]
        if not last_text:
            return new_text

        # Check for overlap at the end of last text and beginning of new text
        words_last = last_text.split()
        words_new = new_text.split()

        if not words_last or not words_new:
            return new_text

        # Find overlap
        max_overlap = min(len(words_last), len(words_new))
        best_overlap = 0

        for overlap_len in range(1, max_overlap + 1):
            if words_last[-overlap_len:] == words_new[:overlap_len]:
                best_overlap = overlap_len

        if best_overlap > 0:
            # Remove overlapping words from new text
            return " ".join(words_new[best_overlap:])

        return new_text
