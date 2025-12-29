"""Audio file player with chunked streaming for simulating real-time input."""

import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Tuple
import time
from dataclasses import dataclass


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""

    data: np.ndarray  # Audio samples
    sample_rate: int  # Sample rate
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    chunk_index: int  # Index of this chunk


class AudioPlayer:
    """
    Plays audio files in chunks, simulating real-time audio input.

    Supports common audio formats including .opus, .wav, .mp3, etc.
    """

    def __init__(
        self,
        chunk_duration_sec: float = 3.0,
        overlap_sec: float = 0.5,
        simulate_realtime: bool = True,
    ):
        """
        Initialize the audio player.

        Args:
            chunk_duration_sec: Duration of each audio chunk in seconds
            overlap_sec: Overlap between chunks to avoid cutting off words
            simulate_realtime: If True, add delays to simulate real-time playback
        """
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_sec = overlap_sec
        self.simulate_realtime = simulate_realtime

        # Audio data (loaded on open)
        self._audio: Optional[np.ndarray] = None
        self._sample_rate: Optional[int] = None
        self._file_path: Optional[Path] = None

    def load(self, file_path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self._file_path = file_path

        # Try different loading methods
        audio, sr = self._load_audio(file_path)

        self._audio = audio
        self._sample_rate = sr

        return audio, sr

    def _load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio using available libraries."""
        suffix = file_path.suffix.lower()

        # Try pydub first (handles many formats including opus)
        try:
            from pydub import AudioSegment

            audio_segment = AudioSegment.from_file(str(file_path))
            sr = audio_segment.frame_rate

            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())

            # Handle stereo
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))

            # Normalize to float32 [-1, 1]
            samples = samples.astype(np.float32) / 32768.0

            return samples, sr
        except Exception as e:
            pass

        # Try torchaudio
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(file_path))
            # Convert to numpy, shape: [channels, samples] -> [samples, channels] or [samples]
            audio = waveform.numpy()
            if audio.shape[0] == 1:
                audio = audio[0]  # Mono
            else:
                audio = audio.T  # Stereo: [samples, channels]
            return audio, sr
        except Exception as e:
            pass

        # Try librosa as last resort
        try:
            import librosa

            audio, sr = librosa.load(str(file_path), sr=None, mono=False)
            if audio.ndim == 2:
                audio = audio.T  # [samples, channels]
            return audio, sr
        except Exception as e:
            raise RuntimeError(
                f"Could not load audio file {file_path}. "
                f"Tried pydub, torchaudio, and librosa. "
                f"Make sure ffmpeg is installed for opus support."
            )

    @property
    def duration_sec(self) -> float:
        """Get total duration of loaded audio in seconds."""
        if self._audio is None or self._sample_rate is None:
            return 0.0
        n_samples = len(self._audio) if self._audio.ndim == 1 else self._audio.shape[0]
        return n_samples / self._sample_rate

    @property
    def sample_rate(self) -> int:
        """Get sample rate of loaded audio."""
        return self._sample_rate or 0

    @property
    def is_stereo(self) -> bool:
        """Check if loaded audio is stereo."""
        if self._audio is None:
            return False
        return self._audio.ndim == 2 and self._audio.shape[1] == 2

    def stream_chunks(self) -> Iterator[AudioChunk]:
        """
        Stream audio in chunks, simulating real-time input.

        Yields:
            AudioChunk objects containing audio data and metadata
        """
        if self._audio is None or self._sample_rate is None:
            raise RuntimeError("No audio loaded. Call load() first.")

        # Calculate chunk parameters
        chunk_samples = int(self.chunk_duration_sec * self._sample_rate)
        overlap_samples = int(self.overlap_sec * self._sample_rate)
        step_samples = chunk_samples - overlap_samples

        # Get total samples (handle mono vs stereo)
        total_samples = (
            len(self._audio) if self._audio.ndim == 1 else self._audio.shape[0]
        )

        chunk_index = 0
        start_sample = 0

        while start_sample < total_samples:
            # Calculate end sample
            end_sample = min(start_sample + chunk_samples, total_samples)

            # Extract chunk
            if self._audio.ndim == 1:
                chunk_data = self._audio[start_sample:end_sample]
            else:
                chunk_data = self._audio[start_sample:end_sample, :]

            # Calculate times
            start_time = start_sample / self._sample_rate
            end_time = end_sample / self._sample_rate

            chunk = AudioChunk(
                data=chunk_data,
                sample_rate=self._sample_rate,
                start_time=start_time,
                end_time=end_time,
                chunk_index=chunk_index,
            )

            # Simulate real-time delay
            if self.simulate_realtime and chunk_index > 0:
                # Wait for the duration of non-overlapping portion
                delay = (end_sample - start_sample - overlap_samples) / self._sample_rate
                if delay > 0:
                    time.sleep(delay)

            yield chunk

            # Move to next chunk
            start_sample += step_samples
            chunk_index += 1

    def get_full_audio(self) -> Tuple[np.ndarray, int]:
        """
        Get the full audio data without chunking.

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if self._audio is None or self._sample_rate is None:
            raise RuntimeError("No audio loaded. Call load() first.")
        return self._audio.copy(), self._sample_rate
