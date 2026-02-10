"""Audio preprocessing utilities."""

import numpy as np
from typing import Tuple, Optional
import io


class AudioProcessor:
    """Handles audio preprocessing for transcription."""

    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize the audio processor.

        Args:
            target_sample_rate: Target sample rate for Whisper (default 16kHz)
        """
        self.target_sample_rate = target_sample_rate

    def resample(
        self, audio: np.ndarray, original_sr: int, target_sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Audio data as numpy array
            original_sr: Original sample rate
            target_sr: Target sample rate (defaults to self.target_sample_rate)

        Returns:
            Resampled audio data
        """
        target_sr = target_sr or self.target_sample_rate

        if original_sr == target_sr:
            return audio

        # Use librosa for high-quality resampling
        try:
            import librosa

            return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
        except ImportError:
            # Fallback to simple linear interpolation
            duration = len(audio) / original_sr
            new_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono.

        Args:
            audio: Audio data (can be mono or stereo)

        Returns:
            Mono audio data
        """
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # Average channels
            return np.mean(audio, axis=0)
        else:
            raise ValueError(f"Unexpected audio dimensions: {audio.ndim}")

    def split_channels(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split stereo audio into left and right channels.

        Useful for dual-channel call recordings where:
        - Left channel = Agent
        - Right channel = Customer

        Args:
            audio: Stereo audio data (2D array with shape [2, samples] or [samples, 2])

        Returns:
            Tuple of (left_channel, right_channel)
        """
        if audio.ndim == 1:
            # Mono audio - return same for both channels
            return audio, audio.copy()

        if audio.shape[0] == 2:
            # Shape is [2, samples]
            return audio[0], audio[1]
        elif audio.shape[1] == 2:
            # Shape is [samples, 2]
            return audio[:, 0], audio[:, 1]
        else:
            raise ValueError(f"Cannot split channels from shape: {audio.shape}")

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.

        Args:
            audio: Audio data

        Returns:
            Normalized audio data
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio

    def prepare_for_whisper(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Prepare audio for Whisper transcription.

        - Converts to mono
        - Resamples to 16kHz
        - Normalizes to float32 in [-1, 1]

        Args:
            audio: Raw audio data
            sample_rate: Original sample rate

        Returns:
            Processed audio ready for Whisper
        """
        # Convert to mono
        audio = self.to_mono(audio)

        # Resample to 16kHz
        audio = self.resample(audio, sample_rate, self.target_sample_rate)

        # Ensure float32
        audio = audio.astype(np.float32)

        # Normalize
        audio = self.normalize(audio)

        return audio
