import numpy as np
import pytest
from audio.processor import AudioProcessor

def test_audio_processor_init():
    processor = AudioProcessor(target_sample_rate=16000)
    assert processor.target_sample_rate == 16000

def test_to_mono():
    processor = AudioProcessor()

    # Already mono
    mono_audio = np.array([0.1, 0.2, 0.3])
    assert np.array_equal(processor.to_mono(mono_audio), mono_audio)

    # Stereo
    stereo_audio = np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]])
    expected_mono = np.array([0.3, 0.4, 0.5])
    processed_mono = processor.to_mono(stereo_audio)
    assert np.allclose(processed_mono, expected_mono)

def test_split_channels():
    processor = AudioProcessor()

    # Stereo [2, samples]
    stereo_audio = np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]])
    left, right = processor.split_channels(stereo_audio)
    assert np.array_equal(left, stereo_audio[0])
    assert np.array_equal(right, stereo_audio[1])

    # Stereo [samples, 2]
    stereo_audio_t = stereo_audio.T
    left, right = processor.split_channels(stereo_audio_t)
    assert np.array_equal(left, stereo_audio_t[:, 0])
    assert np.array_equal(right, stereo_audio_t[:, 1])

def test_normalize():
    processor = AudioProcessor()
    audio = np.array([0.5, -1.0, 2.0])
    normalized = processor.normalize(audio)
    assert np.max(np.abs(normalized)) == 1.0
    assert normalized[2] == 1.0
    assert normalized[1] == -0.5

def test_resample_fallback():
    # Test linear interpolation fallback when librosa is not used/available
    # or just verify it works
    processor = AudioProcessor(target_sample_rate=16000)
    audio = np.array([0.0, 1.0, 0.0, -1.0])
    # Resample from 8000 to 16000 (double the length)
    resampled = processor.resample(audio, original_sr=8000, target_sr=16000)
    assert len(resampled) == 8

def test_prepare_for_whisper():
    processor = AudioProcessor(target_sample_rate=16000)
    stereo_audio = np.array([[0.1, 0.2], [0.3, 0.4]])
    processed = processor.prepare_for_whisper(stereo_audio, sample_rate=8000)

    assert processed.dtype == np.float32
    assert processed.ndim == 1
    assert np.max(np.abs(processed)) <= 1.0
