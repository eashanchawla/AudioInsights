import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from transcription.whisper_transcriber import WhisperTranscriber, TranscriptionResult

def test_whisper_transcriber_init():
    transcriber = WhisperTranscriber(model_name="tiny", language="en")
    assert transcriber.model_name == "tiny"
    assert transcriber.language == "en"

def test_remove_duplicates():
    transcriber = WhisperTranscriber()
    # Mocking the buffer to have some previous text
    transcriber._transcript_buffer = ["Hello world"]

    # Perfect overlap
    new_text = "Hello world"
    assert transcriber.remove_duplicates(new_text) == ""

    # Partial overlap
    new_text = "world how are you"
    assert transcriber.remove_duplicates(new_text) == "how are you"

    # No overlap
    new_text = "nice to meet you"
    assert transcriber.remove_duplicates(new_text) == "nice to meet you"

@patch("transcription.whisper_transcriber.WhisperTranscriber._load_openai_whisper")
def test_load_model(mock_load):
    transcriber = WhisperTranscriber()
    transcriber.load_model()
    # By default, it tries backends. Since all will fail except our mock
    # it depends on how we mock it.
    assert mock_load.called

@patch("transcription.whisper_transcriber.WhisperTranscriber.transcribe")
def test_transcribe_stream(mock_transcribe):
    mock_transcribe.return_value = TranscriptionResult(
        text="Hello", start_time=0.0, end_time=1.0
    )

    transcriber = WhisperTranscriber()
    audio = np.zeros(16000)
    result = transcriber.transcribe_stream(audio)

    assert result.text == "Hello"
    assert transcriber.full_transcript == "Hello"

    # Second call
    mock_transcribe.return_value = TranscriptionResult(
        text="world", start_time=1.0, end_time=2.0
    )
    result = transcriber.transcribe_stream(audio)
    assert transcriber.full_transcript == "Hello world"

def test_reset_transcript():
    transcriber = WhisperTranscriber()
    transcriber._transcript_buffer = ["Hello"]
    transcriber._full_transcript = "Hello"

    transcriber.reset_transcript()
    assert transcriber._transcript_buffer == []
    assert transcriber.full_transcript == ""

def test_transcribe_calls_load_model():
    transcriber = WhisperTranscriber()
    # Ensure _model is None to trigger load_model
    transcriber._model = None
    transcriber._backend = "whisper"

    with patch.object(transcriber, "load_model") as mock_load:
        # Mock load_model to set _model so transcribe doesn't fail later
        def side_effect():
            transcriber._model = MagicMock()
            transcriber._model.transcribe.return_value = {"text": "Test"}

        mock_load.side_effect = side_effect

        audio = np.zeros(16000)
        transcriber.transcribe(audio)
        assert mock_load.called
