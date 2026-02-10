"""Application settings and configuration."""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Settings:
    """Application configuration settings."""

    # Audio settings
    sample_rate: int = 16000  # Whisper expects 16kHz
    chunk_duration_sec: float = 3.0  # Duration of each audio chunk
    overlap_duration_sec: float = 0.5  # Overlap between chunks to avoid cut-off words

    # Transcription settings
    whisper_model: str = "small"  # Options: tiny, base, small, medium, large
    language: str = "en"

    # Analysis settings
    analysis_trigger_interval_sec: float = 10.0  # Max time between analysis runs
    min_words_for_analysis: int = 10  # Minimum new words before triggering analysis

    # LLM settings
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    llm_model: str = "gpt-4o-mini"  # Cost-effective for extraction tasks

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    rubric_path: Path = field(default_factory=lambda: Path(__file__).parent / "rubric.yaml")

    # UI settings
    update_interval_ms: int = 100  # How often to update the UI

    def __post_init__(self):
        """Ensure paths exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Default settings instance
default_settings = Settings()
