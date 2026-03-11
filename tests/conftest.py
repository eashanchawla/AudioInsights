import os
import pytest
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(autouse=True)
def mock_settings_env(monkeypatch):
    """Ensure tests don't require real API keys or specific environments."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-testing")
