# AudioInsights

Real-time conversation analysis for customer service calls.

## Overview

AudioInsights is a prototype that demonstrates real-time analysis of customer service conversations:

- **Live Transcription**: Uses Whisper for real-time speech-to-text
- **Intent/Entity Extraction**: Identifies caller name, company, reason for calling
- **Agent Performance Scoring**: Evaluates agents against a customizable rubric in real-time

## Features

- Real-time transcription with multiple Whisper backends (MLX for Mac, OpenAI Whisper, Transformers)
- PydanticAI-powered extraction for structured outputs
- Configurable rubric checklist with live updates
- Streamlit-based UI for easy visualization

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for intent extraction and rubric scoring)
- FFmpeg (for audio file support)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AudioInsights.git
cd AudioInsights

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Mac with Apple Silicon (recommended for Metal GPU acceleration)
pip install mlx-whisper

# Set up environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Demo

The demo uses a simulated conversation to show the analysis capabilities:

```bash
# Set your API key
export OPENAI_API_KEY='your-key-here'

# Run the demo
streamlit run demo.py
```

### Running with Real Audio

```bash
# Run the full application
streamlit run app.py
```

Then:
1. Upload an audio file (.opus, .wav, .mp3, etc.)
2. Click "Start Analysis"
3. Watch real-time transcription and analysis

## Project Structure

```
AudioInsights/
├── app.py                    # Main Streamlit application
├── demo.py                   # Demo with simulated conversation
├── src/
│   ├── audio/
│   │   ├── player.py         # Audio file playback with chunking
│   │   └── processor.py      # Audio preprocessing
│   ├── transcription/
│   │   └── whisper_transcriber.py  # Multi-backend Whisper integration
│   ├── analysis/
│   │   ├── models.py         # Pydantic models for structured output
│   │   ├── intent_extractor.py     # Intent/entity extraction
│   │   └── rubric_scorer.py        # Rubric evaluation
│   └── config/
│       ├── settings.py       # Application settings
│       └── rubric.yaml       # Customizable rubric definition
├── data/                     # Sample audio files
├── ARCHITECTURE.md           # System design documentation
├── DEPLOYMENT.md             # Production deployment guide
└── requirements.txt
```

## Customizing the Rubric

Edit `src/config/rubric.yaml` to customize the agent performance checklist:

```yaml
rubric:
  - id: greeting
    description: "Greeted the customer professionally"
    detection_hint: "Agent said hello or welcomed the customer"
    priority: 1

  - id: your_custom_item
    description: "Your custom requirement"
    detection_hint: "How to detect this behavior"
    priority: 2
```

## Mac Metal GPU Optimization

For best performance on Apple Silicon:

1. Install MLX-Whisper:
   ```bash
   pip install mlx-whisper
   ```

2. The system automatically detects and uses MLX when available

3. Expected performance:
   - M1: ~2x faster than real-time (small model)
   - M2/M3: ~3-4x faster than real-time

## Analysis Trigger Logic

The system intelligently decides when to run LLM analysis:

1. **Sentence Completion**: Triggers when a complete sentence is detected
2. **Time Threshold**: Triggers every 10-15 seconds if no sentence detected
3. **Word Count**: Triggers after 20+ new words

This balances responsiveness with API cost efficiency.

## Real-World Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for guidance on:

- Dual-channel (agent/customer) audio handling
- Speaker diarization for mono recordings
- Scaling for production call centers
- CRM and quality management integration
- Privacy and compliance considerations

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests (when added)
pytest tests/

# Type checking
mypy src/
```

## Roadmap

- [ ] Add Voice Activity Detection (VAD) for smarter chunking
- [ ] Implement speaker diarization for mono recordings
- [ ] Add real-time suggestions/prompts for agents
- [ ] Build RAG integration for knowledge base lookups
- [ ] Create supervisor dashboard for monitoring multiple calls
- [ ] Add call sentiment trending visualization

## License

MIT License - see LICENSE file
