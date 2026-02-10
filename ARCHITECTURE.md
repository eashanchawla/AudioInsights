# AudioInsights - Real-Time Conversation Analysis Prototype

## Overview

This prototype demonstrates real-time conversation analysis for call center agents, featuring:
- Live transcription using Whisper (optimized for Mac Metal)
- Intent and entity extraction (caller name, company, reason for calling)
- Agent performance rubric scoring with live checklist updates

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Streamlit Frontend                                 │
├─────────────────┬───────────────────────┬───────────────────────────────────┤
│   Transcript    │   Extracted Info      │         Rubric Checklist          │
│   Panel         │   Panel               │         Panel                     │
│                 │                       │                                   │
│   [Live text]   │   Caller: John Smith  │   [✓] Greeted customer            │
│                 │   Company: Acme Corp  │   [✓] Verified identity           │
│                 │   Intent: Billing     │   [ ] Acknowledged issue          │
│                 │   inquiry             │   [ ] Provided solution           │
│                 │                       │   [ ] Confirmed resolution        │
└────────┬────────┴───────────┬───────────┴───────────────┬───────────────────┘
         │                    │                           │
         ▼                    ▼                           ▼
┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐
│  Transcription  │  │  Intent/Entity      │  │  Rubric Evaluator           │
│  Service        │  │  Extractor          │  │  Service                    │
│  (Whisper)      │  │  (PydanticAI)       │  │  (PydanticAI)               │
└────────┬────────┘  └───────────┬─────────┘  └─────────────┬───────────────┘
         │                       │                          │
         └───────────────────────┴──────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Audio Stream Handler  │
                    │   (Chunked Processing)  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Audio Source          │
                    │   (.opus file playback) │
                    └─────────────────────────┘
```

## Key Design Decisions

### 1. When to Run AI Prompts (Intent/Rubric Evaluation)

**Challenge**: Running prompts on every transcription update is wasteful and slow.

**Solution**: Event-driven processing with multiple triggers:

1. **Sentence Completion Trigger**: Run when we detect a complete sentence (period, question mark, or long pause in transcription)

2. **Time-based Fallback**: Run every 10-15 seconds if no sentence completion detected (handles long monologues)

3. **Speaker Change Trigger**: When we detect a speaker change (useful for rubric - agent actions often follow customer statements)

4. **Cumulative Context**: Always send the FULL transcript to the LLM, but ask it to focus on NEW content. This ensures context is maintained while avoiding duplicate processing.

```python
# Pseudo-code for trigger logic
class AnalysisTrigger:
    def should_analyze(self, new_text: str, time_since_last: float) -> bool:
        # Trigger on sentence endings
        if new_text.rstrip().endswith(('.', '?', '!')):
            return True
        # Trigger on time threshold (15 seconds)
        if time_since_last > 15.0:
            return True
        # Trigger on significant new content (50+ words since last)
        if self.words_since_last > 50:
            return True
        return False
```

### 2. Channel Separation (Agent vs Customer)

**Prototype Approach**:
- Use a single .opus file
- Attempt basic speaker diarization using voice characteristics
- Label speakers as "Speaker 1" and "Speaker 2"
- Allow manual assignment of which speaker is the agent

**Real-World Implementation**:
- Telephony systems (Twilio, Genesys, etc.) provide separate audio channels
- Channel 1 = Agent microphone
- Channel 2 = Customer line
- Each channel can be transcribed independently with speaker labels

```python
# Real-world dual-channel handling
class DualChannelProcessor:
    def __init__(self):
        self.agent_channel = AudioChannel(label="Agent")
        self.customer_channel = AudioChannel(label="Customer")

    def process_stereo(self, audio_frame):
        left, right = self.split_channels(audio_frame)
        agent_text = self.transcribe(left)
        customer_text = self.transcribe(right)
        return self.merge_timeline(agent_text, customer_text)
```

### 3. Mac Metal GPU Optimization

Using `mlx-whisper` for Apple Silicon optimization:
- Native Metal GPU acceleration
- 2-5x faster than CPU-based Whisper
- Lower memory footprint

Fallback options:
- `whisper.cpp` with Metal backend
- Standard `openai-whisper` with MPS device

### 4. Rubric Scoring Logic

**Approach**: The rubric is a checklist of expected agent behaviors. We use an LLM to evaluate if each item has been satisfied based on the transcript.

```python
rubric_items = [
    {"id": "greeting", "description": "Agent greeted the customer professionally"},
    {"id": "identity", "description": "Agent verified customer identity"},
    {"id": "acknowledge", "description": "Agent acknowledged the customer's issue"},
    {"id": "solution", "description": "Agent provided a solution or next steps"},
    {"id": "confirm", "description": "Agent confirmed customer satisfaction"},
    {"id": "closing", "description": "Agent closed the call professionally"},
]
```

The LLM returns which items are now satisfied, and we update the UI checkboxes.

## Project Structure

```
AudioInsights/
├── app.py                    # Streamlit main application
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── player.py         # Audio file playback with chunking
│   │   └── processor.py      # Audio preprocessing (resampling, etc.)
│   ├── transcription/
│   │   ├── __init__.py
│   │   └── whisper_transcriber.py  # Whisper integration
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── intent_extractor.py     # Intent/entity extraction
│   │   ├── rubric_scorer.py        # Rubric evaluation
│   │   └── models.py               # Pydantic models for structured output
│   └── config/
│       ├── __init__.py
│       ├── rubric.yaml             # Configurable rubric definition
│       └── settings.py             # Application settings
├── data/
│   └── sample_call.opus            # Sample audio for testing
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

## Data Flow

1. **Audio Chunking**: Audio file is read in 2-3 second chunks (simulating real-time)
2. **Transcription**: Each chunk is transcribed, with overlap handling to avoid cut-off words
3. **Sentence Detection**: Buffer transcriptions until sentence boundaries detected
4. **Analysis Trigger**: On sentence completion, trigger intent extraction and rubric scoring
5. **UI Update**: Stream results to Streamlit frontend via session state

## Dependencies

- `mlx-whisper`: Whisper optimized for Apple Silicon
- `pydantic-ai`: Structured LLM outputs for extraction
- `streamlit`: Web frontend
- `pydub` or `torchaudio`: Audio file handling
- `librosa`: Audio preprocessing (resampling)
