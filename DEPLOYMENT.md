# Real-World Deployment Considerations

This document outlines how the AudioInsights prototype could be deployed in a production call center environment.

## Channel Separation: Agent vs Customer

### The Challenge

In a real call center, distinguishing between the agent and customer voices is critical for:
- Accurate rubric scoring (we only evaluate the agent)
- Proper intent extraction (understanding what the customer wants)
- Compliance monitoring

### Solution Options

#### Option 1: Dual-Channel Recording (Recommended)

Most professional telephony systems provide separate audio channels:

```
┌─────────────────┐     ┌─────────────────┐
│  Agent Phone    │     │  Customer Line  │
│  (Channel 1)    │     │  (Channel 2)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         Telephony System                │
│   (Twilio, Genesys, Five9, etc.)       │
└────────┬───────────────────────┬────────┘
         │                       │
         ▼                       ▼
    Left Channel            Right Channel
    (Agent Audio)          (Customer Audio)
```

**Implementation:**

```python
class DualChannelProcessor:
    def __init__(self):
        self.agent_transcriber = WhisperTranscriber()
        self.customer_transcriber = WhisperTranscriber()

    def process_stereo_frame(self, stereo_audio: np.ndarray):
        # Split channels
        agent_audio = stereo_audio[:, 0]  # Left = Agent
        customer_audio = stereo_audio[:, 1]  # Right = Customer

        # Transcribe separately
        agent_text = self.agent_transcriber.transcribe(agent_audio)
        customer_text = self.customer_transcriber.transcribe(customer_audio)

        return {
            "agent": agent_text,
            "customer": customer_text,
            "merged": self.merge_by_timestamp(agent_text, customer_text)
        }
```

**Telephony Platform Integration:**

| Platform | Dual-Channel Support | API/Integration |
|----------|---------------------|-----------------|
| Twilio | Yes | Media Streams WebSocket |
| Genesys | Yes | AudioHook Integration |
| Five9 | Yes | VoiceStream API |
| Amazon Connect | Yes | Kinesis Video Streams |
| Cisco UCCE | Yes | MediaSense API |

#### Option 2: Speaker Diarization

When only mono audio is available, use AI-based speaker diarization:

```python
from pyannote.audio import Pipeline

class SpeakerDiarizer:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )

    def diarize(self, audio_path: str) -> list:
        """
        Returns segments with speaker labels.

        Output: [
            {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01"},
            ...
        ]
        """
        return self.pipeline(audio_path)
```

**Limitations:**
- Less accurate than dual-channel
- Requires post-processing to assign speaker identities
- May struggle with overlapping speech

#### Option 3: Voice Enrollment

Pre-enroll agent voices for identification:

```python
class AgentVoiceIdentifier:
    def __init__(self):
        self.enrolled_agents = {}  # agent_id -> voice embedding

    def enroll_agent(self, agent_id: str, voice_samples: list):
        """Create voice embedding for agent."""
        embeddings = [self.extract_embedding(s) for s in voice_samples]
        self.enrolled_agents[agent_id] = np.mean(embeddings, axis=0)

    def identify_speaker(self, audio_segment: np.ndarray) -> str:
        """Identify if segment is from enrolled agent."""
        embedding = self.extract_embedding(audio_segment)

        for agent_id, agent_embedding in self.enrolled_agents.items():
            similarity = cosine_similarity(embedding, agent_embedding)
            if similarity > 0.85:
                return agent_id

        return "customer"
```

## Real-Time Processing Architecture

### Production Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Call Center Infrastructure                    │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Audio Stream Gateway                           │
│  (WebSocket server receiving real-time audio from telephony)     │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │  Transcription    │   │  Transcription    │
        │  Worker (Agent)   │   │  Worker (Customer)│
        │  [Whisper GPU]    │   │  [Whisper GPU]    │
        └─────────┬─────────┘   └─────────┬─────────┘
                  │                       │
                  └───────────┬───────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │         Transcript Merger               │
        │    (Combines and timestamps both)       │
        └───────────────────┬─────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────┐
        │         Analysis Service                │
        │  • Intent Extraction                    │
        │  • Rubric Scoring                       │
        │  • Sentiment Analysis                   │
        └───────────────────┬─────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────┐
        │         Agent Desktop UI                │
        │  (Real-time updates via WebSocket)      │
        └─────────────────────────────────────────┘
```

### Latency Optimization

For real-time assistance, latency is critical:

| Component | Target Latency | Optimization |
|-----------|---------------|--------------|
| Audio streaming | < 100ms | WebSocket, binary protocol |
| Transcription | < 500ms | GPU inference, streaming ASR |
| Analysis | < 2s | Async processing, caching |
| UI update | < 50ms | WebSocket push, differential updates |

**Total target: < 3 seconds end-to-end**

### Streaming Whisper

For lowest latency, use streaming ASR:

```python
# Using faster-whisper with streaming
from faster_whisper import WhisperModel

class StreamingTranscriber:
    def __init__(self):
        self.model = WhisperModel("small", device="cuda")
        self.audio_buffer = []

    def process_chunk(self, audio_chunk: np.ndarray):
        self.audio_buffer.extend(audio_chunk)

        # Process when buffer reaches threshold
        if len(self.audio_buffer) >= 16000 * 2:  # 2 seconds
            segments, _ = self.model.transcribe(
                np.array(self.audio_buffer),
                vad_filter=True,
                word_timestamps=True
            )

            for segment in segments:
                yield segment.text

            # Keep overlap for continuity
            self.audio_buffer = self.audio_buffer[-8000:]
```

## Scaling Considerations

### GPU Resource Planning

| Concurrent Calls | Whisper Model | GPU Requirement |
|-----------------|---------------|-----------------|
| 10 | small | 1x RTX 3090 |
| 50 | small | 4x RTX 3090 or 2x A100 |
| 100 | base/tiny | 4x A100 |
| 500+ | tiny | GPU cluster + load balancing |

### Cost Analysis (Estimated)

| Component | Cost per Call (5 min avg) |
|-----------|---------------------------|
| GPU Transcription | $0.02 - $0.05 |
| LLM Analysis (GPT-4o-mini) | $0.01 - $0.03 |
| Infrastructure | $0.005 |
| **Total** | **$0.035 - $0.085** |

## Integration Points

### CRM Integration

```python
class CRMIntegration:
    def on_call_complete(self, call_data: dict):
        """Push extracted data to CRM."""
        crm_update = {
            "call_id": call_data["id"],
            "customer_name": call_data["extracted_info"].caller_name,
            "company": call_data["extracted_info"].caller_company,
            "intent": call_data["extracted_info"].primary_intent,
            "summary": call_data["extracted_info"].intent_summary,
            "agent_score": call_data["rubric_result"].score,
            "transcript": call_data["transcript"],
        }

        self.crm_client.update_contact(crm_update)
```

### Quality Management Integration

```python
class QMIntegration:
    def on_rubric_update(self, agent_id: str, rubric_result: RubricResult):
        """Real-time coaching alerts."""
        if rubric_result.score < 50:
            self.alert_supervisor(
                agent_id=agent_id,
                message="Agent may need assistance",
                rubric=rubric_result
            )
```

## Privacy & Compliance

### Data Handling

1. **PII Detection**: Automatically detect and redact sensitive data
2. **Consent**: Ensure call recording consent is obtained
3. **Retention**: Follow data retention policies (typically 90 days - 7 years)
4. **Access Control**: Role-based access to transcripts and recordings

### GDPR/CCPA Considerations

```python
class PrivacyHandler:
    def redact_pii(self, transcript: str) -> str:
        """Redact personally identifiable information."""
        patterns = {
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': '[PHONE]',
            r'\b\d{3}[-]?\d{2}[-]?\d{4}\b': '[SSN]',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
            r'\b\d{16}\b': '[CARD]',
        }

        for pattern, replacement in patterns.items():
            transcript = re.sub(pattern, replacement, transcript)

        return transcript
```

## Mac Metal Deployment

For Mac-based deployment (development or small-scale production):

### MLX-Whisper Installation

```bash
# Install MLX Whisper for Apple Silicon
pip install mlx-whisper

# Download models
python -c "import mlx_whisper; mlx_whisper.transcribe('test.wav', path_or_hf_repo='mlx-community/whisper-small-mlx')"
```

### Performance on Apple Silicon

| Mac Model | Whisper Model | Real-time Factor |
|-----------|---------------|------------------|
| M1 | small | 0.5x (2x faster than real-time) |
| M1 Pro | small | 0.3x |
| M2 Max | medium | 0.4x |
| M3 Max | large | 0.5x |

### Memory Requirements

| Whisper Model | VRAM/RAM Required |
|---------------|-------------------|
| tiny | 1 GB |
| base | 1.5 GB |
| small | 2.5 GB |
| medium | 5 GB |
| large | 10 GB |

## Recommended Production Stack

1. **Audio Ingestion**: Twilio Media Streams or Amazon Connect
2. **Transcription**: Faster-Whisper on NVIDIA GPUs or MLX-Whisper on Mac
3. **Analysis**: GPT-4o-mini via Azure OpenAI (for enterprise compliance)
4. **Frontend**: React + WebSocket for real-time updates
5. **Backend**: FastAPI with async processing
6. **Database**: PostgreSQL for transcripts, Redis for real-time state
7. **Monitoring**: Prometheus + Grafana for performance metrics
