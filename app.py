"""
AudioInsights - Real-time Conversation Analysis

Streamlit application for live transcription and analysis of customer service calls.
"""

import streamlit as st
import time
from pathlib import Path
import threading
from queue import Queue
import os

# Page config must be first Streamlit command
st.set_page_config(
    page_title="AudioInsights - Real-time Call Analysis",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.audio.player import AudioPlayer
from src.audio.processor import AudioProcessor
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.analysis.intent_extractor import IntentExtractor
from src.analysis.rubric_scorer import RubricScorer
from src.analysis.models import AnalysisTriggerState, ExtractedInfo, RubricResult
from src.config.settings import Settings


def init_session_state():
    """Initialize Streamlit session state."""
    defaults = {
        "is_processing": False,
        "transcript": "",
        "transcript_chunks": [],
        "extracted_info": None,
        "rubric_result": None,
        "current_time": 0.0,
        "audio_loaded": False,
        "error_message": None,
        "analysis_trigger": AnalysisTriggerState(),
        "last_analysis_transcript": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.sidebar.error(
            "⚠️ OpenAI API key not found!\n\n"
            "Set it in your `.env` file or environment:\n"
            "```\nOPENAI_API_KEY='your-key'\n```"
        )
        # Allow user to input API key directly
        api_key = st.sidebar.text_input("Or enter API key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            return True
        return False
    return True


def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.title("🎧 AudioInsights")
    st.sidebar.markdown("Real-time Conversation Analysis")
    st.sidebar.divider()

    # Settings
    st.sidebar.subheader("Settings")

    whisper_model = st.sidebar.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium"],
        index=2,  # Default to 'small'
        help="Larger models are more accurate but slower",
    )

    simulate_realtime = st.sidebar.checkbox(
        "Simulate Real-time",
        value=True,
        help="Add delays to simulate live audio input",
    )

    # File upload
    st.sidebar.divider()
    st.sidebar.subheader("Audio Source")

    uploaded_file = st.sidebar.file_uploader(
        "Upload audio file",
        type=["opus", "wav", "mp3", "m4a", "ogg", "flac"],
        help="Upload a call recording to analyze",
    )

    # Or use sample file
    sample_file = Path(__file__).parent / "data" / "sample_call.opus"
    use_sample = False

    if sample_file.exists():
        use_sample = st.sidebar.checkbox("Use sample file", value=not uploaded_file)

    return {
        "whisper_model": whisper_model,
        "simulate_realtime": simulate_realtime,
        "uploaded_file": uploaded_file,
        "use_sample": use_sample,
        "sample_file": sample_file if use_sample else None,
    }


def render_transcript_panel():
    """Render the live transcript panel."""
    st.subheader("📝 Live Transcript")

    # Create a container for the transcript
    transcript_container = st.container(height=400)

    with transcript_container:
        if st.session_state.transcript:
            # Display transcript with styling
            st.markdown(
                f"""
                <div style="font-family: monospace; line-height: 1.6; padding: 10px;">
                {st.session_state.transcript.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Transcript will appear here when processing begins...")

    # Show current time
    if st.session_state.is_processing:
        st.caption(f"⏱️ Current time: {st.session_state.current_time:.1f}s")


def render_info_panel():
    """Render the extracted information panel."""
    st.subheader("📋 Extracted Information")

    info: ExtractedInfo = st.session_state.extracted_info

    if info is None:
        st.info("Information will be extracted as the conversation progresses...")
        return

    # Caller Info
    st.markdown("**Caller Details**")
    col1, col2 = st.columns(2)

    with col1:
        st.text(f"Name: {info.caller_name or 'Not identified'}")
        st.text(f"Company: {info.caller_company or 'Not mentioned'}")

    with col2:
        st.text(f"Account #: {info.account_number or 'Not provided'}")
        st.text(f"Contact: {info.contact_info or 'Not provided'}")

    # Intent
    st.markdown("**Call Intent**")
    intent_display = info.primary_intent.value.replace("_", " ").title()
    st.info(f"🎯 **{intent_display}**")

    if info.intent_summary:
        st.caption(info.intent_summary)

    # Sentiment & Urgency
    if info.sentiment or info.urgency:
        col1, col2 = st.columns(2)
        with col1:
            if info.sentiment:
                emoji = {"positive": "😊", "neutral": "😐", "frustrated": "😤", "angry": "😠"}.get(
                    info.sentiment.lower(), "❓"
                )
                st.text(f"Sentiment: {emoji} {info.sentiment}")
        with col2:
            if info.urgency:
                st.text(f"Urgency: {info.urgency}")

    # Key Details
    if info.key_details:
        st.markdown("**Key Details**")
        for detail in info.key_details[:5]:  # Show max 5
            st.text(f"• {detail}")


def render_rubric_panel():
    """Render the rubric checklist panel."""
    st.subheader("✅ Agent Performance Rubric")

    result: RubricResult = st.session_state.rubric_result

    if result is None:
        # Show empty rubric
        st.info("Rubric evaluation will update as the conversation progresses...")

        # Still show items as pending
        try:
            from src.analysis.rubric_scorer import RubricScorer

            scorer = RubricScorer.__new__(RubricScorer)
            scorer.rubric_items = scorer._get_default_rubric(scorer)
            for item in scorer.rubric_items:
                st.checkbox(item.description, value=False, disabled=True, key=f"rubric_{item.id}")
        except Exception:
            pass
        return

    # Show score
    score_color = "green" if result.score >= 70 else "orange" if result.score >= 40 else "red"
    st.markdown(
        f"**Score: <span style='color: {score_color}'>{result.score:.0f}%</span>** "
        f"({result.completed_count}/{result.total_count} items)",
        unsafe_allow_html=True,
    )

    st.progress(result.score / 100)

    # Show checklist
    st.divider()

    for item in result.items:
        col1, col2 = st.columns([0.1, 0.9])

        with col1:
            if item.completed:
                st.markdown("✅")
            else:
                st.markdown("⬜")

        with col2:
            if item.completed:
                st.markdown(f"~~{item.description}~~")
                if item.evidence:
                    st.caption(f"📌 {item.evidence[:100]}...")
            else:
                st.markdown(item.description)


def process_audio(
    audio_path: Path,
    settings: dict,
    transcript_queue: Queue,
    status_queue: Queue,
):
    """Process audio in a background thread."""
    try:
        # Initialize components
        player = AudioPlayer(
            chunk_duration_sec=3.0,
            overlap_sec=0.5,
            simulate_realtime=settings["simulate_realtime"],
        )

        processor = AudioProcessor(target_sample_rate=16000)
        transcriber = WhisperTranscriber(
            model_name=settings["whisper_model"],
            language="en",
        )

        # Load model
        status_queue.put({"status": "loading_model"})
        transcriber.load_model()

        # Load audio
        status_queue.put({"status": "loading_audio"})
        player.load(audio_path)

        status_queue.put({"status": "processing", "duration": player.duration_sec})

        # Process chunks
        full_transcript = ""

        for chunk in player.stream_chunks():
            # Prepare audio for Whisper
            audio = processor.prepare_for_whisper(chunk.data, chunk.sample_rate)

            # Transcribe
            result = transcriber.transcribe(
                audio,
                sample_rate=16000,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )

            if result.text:
                # Handle duplicate removal for overlapping chunks
                clean_text = transcriber.remove_duplicates(result.text)
                if clean_text:
                    full_transcript += " " + clean_text
                    full_transcript = full_transcript.strip()

                    transcript_queue.put(
                        {
                            "transcript": full_transcript,
                            "time": chunk.end_time,
                            "chunk_text": clean_text,
                        }
                    )

        status_queue.put({"status": "complete"})

    except Exception as e:
        status_queue.put({"status": "error", "error": str(e)})


def run_analysis(transcript: str, intent_extractor: IntentExtractor, rubric_scorer: RubricScorer):
    """Run intent extraction and rubric scoring."""
    try:
        # Run both analyses
        extracted_info = intent_extractor.extract_sync(transcript)
        rubric_result = rubric_scorer.evaluate_sync(transcript)

        return extracted_info, rubric_result
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None, None


def main():
    """Main application."""
    init_session_state()

    # Check API key
    has_api_key = check_api_key()

    # Render sidebar
    sidebar_config = render_sidebar()

    # Main content area - three columns
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        render_transcript_panel()

    with col2:
        render_info_panel()

    with col3:
        render_rubric_panel()

    # Control buttons
    st.divider()

    col1, col2, col3, _ = st.columns([1, 1, 1, 3])

    with col1:
        start_button = st.button(
            "▶️ Start Analysis",
            disabled=st.session_state.is_processing or not has_api_key,
            use_container_width=True,
        )

    with col2:
        stop_button = st.button(
            "⏹️ Stop",
            disabled=not st.session_state.is_processing,
            use_container_width=True,
        )

    with col3:
        reset_button = st.button("🔄 Reset", use_container_width=True)

    # Handle buttons
    if reset_button:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if start_button and has_api_key:
        # Determine audio source
        audio_path = None

        if sidebar_config["uploaded_file"]:
            # Save uploaded file temporarily
            temp_path = Path("/tmp") / sidebar_config["uploaded_file"].name
            with open(temp_path, "wb") as f:
                f.write(sidebar_config["uploaded_file"].getbuffer())
            audio_path = temp_path

        elif sidebar_config["sample_file"] and sidebar_config["sample_file"].exists():
            audio_path = sidebar_config["sample_file"]

        if audio_path is None:
            st.error("Please upload an audio file or use the sample file.")
        else:
            st.session_state.is_processing = True
            st.session_state.transcript = ""
            st.session_state.extracted_info = None
            st.session_state.rubric_result = None

            # Create queues for thread communication
            transcript_queue = Queue()
            status_queue = Queue()

            # Start processing thread
            thread = threading.Thread(
                target=process_audio,
                args=(
                    audio_path,
                    {
                        "whisper_model": sidebar_config["whisper_model"],
                        "simulate_realtime": sidebar_config["simulate_realtime"],
                    },
                    transcript_queue,
                    status_queue,
                ),
                daemon=True,
            )
            thread.start()

            # Initialize analyzers
            try:
                intent_extractor = IntentExtractor()
                rubric_scorer = RubricScorer()
            except ValueError as e:
                st.session_state.is_processing = False
                st.error(f"Configuration Error: {e}")
                st.rerun()
            analysis_trigger = AnalysisTriggerState()

            # Progress display
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Main processing loop
            total_duration = 0
            last_analysis_time = 0

            while True:
                # Check status queue
                while not status_queue.empty():
                    status = status_queue.get()

                    if status["status"] == "loading_model":
                        status_text.text("Loading Whisper model...")
                    elif status["status"] == "loading_audio":
                        status_text.text("Loading audio file...")
                    elif status["status"] == "processing":
                        total_duration = status["duration"]
                        status_text.text("Processing audio...")
                    elif status["status"] == "complete":
                        st.session_state.is_processing = False
                        progress_bar.progress(1.0)
                        status_text.text("Processing complete!")

                        # Final analysis
                        if st.session_state.transcript:
                            info, rubric = run_analysis(
                                st.session_state.transcript, intent_extractor, rubric_scorer
                            )
                            st.session_state.extracted_info = info
                            st.session_state.rubric_result = rubric

                        st.rerun()
                    elif status["status"] == "error":
                        st.session_state.is_processing = False
                        st.error(f"Error: {status['error']}")
                        st.rerun()

                # Check transcript queue
                while not transcript_queue.empty():
                    data = transcript_queue.get()
                    st.session_state.transcript = data["transcript"]
                    st.session_state.current_time = data["time"]

                    # Update progress
                    if total_duration > 0:
                        progress = min(data["time"] / total_duration, 1.0)
                        progress_bar.progress(progress)

                    # Check if we should run analysis
                    current_time = time.time()
                    if analysis_trigger.should_analyze(
                        st.session_state.transcript, current_time, min_interval_sec=8.0, min_new_words=20
                    ):
                        info, rubric = run_analysis(
                            st.session_state.transcript, intent_extractor, rubric_scorer
                        )
                        if info:
                            st.session_state.extracted_info = info
                        if rubric:
                            st.session_state.rubric_result = rubric
                        analysis_trigger.update_after_analysis(st.session_state.transcript, current_time)

                    st.rerun()

                # Small delay to prevent busy waiting
                time.sleep(0.1)

                # Check if thread is still alive
                if not thread.is_alive() and transcript_queue.empty() and status_queue.empty():
                    break

    if stop_button:
        st.session_state.is_processing = False
        st.rerun()

    # Show error if any
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    # Footer
    st.divider()
    st.caption(
        "💡 **Tip**: For best results, use clear audio recordings with minimal background noise. "
        "The system works best with customer service call recordings."
    )


if __name__ == "__main__":
    main()
