"""
AudioInsights Demo - Simulated Conversation Analysis

This demo shows the analysis capabilities using a simulated conversation,
without requiring an actual audio file or Whisper transcription.

Run with: streamlit run demo.py
"""

import streamlit as st
import time
import os
from typing import Optional

st.set_page_config(
    page_title="AudioInsights Demo",
    page_icon="🎧",
    layout="wide",
)

# Sample conversation that simulates a customer service call
SAMPLE_CONVERSATION = [
    {"speaker": "Agent", "text": "Good morning, thank you for calling TechCorp support. My name is Sarah, how can I help you today?", "delay": 2.0},
    {"speaker": "Customer", "text": "Hi Sarah, my name is John Smith and I'm calling from Acme Industries. I'm having trouble with my account.", "delay": 2.5},
    {"speaker": "Agent", "text": "I'd be happy to help you with that, John. Can you please verify your account number for me?", "delay": 2.0},
    {"speaker": "Customer", "text": "Sure, it's AC-789456. We've been getting some billing errors on our monthly statements.", "delay": 2.5},
    {"speaker": "Agent", "text": "Thank you for that information. I can see your account here. I understand billing issues can be frustrating. Let me take a look at your recent statements.", "delay": 3.0},
    {"speaker": "Customer", "text": "We were charged twice for the same service last month, around $500 extra.", "delay": 2.0},
    {"speaker": "Agent", "text": "I can see the duplicate charge on my end. I apologize for this error. I'll process a refund for the $500 right away. It should appear in your account within 3-5 business days.", "delay": 3.5},
    {"speaker": "Customer", "text": "That sounds good. Thank you for resolving this so quickly.", "delay": 1.5},
    {"speaker": "Agent", "text": "You're welcome! Is there anything else I can help you with today?", "delay": 2.0},
    {"speaker": "Customer", "text": "No, that's all I needed. Thanks again.", "delay": 1.5},
    {"speaker": "Agent", "text": "Thank you for calling TechCorp, John. Have a great day!", "delay": 2.0},
]


def init_session_state():
    """Initialize session state."""
    defaults = {
        "demo_running": False,
        "demo_complete": False,
        "current_index": 0,
        "transcript": "",
        "extracted_info": None,
        "rubric_result": None,
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_api_key() -> bool:
    """Check for API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            return True
        st.sidebar.warning("⚠️ API key required for analysis features")
        return False
    return True


def run_analysis(transcript: str):
    """Run intent extraction and rubric scoring."""
    try:
        from src.analysis.intent_extractor import IntentExtractor
        from src.analysis.rubric_scorer import RubricScorer

        extractor = IntentExtractor()
        scorer = RubricScorer()

        info = extractor.extract_sync(transcript)
        rubric = scorer.evaluate_sync(transcript)

        return info, rubric
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None, None


def render_transcript():
    """Render transcript panel."""
    st.subheader("📝 Live Transcript")

    transcript_area = st.container(height=350)
    with transcript_area:
        for msg in st.session_state.messages:
            speaker = msg["speaker"]
            text = msg["text"]

            if speaker == "Agent":
                st.markdown(f"**🎧 Agent:** {text}")
            else:
                st.markdown(f"**👤 Customer:** {text}")


def render_info():
    """Render extracted info panel."""
    st.subheader("📋 Extracted Info")

    info = st.session_state.extracted_info
    if info is None:
        st.info("Analyzing conversation...")
        return

    st.markdown("**Caller Details**")
    st.text(f"Name: {info.caller_name or 'N/A'}")
    st.text(f"Company: {info.caller_company or 'N/A'}")
    st.text(f"Account: {info.account_number or 'N/A'}")

    st.markdown("**Intent**")
    intent = info.primary_intent.value.replace("_", " ").title()
    st.success(f"🎯 {intent}")

    if info.intent_summary:
        st.caption(info.intent_summary)

    if info.key_details:
        st.markdown("**Key Details**")
        for detail in info.key_details[:3]:
            st.text(f"• {detail}")


def render_rubric():
    """Render rubric panel."""
    st.subheader("✅ Performance Rubric")

    result = st.session_state.rubric_result
    if result is None:
        # Show default rubric items as pending
        default_items = [
            "Greeted customer professionally",
            "Introduced themselves by name",
            "Verified customer identity",
            "Acknowledged customer's issue",
            "Provided solution or next steps",
            "Confirmed customer understanding",
            "Offered additional assistance",
            "Closed call professionally",
        ]
        for item in default_items:
            st.checkbox(item, value=False, disabled=True)
        return

    # Show score
    color = "green" if result.score >= 70 else "orange" if result.score >= 40 else "red"
    st.markdown(
        f"**Score: <span style='color:{color}'>{result.score:.0f}%</span>** ({result.completed_count}/{result.total_count})",
        unsafe_allow_html=True,
    )
    st.progress(result.score / 100)

    st.divider()

    for item in result.items:
        if item.completed:
            st.markdown(f"✅ ~~{item.description}~~")
        else:
            st.markdown(f"⬜ {item.description}")


def main():
    """Main demo application."""
    init_session_state()

    st.title("🎧 AudioInsights Demo")
    st.markdown("Real-time conversation analysis prototype")

    # Sidebar
    st.sidebar.title("Controls")
    has_api_key = check_api_key()

    st.sidebar.divider()
    st.sidebar.markdown("**About this Demo**")
    st.sidebar.info(
        "This demo simulates a customer service call being transcribed and analyzed in real-time. "
        "In a production system, audio would come from a live call or recording."
    )

    # Main layout
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        render_transcript()

    with col2:
        render_info()

    with col3:
        render_rubric()

    # Controls
    st.divider()
    col1, col2, col3, _ = st.columns([1, 1, 1, 3])

    with col1:
        start = st.button(
            "▶️ Start Demo",
            disabled=st.session_state.demo_running or not has_api_key,
            use_container_width=True,
        )

    with col2:
        stop = st.button(
            "⏹️ Stop",
            disabled=not st.session_state.demo_running,
            use_container_width=True,
        )

    with col3:
        reset = st.button("🔄 Reset", use_container_width=True)

    if reset:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if stop:
        st.session_state.demo_running = False
        st.rerun()

    if start and has_api_key:
        st.session_state.demo_running = True
        st.session_state.demo_complete = False
        st.session_state.current_index = 0
        st.session_state.messages = []
        st.session_state.transcript = ""
        st.session_state.extracted_info = None
        st.session_state.rubric_result = None
        st.rerun()

    # Continue processing if demo is running
    if st.session_state.demo_running and st.session_state.current_index < len(SAMPLE_CONVERSATION):
        progress = st.progress(st.session_state.current_index / len(SAMPLE_CONVERSATION))
        status = st.empty()

        i = st.session_state.current_index
        turn = SAMPLE_CONVERSATION[i]

        # Add to messages
        st.session_state.messages.append(turn)
        st.session_state.transcript += f"\n{turn['speaker']}: {turn['text']}"
        st.session_state.current_index += 1

        # Update progress
        progress.progress(st.session_state.current_index / len(SAMPLE_CONVERSATION))
        status.text(f"Processing... ({st.session_state.current_index}/{len(SAMPLE_CONVERSATION)})")

        # Run analysis every few turns
        if st.session_state.current_index % 3 == 0 or st.session_state.current_index == len(SAMPLE_CONVERSATION):
            status.text("Running analysis...")
            info, rubric = run_analysis(st.session_state.transcript)
            if info:
                st.session_state.extracted_info = info
            if rubric:
                st.session_state.rubric_result = rubric

        # Check if done
        if st.session_state.current_index >= len(SAMPLE_CONVERSATION):
            st.session_state.demo_running = False
            st.session_state.demo_complete = True
            st.rerun()
        else:
            # Delay to simulate real-time, then continue
            time.sleep(turn["delay"] * 0.5)  # Faster for demo
            st.rerun()

    if st.session_state.demo_complete:
        st.success("✅ Demo complete! The analysis shows how the system tracks agent performance in real-time.")

    # Footer
    st.divider()
    st.caption(
        "💡 This demo uses a pre-scripted conversation. In production, the transcript would come from "
        "real-time Whisper transcription of audio."
    )


if __name__ == "__main__":
    main()
