"""Intent and entity extraction using PydanticAI."""

from typing import Optional
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from .models import ExtractedInfo, IntentCategory, CallerType


# System prompt for intent/entity extraction
EXTRACTION_SYSTEM_PROMPT = """You are an AI assistant specialized in analyzing customer service call transcripts.
Your task is to extract key information from the conversation in real-time.

Guidelines:
1. Extract caller information (name, company, contact details) when mentioned
2. Identify the primary intent/reason for the call
3. Note any account numbers or reference numbers
4. Assess the caller's sentiment and urgency
5. Capture key details that would be helpful for the agent

Important:
- Only extract information that is explicitly stated in the transcript
- If information is not mentioned, leave the field as None/null
- Be conservative - don't infer or guess details that aren't clearly stated
- The transcript may be partial/in-progress, so some information may appear later

The transcript comes from a customer service call. Speaker labels may or may not be present.
If speaker labels are present:
- "Agent:" or "Speaker 1:" typically refers to the customer service representative
- "Customer:" or "Speaker 2:" typically refers to the caller
"""


class IntentExtractor:
    """Extracts intent and entities from conversation transcripts."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the intent extractor.

        Args:
            model_name: OpenAI model to use
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Create the PydanticAI agent
        model = OpenAIModel(model_name, api_key=self.api_key)
        self.agent = Agent(
            model,
            result_type=ExtractedInfo,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

        # Cache for last extraction to avoid redundant calls
        self._last_transcript: str = ""
        self._last_result: Optional[ExtractedInfo] = None

    async def extract(self, transcript: str) -> ExtractedInfo:
        """
        Extract intent and entities from transcript.

        Args:
            transcript: The conversation transcript

        Returns:
            ExtractedInfo with extracted information
        """
        if not transcript.strip():
            return ExtractedInfo()

        # Skip if transcript hasn't changed significantly
        if self._should_skip_extraction(transcript):
            return self._last_result or ExtractedInfo()

        prompt = f"""Analyze this customer service call transcript and extract relevant information:

TRANSCRIPT:
{transcript}

Extract the caller's information, their intent, and any other relevant details."""

        try:
            result = await self.agent.run(prompt)
            self._last_transcript = transcript
            self._last_result = result.data
            return result.data
        except Exception as e:
            print(f"Error during extraction: {e}")
            return self._last_result or ExtractedInfo()

    def extract_sync(self, transcript: str) -> ExtractedInfo:
        """
        Synchronous version of extract.

        Args:
            transcript: The conversation transcript

        Returns:
            ExtractedInfo with extracted information
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Handle nested event loops (e.g., in Jupyter or Streamlit)
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass

        return loop.run_until_complete(self.extract(transcript))

    def _should_skip_extraction(self, transcript: str) -> bool:
        """Check if we should skip extraction (transcript hasn't changed enough)."""
        if not self._last_transcript:
            return False

        # Check if transcript has grown significantly
        old_words = len(self._last_transcript.split())
        new_words = len(transcript.split())

        # Skip if less than 10 new words
        if new_words - old_words < 10:
            return True

        return False

    def get_intent_display(self, info: ExtractedInfo) -> str:
        """Get a formatted display string for the intent."""
        if info.primary_intent == IntentCategory.UNKNOWN:
            return "Determining intent..."

        intent_map = {
            IntentCategory.BILLING: "Billing Inquiry",
            IntentCategory.TECHNICAL_SUPPORT: "Technical Support",
            IntentCategory.ACCOUNT_MANAGEMENT: "Account Management",
            IntentCategory.SALES: "Sales Inquiry",
            IntentCategory.COMPLAINT: "Complaint",
            IntentCategory.INQUIRY: "General Inquiry",
            IntentCategory.CANCELLATION: "Cancellation Request",
            IntentCategory.OTHER: "Other",
        }

        return intent_map.get(info.primary_intent, str(info.primary_intent))

    def format_caller_info(self, info: ExtractedInfo) -> dict:
        """Format caller info for display."""
        return {
            "Name": info.caller_name or "Not identified",
            "Company": info.caller_company or "Not mentioned",
            "Type": info.caller_type.value.replace("_", " ").title(),
            "Contact": info.contact_info or "Not provided",
            "Account #": info.account_number or "Not provided",
        }
