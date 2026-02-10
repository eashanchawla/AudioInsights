"""Pydantic models for structured LLM outputs."""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class CallerType(str, Enum):
    """Type of caller."""

    INDIVIDUAL = "individual"
    BUSINESS = "business"
    UNKNOWN = "unknown"


class IntentCategory(str, Enum):
    """High-level intent categories for calls."""

    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    ACCOUNT_MANAGEMENT = "account_management"
    SALES = "sales"
    COMPLAINT = "complaint"
    INQUIRY = "inquiry"
    CANCELLATION = "cancellation"
    OTHER = "other"
    UNKNOWN = "unknown"


class ExtractedInfo(BaseModel):
    """Information extracted from the conversation."""

    # Caller identification
    caller_name: Optional[str] = Field(
        None, description="Name of the person calling, if mentioned"
    )
    caller_company: Optional[str] = Field(
        None, description="Company name the caller represents, if mentioned"
    )
    caller_type: CallerType = Field(
        CallerType.UNKNOWN, description="Whether caller is individual or business"
    )
    contact_info: Optional[str] = Field(
        None, description="Any contact information mentioned (phone, email, etc.)"
    )
    account_number: Optional[str] = Field(
        None, description="Account or reference number if mentioned"
    )

    # Intent
    primary_intent: IntentCategory = Field(
        IntentCategory.UNKNOWN, description="Primary reason for the call"
    )
    intent_summary: Optional[str] = Field(
        None, description="Brief summary of what the caller wants"
    )

    # Additional context
    urgency: Optional[str] = Field(
        None, description="Urgency level if apparent (low, medium, high)"
    )
    sentiment: Optional[str] = Field(
        None, description="Caller sentiment (positive, neutral, frustrated, angry)"
    )
    key_details: List[str] = Field(
        default_factory=list,
        description="Key details or facts mentioned in the conversation",
    )


class RubricItem(BaseModel):
    """A single rubric item."""

    id: str = Field(..., description="Unique identifier for this rubric item")
    description: str = Field(..., description="Description of the expected behavior")
    detection_hint: str = Field(
        "", description="Hint for detecting when this behavior occurs"
    )
    priority: int = Field(1, description="Priority/order of this item (1=highest)")
    completed: bool = Field(False, description="Whether this item has been satisfied")
    evidence: Optional[str] = Field(
        None, description="Evidence from transcript that this item was completed"
    )


class RubricResult(BaseModel):
    """Result of rubric evaluation."""

    items: List[RubricItem] = Field(
        default_factory=list, description="All rubric items with their status"
    )
    score: float = Field(
        0.0, description="Overall score (0-100) based on completed items"
    )
    completed_count: int = Field(0, description="Number of completed items")
    total_count: int = Field(0, description="Total number of items")

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.completed_count / self.total_count) * 100


class AnalysisTriggerState(BaseModel):
    """Tracks when to trigger analysis."""

    last_analysis_time: float = Field(0.0, description="Timestamp of last analysis")
    last_word_count: int = Field(0, description="Word count at last analysis")
    sentence_buffer: str = Field("", description="Buffer for incomplete sentences")

    def should_analyze(
        self,
        transcript: str,
        current_time: float,
        min_interval_sec: float = 10.0,
        min_new_words: int = 15,
    ) -> bool:
        """
        Determine if we should run analysis.

        Triggers on:
        1. Complete sentence (ends with . ? !)
        2. Time threshold exceeded
        3. Significant new content (word count)

        Args:
            transcript: Current full transcript
            current_time: Current timestamp
            min_interval_sec: Minimum seconds between analysis
            min_new_words: Minimum new words before analysis

        Returns:
            True if analysis should be triggered
        """
        current_words = len(transcript.split())
        new_words = current_words - self.last_word_count
        time_since_last = current_time - self.last_analysis_time

        # Check for sentence completion
        if transcript.rstrip().endswith((".", "?", "!")):
            if new_words >= 5:  # At least a few new words
                return True

        # Time-based trigger
        if time_since_last >= min_interval_sec and new_words >= 5:
            return True

        # Word count trigger
        if new_words >= min_new_words:
            return True

        return False

    def update_after_analysis(self, transcript: str, current_time: float) -> None:
        """Update state after running analysis."""
        self.last_analysis_time = current_time
        self.last_word_count = len(transcript.split())
