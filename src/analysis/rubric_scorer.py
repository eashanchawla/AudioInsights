"""Rubric scoring for agent performance evaluation."""

from typing import List, Optional
import os
from pathlib import Path
import yaml

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field

from .models import RubricItem, RubricResult


class RubricEvaluation(BaseModel):
    """Model for LLM rubric evaluation response."""

    completed_items: List[str] = Field(
        default_factory=list,
        description="List of rubric item IDs that have been completed",
    )
    evidence: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of item ID to evidence from transcript",
    )


RUBRIC_SYSTEM_PROMPT = """You are an AI assistant specialized in evaluating customer service agent performance.
You will be given a conversation transcript and a rubric of expected agent behaviors.

Your task is to identify which rubric items the agent has completed based on the transcript.

Guidelines:
1. Only mark an item as completed if there is clear evidence in the transcript
2. Provide the specific quote or paraphrase as evidence for each completed item
3. Be somewhat generous - if the agent substantially meets the requirement, count it
4. Consider the context - early in the call, not all items will be completed yet
5. Some items may not apply to every call type

Focus on the AGENT's behavior, not the customer's. Look for what the agent says and does.
"""


class RubricScorer:
    """Evaluates agent performance against a rubric."""

    def __init__(
        self,
        rubric_path: Optional[Path] = None,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the rubric scorer.

        Args:
            rubric_path: Path to rubric YAML file
            model_name: OpenAI model to use
            api_key: OpenAI API key
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Load rubric
        self.rubric_items = self._load_rubric(rubric_path)

        # Create the PydanticAI agent
        model = OpenAIModel(model_name, api_key=self.api_key)
        self.agent = Agent(
            model,
            result_type=RubricEvaluation,
            system_prompt=RUBRIC_SYSTEM_PROMPT,
        )

        # Track completed items across evaluations
        self._completed_items: set[str] = set()
        self._evidence: dict[str, str] = {}

    def _load_rubric(self, rubric_path: Optional[Path]) -> List[RubricItem]:
        """Load rubric from YAML file."""
        if rubric_path is None:
            rubric_path = Path(__file__).parent.parent / "config" / "rubric.yaml"

        if not rubric_path.exists():
            # Return default rubric if file doesn't exist
            return self._get_default_rubric()

        with open(rubric_path) as f:
            data = yaml.safe_load(f)

        items = []
        for item_data in data.get("rubric", []):
            items.append(
                RubricItem(
                    id=item_data["id"],
                    description=item_data["description"],
                    detection_hint=item_data.get("detection_hint", ""),
                    priority=item_data.get("priority", 1),
                    completed=False,
                )
            )

        return sorted(items, key=lambda x: x.priority)

    def _get_default_rubric(self) -> List[RubricItem]:
        """Return default rubric items."""
        return [
            RubricItem(
                id="greeting",
                description="Greeted the customer professionally",
                detection_hint="Agent said hello or welcomed the customer",
                priority=1,
            ),
            RubricItem(
                id="identification",
                description="Introduced themselves by name",
                detection_hint="Agent stated their name",
                priority=1,
            ),
            RubricItem(
                id="verify_customer",
                description="Verified customer identity",
                detection_hint="Agent asked for customer's name or account info",
                priority=2,
            ),
            RubricItem(
                id="acknowledge_issue",
                description="Acknowledged the customer's issue",
                detection_hint="Agent showed understanding of the problem",
                priority=2,
            ),
            RubricItem(
                id="provide_solution",
                description="Provided a solution or next steps",
                detection_hint="Agent offered resolution or explained next steps",
                priority=3,
            ),
            RubricItem(
                id="confirm_understanding",
                description="Confirmed customer understanding",
                detection_hint="Agent asked if customer understood",
                priority=3,
            ),
            RubricItem(
                id="professional_closing",
                description="Closed the call professionally",
                detection_hint="Agent thanked customer and said goodbye",
                priority=4,
            ),
        ]

    async def evaluate(self, transcript: str) -> RubricResult:
        """
        Evaluate the transcript against the rubric.

        Args:
            transcript: The conversation transcript

        Returns:
            RubricResult with item statuses and scores
        """
        if not transcript.strip():
            return self._build_result()

        # Build the rubric description for the prompt
        rubric_desc = "\n".join(
            [
                f"- {item.id}: {item.description} (Hint: {item.detection_hint})"
                for item in self.rubric_items
            ]
        )

        prompt = f"""Evaluate this customer service call transcript against the following rubric.
Identify which items the AGENT has completed.

RUBRIC ITEMS:
{rubric_desc}

TRANSCRIPT:
{transcript}

For each completed item, provide the evidence (quote or paraphrase) from the transcript."""

        try:
            result = await self.agent.run(prompt)
            evaluation = result.data

            # Update completed items (items stay completed once checked)
            self._completed_items.update(evaluation.completed_items)
            self._evidence.update(evaluation.evidence)

            return self._build_result()

        except Exception as e:
            print(f"Error during rubric evaluation: {e}")
            return self._build_result()

    def evaluate_sync(self, transcript: str) -> RubricResult:
        """
        Synchronous version of evaluate.

        Args:
            transcript: The conversation transcript

        Returns:
            RubricResult with item statuses and scores
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass

        return loop.run_until_complete(self.evaluate(transcript))

    def _build_result(self) -> RubricResult:
        """Build the result object from current state."""
        items = []
        completed_count = 0

        for item in self.rubric_items:
            is_completed = item.id in self._completed_items
            if is_completed:
                completed_count += 1

            items.append(
                RubricItem(
                    id=item.id,
                    description=item.description,
                    detection_hint=item.detection_hint,
                    priority=item.priority,
                    completed=is_completed,
                    evidence=self._evidence.get(item.id),
                )
            )

        total = len(self.rubric_items)
        score = (completed_count / total * 100) if total > 0 else 0

        return RubricResult(
            items=items,
            score=score,
            completed_count=completed_count,
            total_count=total,
        )

    def reset(self) -> None:
        """Reset the scorer state."""
        self._completed_items = set()
        self._evidence = {}

    def get_items_display(self) -> List[dict]:
        """Get rubric items formatted for display."""
        result = self._build_result()
        return [
            {
                "id": item.id,
                "description": item.description,
                "completed": item.completed,
                "evidence": item.evidence,
            }
            for item in result.items
        ]
