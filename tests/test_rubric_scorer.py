import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from analysis.rubric_scorer import RubricScorer, RubricEvaluation
from analysis.models import RubricResult

@pytest.mark.asyncio
async def test_rubric_scorer_init():
    scorer = RubricScorer(api_key="sk-test-key")
    assert scorer.api_key == "sk-test-key"
    assert len(scorer.rubric_items) > 0

@pytest.mark.asyncio
@patch("analysis.rubric_scorer.Agent")
async def test_evaluate_async(mock_agent_class):
    mock_agent = mock_agent_class.return_value
    mock_run_result = MagicMock()
    mock_run_result.output = RubricEvaluation(
        completed_items=["greeting"],
        evidence={"greeting": "Hello how can I help?"}
    )
    mock_agent.run = AsyncMock(return_value=mock_run_result)

    scorer = RubricScorer(api_key="sk-test-key")
    result = await scorer.evaluate("Hello how can I help?")

    assert result.completed_count == 1
    assert result.items[0].id == "greeting"
    assert result.items[0].completed is True
    assert result.items[0].evidence == "Hello how can I help?"
    assert mock_agent.run.called

def test_evaluate_sync():
    scorer = RubricScorer(api_key="sk-test-key")
    with patch.object(scorer, "evaluate", new_callable=AsyncMock) as mock_evaluate:
        mock_evaluate.return_value = RubricResult(
            items=[],
            score=100.0,
            completed_count=1,
            total_count=1
        )
        result = scorer.evaluate_sync("Hello")
        assert result.score == 100.0
        assert mock_evaluate.called

def test_reset():
    scorer = RubricScorer(api_key="sk-test-key")
    scorer._completed_items.add("greeting")
    scorer._evidence["greeting"] = "evidence"

    scorer.reset()
    assert len(scorer._completed_items) == 0
    assert len(scorer._evidence) == 0

def test_get_items_display():
    scorer = RubricScorer(api_key="sk-test-key")
    scorer._completed_items.add("greeting")
    scorer._evidence["greeting"] = "evidence"

    display = scorer.get_items_display()
    greeting_item = next(item for item in display if item["id"] == "greeting")
    assert greeting_item["completed"] is True
    assert greeting_item["evidence"] == "evidence"
