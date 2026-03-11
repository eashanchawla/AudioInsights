import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from analysis.intent_extractor import IntentExtractor
from analysis.models import ExtractedInfo, IntentCategory, CallerType

@pytest.mark.asyncio
async def test_intent_extractor_init():
    extractor = IntentExtractor(api_key="sk-test-key")
    assert extractor.api_key == "sk-test-key"
    assert extractor.model_name == "gpt-4o-mini"

@pytest.mark.asyncio
@patch("analysis.intent_extractor.Agent")
async def test_extract_async(mock_agent_class):
    # Mocking the Agent run result
    mock_agent = mock_agent_class.return_value
    mock_run_result = MagicMock()
    mock_run_result.output = ExtractedInfo(
        caller_name="John Doe",
        primary_intent=IntentCategory.BILLING
    )
    mock_agent.run = AsyncMock(return_value=mock_run_result)

    extractor = IntentExtractor(api_key="sk-test-key")
    result = await extractor.extract("My name is John Doe and I want to pay my bill.")

    assert result.caller_name == "John Doe"
    assert result.primary_intent == IntentCategory.BILLING
    assert mock_agent.run.called

def test_extract_sync():
    # Since it calls the async version, we can mock extract
    extractor = IntentExtractor(api_key="sk-test-key")
    with patch.object(extractor, "extract", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = ExtractedInfo(caller_name="Jane Doe")
        result = extractor.extract_sync("My name is Jane Doe")
        assert result.caller_name == "Jane Doe"
        assert mock_extract.called

def test_get_intent_display():
    extractor = IntentExtractor(api_key="sk-test-key")

    info = ExtractedInfo(primary_intent=IntentCategory.BILLING)
    assert extractor.get_intent_display(info) == "Billing Inquiry"

    info = ExtractedInfo(primary_intent=IntentCategory.UNKNOWN)
    assert extractor.get_intent_display(info) == "Determining intent..."

def test_format_caller_info():
    extractor = IntentExtractor(api_key="sk-test-key")
    info = ExtractedInfo(
        caller_name="John Doe",
        caller_company="Acme Corp",
        caller_type=CallerType.BUSINESS,
        contact_info="john@example.com",
        account_number="12345"
    )

    formatted = extractor.format_caller_info(info)
    assert formatted["Name"] == "John Doe"
    assert formatted["Company"] == "Acme Corp"
    assert formatted["Type"] == "Business"
    assert formatted["Contact"] == "john@example.com"
    assert formatted["Account #"] == "12345"

def test_should_skip_extraction():
    extractor = IntentExtractor(api_key="sk-test-key")

    # First call, don't skip
    assert extractor._should_skip_extraction("Short text") is False

    # Set last transcript
    extractor._last_transcript = "This is a long enough transcript to start with."

    # Tiny change, skip
    assert extractor._should_skip_extraction(extractor._last_transcript + " more words") is True

    # Large change, don't skip
    assert extractor._should_skip_extraction(extractor._last_transcript + " word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11") is False
