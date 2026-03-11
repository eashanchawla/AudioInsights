import pytest
from analysis.models import (
    RubricResult,
    RubricItem,
    AnalysisTriggerState,
    IntentCategory,
    CallerType
)

def test_rubric_result_completion_percentage():
    result = RubricResult(
        items=[],
        score=0.0,
        completed_count=2,
        total_count=10
    )
    assert result.completion_percentage == 20.0

    result.total_count = 0
    assert result.completion_percentage == 0.0

def test_analysis_trigger_state_should_analyze():
    state = AnalysisTriggerState(
        last_analysis_time=100.0,
        last_word_count=50
    )

    # Sentence completion trigger
    # In src/analysis/models.py:
    # new_words = current_words - self.last_word_count
    # if transcript.rstrip().endswith((".", "?", "!")):
    #     if new_words >= 5:

    # "This is a complete sentence." has 5 words. 5 - 0 = 5. Wait, last_word_count=50.
    # If transcript is "This is a complete sentence.", current_words is 5.
    # new_words = 5 - 50 = -45.
    # So we need a transcript with 55 words to have new_words = 5.

    base_transcript = "word " * 50
    complete_sentence = "This is a complete sentence." # 5 words
    transcript = base_transcript + complete_sentence

    assert state.should_analyze(transcript, 101.0) is True

    # Too few new words for sentence completion trigger
    short_sentence = "Wait." # 1 word
    assert state.should_analyze(base_transcript + short_sentence, 101.0) is False

    # Time threshold trigger
    # if time_since_last >= min_interval_sec and new_words >= 5:
    # min_interval_sec defaults to 10.0
    assert state.should_analyze(base_transcript + "word " * 5, 111.0) is True

    # Word count trigger
    # if new_words >= min_new_words:
    # min_new_words defaults to 15
    assert state.should_analyze(base_transcript + "word " * 15, 101.0) is True

    # No trigger
    assert state.should_analyze(base_transcript + "word " * 4, 101.0) is False

def test_analysis_trigger_state_update():
    state = AnalysisTriggerState(
        last_analysis_time=100.0,
        last_word_count=50
    )

    state.update_after_analysis("word " * 60, 110.0)
    assert state.last_analysis_time == 110.0
    assert state.last_word_count == 60
