# AudioInsights TODO

## Task 1: Fix Dependency Issues
**Files**: `requirements.txt`

**Acceptance Criteria**:
- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `python -c "from src.analysis import IntentExtractor, RubricScorer"` succeeds
- [ ] `python -c "from src.transcription import WhisperTranscriber"` succeeds

---

## Task 2: Create Data Directory for Opus File Support
**Files**: `data/.gitkeep`, `README.md`

**Acceptance Criteria**:
- [ ] `data/` directory exists in repo
- [ ] User can place `sample_call.opus` in `data/` and app.py loads it
- [ ] Upload fallback works when no sample file present

---

## Task 3: Fix Code Edge Cases
**Files**: `src/transcription/whisper_transcriber.py`, `src/audio/player.py`

**Acceptance Criteria**:
- [ ] Empty/silent audio array doesn't crash (no division by zero)
- [ ] Invalid overlap >= chunk_duration raises clear `ValueError`
- [ ] Audio loading failures produce useful error messages (not silent)
- [ ] Sample rate of 0 raises `ValueError`

---

## Task 4: Add Setup Script
**Files**: `setup.sh`, `README.md`

**Acceptance Criteria**:
- [ ] `./setup.sh` creates virtualenv and installs dependencies
- [ ] Script checks for ffmpeg and shows clear error if missing
- [ ] Script prompts for OpenAI API key
- [ ] Works on both Mac and Linux

---

## Task 5: Add Basic Test Suite
**Files**: `tests/`, `pytest.ini`

**Acceptance Criteria**:
- [ ] `pytest tests/` passes
- [ ] Tests run without OpenAI API key (mocked)
- [ ] Tests run without Whisper model loaded (mocked)
- [ ] Core modules have test coverage

---

## Task 6: Environment Configuration
**Files**: `.env.example`, `src/config/settings.py`

**Acceptance Criteria**:
- [ ] `.env.example` documents all required variables
- [ ] `cp .env.example .env` + edit API key works
- [ ] Missing API key shows clear error message, not stack trace

---

## Task 7: Validate Demo End-to-End
**Files**: N/A (validation only)

**Acceptance Criteria**:
- [ ] `streamlit run app.py` starts without errors
- [ ] Loading opus file from `data/` works
- [ ] Live transcript appears as audio plays
- [ ] Extracted info panel updates with caller details
- [ ] Rubric checkmarks appear as call progresses
- [ ] Final score displays when complete
