# AudioInsights - GitHub Issues for Parallel Development

## Context

The user wants to create GitHub issues from the TODO.md / plan, broken down so they can be worked on **in parallel via separate branches** (e.g., by Jules). Each issue must be self-contained, have clear acceptance criteria, and minimize file overlap with other issues.

## Parallelism Strategy

Issues are grouped into **3 waves** based on dependencies. Within each wave, all issues can run in parallel on separate branches.

```
Wave 1 (no dependencies - start immediately, all parallel):
  Issue 1: Fix requirements.txt
  Issue 2: Fix audio edge cases (player.py, processor.py)
  Issue 3: Fix transcription edge cases (whisper_transcriber.py)
  Issue 4: Add .env.example + settings validation (settings.py)
  Issue 5: Add setup.sh script (new file, no overlap)
  Issue 6: Create data/ directory + app.py sample file handling

Wave 2 (depends on Wave 1 merging):
  Issue 7: Add unit test suite (tests/, pytest config)
  Issue 8: Add logging throughout codebase (replace print statements)
  Issue 9: Improve RubricScorer thread safety and state management

Wave 3 (depends on Wave 2):
  Issue 10: End-to-end validation and README polish
```

---

## Wave 1 Issues (All Parallel)

---

### Issue 1: Fix broken dependencies in requirements.txt

**Branch**: `fix/requirements-txt`
**Files**: `requirements.txt`
**Priority**: P0 (blocks installation)

**Problem**: `pydantic-ai>=1.56.0` doesn't exist (latest is 0.x). Several version pins are too strict or incompatible.

**What to do**:
- Fix `pydantic-ai` to a valid version (check PyPI for latest)
- Loosen `pydantic>=2.10` to `pydantic>=2.0,<3.0`
- Verify all packages install together without conflicts
- Pin versions that work together (tested)

**Acceptance Criteria**:
- [ ] `pip install -r requirements.txt` completes without errors in a fresh virtualenv
- [ ] `python -c "from src.analysis import IntentExtractor, RubricScorer"` succeeds
- [ ] `python -c "from src.transcription import WhisperTranscriber"` succeeds
- [ ] `python -c "from src.audio import AudioPlayer, AudioProcessor"` succeeds
- [ ] `python -c "import streamlit; import pydantic_ai; import whisper"` succeeds
- [ ] No dependency version conflicts in pip check output

---

### Issue 2: Fix edge cases in audio player and processor

**Branch**: `fix/audio-edge-cases`
**Files**: `src/audio/player.py`, `src/audio/processor.py`
**Priority**: P1

**Problems found**:
1. `player.py:94` - Audio normalization hardcodes 32768.0 (assumes 16-bit). Fails for 24/32-bit audio.
2. `player.py:97,112,123` - Silent `except: pass` blocks swallow all errors, making debugging impossible.
3. `player.py:~162` - `stream_chunks()` doesn't validate `overlap_sec >= chunk_duration_sec`, which causes infinite loop.
4. `player.py` - No validation that `sample_rate > 0`.
5. `player.py:~162` - Stereo chunk length calculation uses `len(audio)` which returns channel count for shape `[channels, samples]`.
6. `processor.py:107` - `normalize()` handles zero audio safely but doesn't log a warning.
7. `processor.py:64` - `to_mono()` uses `axis=0` which averages across rows. For `[samples, 2]` shaped audio (the convention used by `player.py`), this produces a 2-element array instead of a mono signal. Should use `axis=1`.

**What to do**:
- Replace hardcoded 32768.0 with dynamic `np.iinfo` or float detection
- Replace bare `except: pass` with `except Exception as e: logger.warning(...)` and re-raise on final backend failure
- Add validation: `overlap_sec < chunk_duration_sec` or raise `ValueError`
- Add validation: `sample_rate > 0` or raise `ValueError`
- Fix stereo chunk calculation to use correct dimension (`audio.shape[-1]` or similar)
- Fix `processor.py` `to_mono()`: change `axis=0` to `axis=1` for `[samples, channels]` convention, or handle both shapes explicitly
- Add `import logging` and use `logger = logging.getLogger(__name__)`

**Acceptance Criteria**:
- [ ] `AudioPlayer(chunk_duration_sec=3, overlap_sec=3)` raises `ValueError` with clear message
- [ ] `AudioPlayer(chunk_duration_sec=3, overlap_sec=5)` raises `ValueError`
- [ ] Loading a file that fails all 3 backends raises an exception with details (not silent)
- [ ] `AudioProcessor().normalize(np.zeros(16000))` returns zeros without error
- [ ] Stereo audio streams correct number of chunks (not 2)
- [ ] All audio loading attempts log which backend failed and why
- [ ] No bare `except: pass` remains in either file
- [ ] `AudioProcessor().to_mono(np.array([[1.0, 2.0], [3.0, 4.0]]))` returns `[1.5, 3.5]` (not `[2.0, 3.0]`)

---

### Issue 3: Fix edge cases in WhisperTranscriber

**Branch**: `fix/transcription-edge-cases`
**Files**: `src/transcription/whisper_transcriber.py`
**Priority**: P1

**Problems found**:
1. Line ~162: Division by zero if audio is all zeros (silent): `audio = audio / max(abs(audio.max()), abs(audio.min()))` where both max and min are 0.
2. Lines 74-78: Backend loading failures are caught silently, loop continues. If ALL backends fail, error message is unclear.
3. `TranscriptionResult.confidence` is never populated (always None).
4. `remove_duplicates()` only handles exact word matches - fragile.
5. `_transcript_buffer` and `_full_transcript` can get out of sync on mid-stream failures.

**What to do**:
- Guard division: check `max_val > 0` before normalizing, return zeros array if silent
- After backend loading loop, if no backend loaded, raise clear `RuntimeError` listing all attempted backends and their errors
- Add `import logging` and use `logger = logging.getLogger(__name__)` for all warnings
- Add a clear docstring noting confidence is not yet implemented (placeholder)
- Add buffer consistency: wrap `transcribe_stream` in try/except that doesn't leave partial state

**Acceptance Criteria**:
- [ ] `transcribe(np.zeros(16000), 16000)` returns empty/blank `TranscriptionResult` without error
- [ ] `transcribe(np.array([]), 16000)` returns empty result without error
- [ ] If no Whisper backend available, `load_model()` raises `RuntimeError` with message listing tried backends
- [ ] All backend load attempts are logged (not just printed)
- [ ] `reset_transcript()` fully clears both `_transcript_buffer` and `_full_transcript`
- [ ] No division-by-zero possible in any code path

---

### Issue 4: Add environment configuration (.env.example + settings validation)

**Branch**: `feat/env-configuration`
**Files**: `.env.example` (new), `src/config/settings.py`, `.gitignore`
**Priority**: P2

**Problems found**:
1. No `.env.example` to document required/optional env vars
2. `.env.example` is actively gitignored (lines 5 and 164 of `.gitignore`)
3. `settings.py` loads API key at import time, never validates
4. No validation for invalid settings (negative sample rate, zero chunk duration, etc.)
5. Paths are relative to `settings.py` - fragile if imported from different directory

**What to do**:
- **Remove `.env.example` from `.gitignore`** (it appears on lines 5 and 164 of `.gitignore` -- both must be removed)
- Create `.env.example` documenting: `OPENAI_API_KEY`, `WHISPER_MODEL`, `LLM_MODEL`, `SAMPLE_RATE`, `CHUNK_DURATION_SEC`
- Add `__post_init__` validation in Settings: sample_rate > 0, chunk_duration > 0, overlap < chunk, model name non-empty
- Use `pathlib.Path(__file__).parent` for robust path resolution (already done, verify)
- Add `load_dotenv()` call if python-dotenv available
- Show clear error message (not stack trace) when API key is missing

**Acceptance Criteria**:
- [ ] `.env.example` is NOT listed in `.gitignore` (remove from lines 5 and 164)
- [ ] `.env.example` file exists with all configurable variables documented with comments
- [ ] `cp .env.example .env` + fill in API key works with `python -c "from src.config import Settings; Settings()"`
- [ ] `Settings(sample_rate=0)` raises `ValueError`
- [ ] `Settings(chunk_duration_sec=-1)` raises `ValueError`
- [ ] `Settings(overlap_duration_sec=5, chunk_duration_sec=3)` raises `ValueError`
- [ ] Missing API key produces a clear one-line error, not a traceback

---

### Issue 5: Add setup.sh script

**Branch**: `feat/setup-script`
**Files**: `setup.sh` (new)
**Priority**: P1

**What to do**:
- Create `setup.sh` that:
  1. Checks for Python 3.9+
  2. Creates virtualenv in `venv/`
  3. Installs `requirements.txt`
  4. Checks for `ffmpeg` and warns if missing
  5. Creates `data/` directory if not exists
  6. Prompts for OpenAI API key and writes to `.env`
  7. Prints success message with "how to run" instructions
- Make it work on both macOS and Linux
- Use `#!/usr/bin/env bash` and `set -e`

**Acceptance Criteria**:
- [ ] `chmod +x setup.sh && ./setup.sh` runs end-to-end on Linux
- [ ] Script creates `venv/` directory with working Python
- [ ] Script runs `pip install -r requirements.txt` successfully
- [ ] If ffmpeg is not installed, script prints clear warning (not fatal)
- [ ] Script creates `data/` directory
- [ ] Script prompts for API key and saves to `.env`
- [ ] Script prints final instructions showing how to run the app
- [ ] Script is idempotent (running twice doesn't break anything)
- [ ] `shellcheck setup.sh` passes without errors

---

### Issue 6: Create data/ directory and improve sample file handling in app.py

**Branch**: `feat/data-directory`
**Files**: `data/.gitkeep` (new), `app.py` (sidebar section only)
**Priority**: P1

**What to do**:
- Create `data/` directory with `.gitkeep`
- Add `data/*.opus`, `data/*.wav`, `data/*.mp3` to `.gitignore`
- In `app.py` sidebar, improve the sample file detection:
  - Scan `data/` for audio files (not just hardcoded `sample_call.opus`)
  - Show dropdown if multiple files found
  - Show clear message if no files in `data/` ("Place audio files in data/ directory")
  - Upload widget always available as alternative

**Acceptance Criteria**:
- [ ] `data/` directory exists in repo (with `.gitkeep`)
- [ ] Audio files in `data/` are gitignored
- [ ] Placing any `.opus`/`.wav`/`.mp3` file in `data/` makes it appear in the app sidebar
- [ ] If `data/` is empty, app shows helpful message instead of error
- [ ] File upload still works regardless of `data/` contents
- [ ] App starts without errors when `data/` is empty

---

## Wave 2 Issues (After Wave 1 merges)

---

### Issue 7: Add unit test suite with mocked dependencies

**Branch**: `feat/test-suite`
**Files**: `tests/` (new directory), `pyproject.toml` or `pytest.ini` (new)
**Priority**: P1
**Depends on**: Issues 1, 2, 3

**What to do**:
- Create `tests/` directory structure:
  ```
  tests/
  ├── __init__.py
  ├── conftest.py          # Shared fixtures, mock API key
  ├── test_audio_player.py
  ├── test_audio_processor.py
  ├── test_whisper_transcriber.py
  ├── test_intent_extractor.py
  ├── test_rubric_scorer.py
  ├── test_models.py
  └── test_settings.py
  ```
- Add `pytest` and `pytest-cov` to requirements.txt (or dev-requirements.txt)
- Create `conftest.py` with:
  - Fixture that sets `OPENAI_API_KEY=test-key`
  - Fixture that mocks Whisper model loading
  - Sample audio arrays (zeros, sine wave, stereo)
- Write tests for:
  - `AudioProcessor`: resample, to_mono, normalize, prepare_for_whisper
  - `AudioPlayer`: chunk generation, overlap handling, edge cases from Issue 2
  - `WhisperTranscriber`: mock transcription, duplicate removal, edge cases from Issue 3
  - `IntentExtractor`: mock PydanticAI agent, extract_sync
  - `RubricScorer`: mock evaluation, score calculation, YAML loading
  - `models.py`: AnalysisTriggerState.should_analyze, RubricResult.completion_percentage
  - `Settings`: validation, defaults

**Acceptance Criteria**:
- [ ] `pytest tests/` passes with 0 failures
- [ ] Tests run without OpenAI API key set (all LLM calls mocked)
- [ ] Tests run without Whisper model / GPU (all model loading mocked)
- [ ] Tests run without ffmpeg installed (audio loading mocked where needed)
- [ ] `pytest tests/ --co` shows at least 20 test cases
- [ ] `pytest tests/ --cov=src --cov-report=term` shows >70% coverage on core modules
- [ ] Tests complete in under 30 seconds (no real API calls, no real model loading)

---

### Issue 8: Replace print statements with proper logging

**Branch**: `feat/add-logging`
**Files**: `src/analysis/intent_extractor.py`, `src/analysis/rubric_scorer.py`, `src/transcription/whisper_transcriber.py`, `src/audio/player.py`, `app.py`
**Priority**: P2
**Depends on**: Issues 2, 3

**What to do**:
- Add `import logging` and `logger = logging.getLogger(__name__)` to each module
- Replace all `print()` statements with appropriate log levels:
  - Errors: `logger.error()` (LLM failures, audio load failures)
  - Warnings: `logger.warning()` (fallback used, silent audio, missing config)
  - Info: `logger.info()` (model loaded, analysis triggered, file loaded)
  - Debug: `logger.debug()` (chunk details, timing, intermediate results)
- Configure basic logging in `app.py` entry point: `logging.basicConfig(level=logging.INFO)`
- Do NOT add logging to `models.py` or `settings.py` (they don't have print statements)

**Acceptance Criteria**:
- [ ] Zero `print()` calls remain in `src/` directory (grep for `print(`)
- [ ] Each module in `src/` uses `logger = logging.getLogger(__name__)`
- [ ] Running `streamlit run app.py` shows structured log output (not bare prints)
- [ ] Error conditions log at ERROR level
- [ ] Backend fallbacks log at WARNING level
- [ ] Setting `LOG_LEVEL=DEBUG` in env produces verbose output

---

### Issue 9: Fix RubricScorer state management and thread safety

**Branch**: `fix/rubric-scorer-state`
**Files**: `src/analysis/rubric_scorer.py`
**Priority**: P2
**Depends on**: Issue 1

**Problems found**:
1. `_completed_items` and `_evidence` dicts accumulate forever - items marked complete can never be uncompleted, even across different calls/sessions
2. Mutable state (`_completed_items`, `_evidence`) is not thread-safe - concurrent `evaluate()` calls could corrupt state
3. Error handling silently prints instead of raising
4. No timeout on LLM calls - could hang indefinitely
5. Creates new PydanticAI Agent on every `evaluate()` call (wasteful)

**What to do**:
- Add `threading.Lock` around `_completed_items` and `_evidence` mutations
- Add `reset()` call documentation - make it clear when to reset between sessions
- Reuse the PydanticAI Agent instance (create once in `__init__`, reuse in `evaluate`)
- Add timeout parameter to LLM calls (default 30s)
- Replace error printing with raising or logging

**Acceptance Criteria**:
- [ ] `RubricScorer.reset()` clears all state (completed items, evidence, scores)
- [ ] Concurrent `evaluate()` calls from different threads don't corrupt state
- [ ] LLM calls have a configurable timeout (default 30 seconds)
- [ ] PydanticAI Agent is created once and reused (not re-created per call)
- [ ] Failed evaluations raise exceptions or log errors (no silent `print`)

---

## Wave 3 Issues (After Wave 2)

---

### Issue 10: End-to-end validation and README update

**Branch**: `chore/e2e-validation`
**Files**: `README.md`, `TODO.md`
**Priority**: P0 (final gate)
**Depends on**: All previous issues

**What to do**:
- Update README.md:
  - Verified quick start instructions that actually work
  - Add "Development" section with test commands
  - Add "Configuration" section referencing `.env.example`
  - Update project structure to match current state
  - Add troubleshooting section (ffmpeg, API key, common errors)
- Validate demo flow:
  - `streamlit run demo.py` starts and works
  - `streamlit run app.py` starts without crashes
  - Upload audio file works
  - Sample file from `data/` works
  - Transcript, extraction, and rubric all update
- Update or remove `TODO.md` items that are completed
- Update `ARCHITECTURE.md` if anything changed

**Acceptance Criteria**:
- [ ] `streamlit run demo.py` starts without errors
- [ ] Clicking "Start Demo" shows transcript appearing in real-time
- [ ] Extracted info panel updates with: caller name, company, intent, sentiment
- [ ] Rubric checkmarks appear progressively as conversation proceeds
- [ ] Final score displays when demo completes
- [ ] No Python errors or tracebacks in console during full demo run
- [ ] `streamlit run app.py` starts without errors (even with no audio file)
- [ ] README quick start can be followed by a new user (clone -> setup -> run -> see demo)
- [ ] `pytest tests/` still passes after all changes merged
- [ ] TODO.md reflects current state (completed items checked off or removed)

---

## File Ownership Matrix (Conflict Prevention)

| File | Issue 1 | Issue 2 | Issue 3 | Issue 4 | Issue 5 | Issue 6 | Issue 7 | Issue 8 | Issue 9 | Issue 10 |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|
| `requirements.txt` | **PRIMARY** | | | | | | minor | | | |
| `src/audio/player.py` | | **PRIMARY** | | | | | | minor | | |
| `src/audio/processor.py` | | **PRIMARY** | | | | | | | | |
| `src/transcription/whisper_transcriber.py` | | | **PRIMARY** | | | | | minor | | |
| `src/config/settings.py` | | | | **PRIMARY** | | | | | | |
| `.env.example` | | | | **PRIMARY** | | | | | | |
| `.gitignore` | | | | **PRIMARY** | | minor | | | | |
| `setup.sh` | | | | | **PRIMARY** | | | | | |
| `data/.gitkeep` | | | | | | **PRIMARY** | | | | |
| `app.py` | | | | | | **PRIMARY** | | minor | | minor |
| `tests/` | | | | | | | **PRIMARY** | | | |
| `src/analysis/rubric_scorer.py` | | | | | | | | minor | **PRIMARY** | |
| `src/analysis/intent_extractor.py` | | | | | | | | minor | | |
| `README.md` | | | | | | | | | | **PRIMARY** |
| `TODO.md` | | | | | | | | | | **PRIMARY** |

**PRIMARY** = main changes, **minor** = small additions (e.g., adding logging import)

---

## Jules Assignment Commands

```bash
# Wave 1 - all parallel
gh issue create --title "Fix broken dependencies in requirements.txt" --label "P0,wave-1" --body "..."
gh issue create --title "Fix edge cases in audio player and processor" --label "P1,wave-1" --body "..."
gh issue create --title "Fix edge cases in WhisperTranscriber" --label "P1,wave-1" --body "..."
gh issue create --title "Add .env.example and settings validation" --label "P2,wave-1" --body "..."
gh issue create --title "Add setup.sh script" --label "P1,wave-1" --body "..."
gh issue create --title "Create data/ directory and improve sample file handling" --label "P1,wave-1" --body "..."

# Wave 2 - after Wave 1 merges
gh issue create --title "Add unit test suite with mocked dependencies" --label "P1,wave-2" --body "..."
gh issue create --title "Replace print statements with proper logging" --label "P2,wave-2" --body "..."
gh issue create --title "Fix RubricScorer state management and thread safety" --label "P2,wave-2" --body "..."

# Wave 3 - final
gh issue create --title "End-to-end validation and README update" --label "P0,wave-3" --body "..."
```
