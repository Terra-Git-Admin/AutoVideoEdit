# AutoVideoEdit — Integration Guide

## What it does

Takes AI-generated videos that contain noisy/gibberish audio, trims each to only the intended dialogue, and plays them in sequence in the browser.

**Pipeline per video:**
1. **Whisper** (local) — transcribes audio with word-level timestamps
2. **Claude API** — compares transcript against intended script, outputs keep-ranges in seconds
3. **ffmpeg** — cuts and concatenates exactly those segments into the final video

---

## Stack

| Component | Version | Purpose |
|---|---|---|
| Python | 3.13 | Runtime |
| FastAPI + uvicorn | 0.135 / 0.44 | HTTP server |
| openai-whisper | 20250625 | Local speech-to-text with word-level timestamps |
| anthropic SDK | 0.89 | Claude API for AI editing decisions |
| auto-editor | 30.1.0 | Installed for ffmpeg only (brew pulls ffmpeg as a dep) |
| ffmpeg | 8.1 | Actual video cutting and concatenation |

> **Note:** auto-editor is not used for rendering. It was abandoned because it applies its own silence detection on top of explicit cuts, producing incorrect output durations. ffmpeg cuts at exact timestamps.

---

## Setup

```bash
# 1. Install auto-editor (pulls in ffmpeg as a dependency)
brew install auto-editor

# 2. Create Python virtual environment
python3.13 -m venv .venv
.venv/bin/pip install fastapi "uvicorn[standard]" python-multipart \
    aiofiles openai-whisper anthropic

# 3. Set API key
#    Must be a real key from console.anthropic.com
#    The Claude Code session key returns 401 for direct API calls
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# 4. Run
ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" .venv/bin/uvicorn server:app --port 8000

# 5. Open
open http://localhost:8000
```

---

## Files

```
AutoVideoEdit/
├── server.py          — FastAPI backend, full pipeline
├── system_prompt.txt  — AI system prompt loaded at startup (swap to replace Claude's instructions)
├── index.html         — Single-page frontend (served at GET /)
├── requirements.txt   — Python dependencies
├── server.log         — Runtime log, written per job (auto-created)
└── .venv/             — Python virtual environment
```

**Runtime dirs** (auto-created under `/tmp/autovideoedit/`):
- `uploads/` — raw uploaded videos
- `outputs/` — trimmed final videos
- `timelines/` — Claude's keep-range decision JSON per job (useful for debugging)

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serves `index.html` |
| `POST` | `/upload` | Accept videos + metadata, start processing |
| `GET` | `/status` | Poll for per-job status and durations |
| `GET` | `/videos/{job_id}` | Stream processed video (range-request capable) |

### POST /upload — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `files[]` | video files | Up to 5 videos |
| `sequences[]` | int per file | Playback order (1–5, no duplicates) |
| `intended_dialogues[]` | string per file | Target script — what should be said in this video |
| `shot_summaries[]` | string per file | Optional scene context |
| `rules` | string | Global editing rules applied to all videos |

### GET /status — response per job

```json
{
  "sequence": 1,
  "original_name": "clip.mp4",
  "status": "done",
  "status_detail": "",
  "original_duration": 4.01,
  "kept_duration": 1.34,
  "error": null
}
```

`kept_duration` is the **actual rendered file duration** measured by ffprobe — not an estimate.

**Status lifecycle:** `pending → transcribing → ai_editing → rendering → done | error`

---

## Production Integration (bring your own AI)

If your team has its own AI backend and frontend, you can bypass Claude entirely using **manual mode**. The server runs Whisper, returns the transcript, and waits. Your code calls your own AI, then POSTs the keep ranges back to trigger ffmpeg rendering.

### Pipeline (manual mode)

```
POST /upload  (include mode=manual)
  → status: pending → transcribing → awaiting_timeline

GET /status
  → transcript field is populated (Whisper output, word-level timestamps)

[your AI backend call with transcript] → keep ranges [{start, end}, ...]

POST /jobs/{job_id}/render
  Body: {"keep": [{"start": 1.02, "end": 3.45}, ...]}
  → status: rendering → done | error

GET /videos/{job_id}  → rendered video stream
```

### POST /upload — manual mode

Add `mode=manual` as a form field alongside the existing fields. All other fields (`sequences[]`, `intended_dialogues[]`, `shot_summaries[]`, `rules`) are accepted but `intended_dialogues` and `rules` are ignored in manual mode — your AI handles editing logic.

### GET /status — transcript field

When a job is in `awaiting_timeline` status, the response includes:

```json
{
  "job_id": {
    "status": "awaiting_timeline",
    "transcript": [
      {
        "start": 0.0,
        "end": 1.8,
        "text": "Welcome to the product demo",
        "words": [
          {"word": "Welcome", "start": 0.0, "end": 0.42},
          {"word": "to", "start": 0.42, "end": 0.55},
          ...
        ]
      }
    ]
  }
}
```

### POST /jobs/{job_id}/render

Submit keep ranges to trigger ffmpeg:

```
POST /jobs/{job_id}/render
Content-Type: application/json

{"keep": [{"start": 0.0, "end": 1.8}]}
```

Returns `{"status": "rendering"}`. Poll `GET /status` until `done`, then fetch the video.

### Status lifecycle

| Mode | Lifecycle |
|------|-----------|
| `auto` (default) | `pending → transcribing → ai_editing → rendering → done \| error` |
| `manual` | `pending → transcribing → awaiting_timeline → rendering → done \| error` |

### AI output format (what your AI must return)

Your AI backend must return keep ranges in this exact JSON shape:

```json
{"keep": [{"start": 1.02, "end": 3.45}, {"start": 5.10, "end": 7.80}]}
```

- `start` and `end` are floats in **seconds**
- Ranges must not overlap; order does not matter (server sorts them)
- Ranges are clamped to `[0, video_duration]` automatically
- Empty `keep` array is valid (produces a zero-length output — handle upstream)

The server passes this directly to ffmpeg as `-ss {start} -t {duration}` per segment.

### Swapping the AI system prompt

The Claude system prompt lives in `system_prompt.txt` in the project root. It is loaded once at server startup. To replace it:

1. Edit or overwrite `system_prompt.txt` with your own instructions
2. Restart the server — no code changes needed

In production (`manual` mode) this file is irrelevant since Claude is bypassed entirely. It only matters for the `auto` mode dev workflow.

### Notes for integration

- `intended_dialogues[]` and `shot_summaries[]` are optional in manual mode — omit them or pass empty strings.
- The frontend (`index.html`) is purely a thin client over the REST API. Replace it with any frontend or remove it entirely — the API contract doesn't change.
- The rendered video is served at `GET /videos/{job_id}` regardless of mode. Your frontend can use the same endpoint.
- The dev localhost still works with the default `auto` mode — no changes to the frontend needed.

---

## How Claude edits

Claude receives per video:
- **Intended dialogue** — the target script (primary signal)
- **Whisper transcript** — what was actually said, with segment and word-level timestamps
- **Shot context** — optional scene description
- **Editing rules** — global rules (cut fillers, cut pauses, etc.)

Claude outputs:
```json
{"keep": [{"start": 1.02, "end": 1.34}]}
```

These are passed to ffmpeg as `-ss {start} -t {duration}` per segment.

### Dialogue conventions Claude understands

**Trailing dash = interrupted line**
```
Intended:   "How dare-"
Transcript: "How" [0.00–1.02s]  "dare" [1.02–1.34s]  "you!" [1.34–1.80s]
Result:     keep 0.00–1.34s, cut "you!"
```

**Partial repetition / stutter**
```
Intended:   "Stay out of it princess"
Transcript: "Stay out, stay out of it, princess."
Result:     keep only the first clean complete rendition, cut all earlier partial attempts
```

---

## Integration notes

1. **Whisper model** — defaults to `base` (~74MB, downloads on first use). Swap to `small` or `medium` in `get_whisper_model()` for better accuracy at the cost of speed.

2. **Word-level timestamps** — `word_timestamps=True` in Whisper gives per-word `{word, start, end}`. Passed to Claude so it can cut at word boundaries, not just segment boundaries.

3. **In-memory job store** — `jobs` dict is cleared on each new upload and lost on server restart. Replace with Redis or a DB for persistence or concurrent multi-user sessions.

4. **Single-session assumption** — `jobs.clear()` on every upload means only one session at a time. Add a session ID to support multiple concurrent users.

5. **Background tasks** — FastAPI's `BackgroundTasks` runs jobs in a thread pool. Whisper + Claude + ffmpeg per video takes 10–60s depending on length. For production, replace with Celery or a job queue.

6. **File storage** — uploads and outputs go to `/tmp` (wiped on reboot). Point `UPLOAD_DIR` / `OUTPUT_DIR` to persistent storage for production.

7. **ANTHROPIC_API_KEY** — must be from [console.anthropic.com](https://console.anthropic.com). Pass it explicitly at server startup; do not rely on shell inheritance when running as a background process.

8. **Debugging** — every run appends to `server.log` with full detail: Whisper words per segment, Claude's raw JSON response, ffmpeg commands with exact timestamps, actual vs estimated durations.
