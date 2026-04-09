import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List

# Whisper/Numba are not thread-safe — serialize all transcription calls
_whisper_lock = threading.Lock()

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

UPLOAD_DIR = Path("/tmp/autovideoedit/uploads")
OUTPUT_DIR = Path("/tmp/autovideoedit/outputs")
TIMELINE_DIR = Path("/tmp/autovideoedit/timelines")
LOG_FILE = Path(__file__).parent / "server.log"

# File + console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("autovideoedit")

# In-memory job store
# {job_id: {original_name, sequence, status, status_detail, output_file, shot_summary, error}}
jobs: Dict[str, dict] = {}

AE_BIN: str = ""
_whisper_model = None  # lazy-loaded on first use


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global AE_BIN
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Logging to {LOG_FILE}")

    AE_BIN = shutil.which("auto-editor") or ""
    if AE_BIN:
        log.info(f"auto-editor found: {AE_BIN}")
    else:
        log.error("auto-editor not found. Run: brew install auto-editor")

    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        log.info(f"GEMINI_API_KEY present (prefix: {key[:12]}...)")
    else:
        log.error("GEMINI_API_KEY not set — AI editing will fail")


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_videos(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    sequences: List[int] = Form(...),
    job_ids: List[str] = Form(...),              # frontend provides job IDs
    intended_dialogues: List[str] = Form(None),   # optional in manual mode
    shot_summaries: List[str] = Form(None),        # optional in manual mode
    rules: str = Form(""),
    mode: str = Form("auto"),  # "auto" (Whisper+Gemini+ffmpeg) or "manual" (Whisper only, caller submits keep ranges)
):
    if len(files) > 5:
        raise HTTPException(400, "Max 5 files allowed")
    if len(files) != len(sequences):
        raise HTTPException(400, "File / sequence count mismatch")
    if len(files) != len(job_ids):
        raise HTTPException(400, "File / job_id count mismatch")
    if len(set(sequences)) != len(sequences):
        raise HTTPException(400, "Duplicate sequence numbers")
    if len(set(job_ids)) != len(job_ids):
        raise HTTPException(400, "Duplicate job IDs")

    # Pad optional fields to match file count
    n = len(files)
    dialogues = list(intended_dialogues) if intended_dialogues else [""] * n
    summaries = list(shot_summaries) if shot_summaries else [""] * n
    if len(dialogues) != n or len(summaries) != n:
        raise HTTPException(400, "intended_dialogues / shot_summaries count must match file count when provided")

    # Purge finished jobs from previous batches to keep memory clean
    stale = [jid for jid, j in jobs.items() if j["status"] in ("done", "error")]
    for jid in stale:
        del jobs[jid]
    if stale:
        log.info(f"Purged {len(stale)} completed job(s) from memory")

    log.info(f"=== NEW UPLOAD BATCH === {len(files)} file(s), mode={mode}, rules={rules.strip()[:100] or '(default)'}")

    for file, seq, job_id, intended_dialogue, summary in zip(files, sequences, job_ids, dialogues, summaries):
        safe_filename = Path(file.filename).name if file.filename else "video.mp4"
        upload_path = UPLOAD_DIR / f"{job_id}_{safe_filename}"
        output_path = OUTPUT_DIR / f"{job_id}_{safe_filename}"

        log.info(f"[{job_id[:8]}] Receiving file '{safe_filename}' seq={seq}")

        bytes_written = 0
        async with aiofiles.open(upload_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                await out.write(chunk)
                bytes_written += len(chunk)

        log.info(f"[{job_id[:8]}] Saved to {upload_path} ({bytes_written / 1024:.1f} KB)")

        jobs[job_id] = {
            "original_name": file.filename,
            "sequence": seq,
            "status": "pending",
            "status_detail": "Waiting to start…",
            "upload_file": str(upload_path),
            "output_file": str(output_path),
            "shot_summary": summary,
            "intended_dialogue": intended_dialogue,
            "original_duration": None,
            "kept_duration": None,
            "transcript": None,
            "mode": mode,
            "error": None,
        }
        background_tasks.add_task(
            process_video, job_id, str(upload_path), str(output_path),
            intended_dialogue, summary, rules, mode
        )
        log.info(f"[{job_id[:8]}] Queued for processing")

    log.info(f"=== BATCH ACCEPTED === job_ids: {list(jobs.keys())}")
    return {"job_ids": list(jobs.keys())}


# ── Pipeline ─────────────────────────────────────────────────────────────────

def set_status(job_id: str, status: str, detail: str = ""):
    prev = jobs[job_id]["status"]
    jobs[job_id]["status"] = status
    jobs[job_id]["status_detail"] = detail
    log.info(f"[{job_id[:8]}] STATUS  {prev} → {status}  {('| ' + detail) if detail else ''}")


def process_video(
    job_id: str,
    input_path: str,
    output_path: str,
    intended_dialogue: str,
    shot_summary: str,
    rules: str,
    mode: str = "auto",
):
    tag = job_id[:8]
    log.info(f"[{tag}] ─────────────────────────────────────────────")
    log.info(f"[{tag}] PIPELINE START  file='{Path(input_path).name}'  mode={mode}")
    log.info(f"[{tag}] intended_dialogue : {intended_dialogue.strip()[:200] or '(none)'}")
    log.info(f"[{tag}] shot_summary      : {shot_summary.strip()[:200] or '(none)'}")
    log.info(f"[{tag}] rules             : {rules.strip()[:200] or '(default)'}")

    if not AE_BIN:
        msg = "auto-editor not installed. Run: brew install auto-editor"
        log.error(f"[{tag}] ABORT — {msg}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = msg
        return

    try:
        # ── Step 1: Whisper transcription ─────────────────────────────────
        log.info(f"[{tag}] STEP 1/3 — Whisper transcription")
        set_status(job_id, "transcribing", "Transcribing audio with Whisper…")
        transcript = transcribe(input_path)
        duration = get_video_duration(input_path)
        jobs[job_id]["original_duration"] = duration
        log.info(f"[{tag}] Whisper done — {len(transcript)} segment(s), video duration={duration:.2f}s")
        if transcript:
            for seg in transcript:
                log.info(f"[{tag}]   [{seg['start']:.2f}s–{seg['end']:.2f}s] \"{seg['text']}\"")
                for w in seg.get("words", []):
                    log.info(f"[{tag}]     word {w['start']:.2f}s–{w['end']:.2f}s \"{w['word']}\"")
        else:
            log.warning(f"[{tag}]   No speech detected in audio")

        # ── Manual mode: stop here ────────────────────────────────────────
        if mode == "manual":
            jobs[job_id]["transcript"] = transcript
            set_status(job_id, "awaiting_timeline", "Transcript ready — waiting for keep ranges")
            log.info(f"[{tag}] Manual mode — paused, awaiting keep ranges from caller")
            return

        # ── Step 2: Gemini decides what to keep ───────────────────────────
        log.info(f"[{tag}] STEP 2/3 — Gemini AI editing")
        set_status(job_id, "ai_editing", "Gemini is reviewing dialogue and editing…")
        keep_ranges = claude_edit(transcript, duration, intended_dialogue, shot_summary, rules)
        estimated_kept = sum(r["end"] - r["start"] for r in keep_ranges)
        log.info(f"[{tag}] Gemini decision — {len(keep_ranges)} keep range(s): {keep_ranges}")
        log.info(f"[{tag}] Kept {estimated_kept:.2f}s of {duration:.2f}s ({100*estimated_kept/duration:.1f}% retained)")

        decision_path = str(TIMELINE_DIR / f"{job_id}_decision.json")
        with open(decision_path, "w") as f:
            json.dump({"keep": keep_ranges, "duration": duration, "kept": estimated_kept}, f, indent=2)
        log.info(f"[{tag}] Decision saved → {decision_path}")

        # ── Step 3: Render with ffmpeg ────────────────────────────────────
        log.info(f"[{tag}] STEP 3/3 — ffmpeg render")
        set_status(job_id, "rendering", "Rendering final video…")
        render_with_ffmpeg(keep_ranges, input_path, output_path, tag)

        if not Path(output_path).exists():
            msg = "Output file missing after render"
            log.error(f"[{tag}] FAILED — {msg}")
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = msg
            return

        actual_duration = get_video_duration(output_path)
        jobs[job_id]["kept_duration"] = actual_duration

        # ── Full transcript text (what Whisper heard) ─────────────────────
        full_transcript = " ".join(s["text"] for s in transcript) if transcript else "(no speech)"

        # ── Cut segments (what was removed) ──────────────────────────────
        cut_ranges = keep_to_cut_ranges(keep_ranges, duration)
        total_cut = sum(e - s for s, e in cut_ranges)

        # ── Final summary ─────────────────────────────────────────────────
        log.info(f"[{tag}] ╔══════════════════════════════════════════════╗")
        log.info(f"[{tag}] ║           JOB SUMMARY                        ║")
        log.info(f"[{tag}] ╠══════════════════════════════════════════════╣")
        log.info(f"[{tag}] ║ File        : {Path(input_path).name}")
        log.info(f"[{tag}] ║ Job ID      : {job_id}")
        log.info(f"[{tag}] ║ Mode        : {mode}")
        log.info(f"[{tag}] ╠── Input ──────────────────────────────────────")
        log.info(f"[{tag}] ║ Duration    : {duration:.2f}s")
        log.info(f"[{tag}] ║ Segments    : {len(transcript)} Whisper segment(s)")
        log.info(f"[{tag}] ║ Transcript  : {full_transcript[:300]}")
        log.info(f"[{tag}] ║ Intended    : {intended_dialogue.strip()[:200] or '(none)'}")
        log.info(f"[{tag}] ║ Rules       : {rules.strip()[:200] or '(default)'}")
        log.info(f"[{tag}] ╠── Gemini Decision ────────────────────────────")
        log.info(f"[{tag}] ║ Keep ranges : {len(keep_ranges)} segment(s)")
        for i, r in enumerate(keep_ranges):
            log.info(f"[{tag}] ║   keep[{i}]  : {r['start']:.3f}s → {r['end']:.3f}s  ({r['end']-r['start']:.3f}s)")
        log.info(f"[{tag}] ║ Cut ranges  : {len(cut_ranges)} segment(s)")
        for i, (s, e) in enumerate(cut_ranges):
            log.info(f"[{tag}] ║   cut[{i}]   : {s:.3f}s → {e:.3f}s  ({e-s:.3f}s removed)")
        log.info(f"[{tag}] ╠── Result ─────────────────────────────────────")
        log.info(f"[{tag}] ║ Original    : {duration:.2f}s")
        log.info(f"[{tag}] ║ Kept        : {actual_duration:.2f}s  ({100*actual_duration/duration:.1f}% retained)")
        log.info(f"[{tag}] ║ Trimmed     : {total_cut:.2f}s  ({100*total_cut/duration:.1f}% removed)")
        log.info(f"[{tag}] ║ Output      : {Path(output_path).name}")
        log.info(f"[{tag}] ╚══════════════════════════════════════════════╝")

        set_status(job_id, "done", "")

    except Exception as e:
        log.exception(f"[{tag}] PIPELINE FAILED — unhandled exception: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_duration(input_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            input_path,
        ],
        capture_output=True, text=True, timeout=30,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def keep_to_cut_ranges(keep_ranges: list, duration: float) -> list:
    """Convert kept segments [{start, end}] to cut ranges [(start, end)]."""
    keep_ranges = sorted(keep_ranges, key=lambda r: r["start"])
    cut_ranges = []
    prev_end = 0.0
    for r in keep_ranges:
        if r["start"] > prev_end + 0.05:
            cut_ranges.append((prev_end, r["start"]))
        prev_end = max(prev_end, r["end"])
    if prev_end < duration - 0.05:
        cut_ranges.append((prev_end, duration))
    return cut_ranges


def render_with_ffmpeg(keep_ranges: list, input_path: str, output_path: str, tag: str = ""):
    """Cut and concatenate keep ranges using ffmpeg. Exact timestamps, no silence detection."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")

    tmp = Path(output_path).parent
    seg_paths = []

    for i, r in enumerate(keep_ranges):
        seg = str(tmp / f"{Path(output_path).stem}_seg{i}.mp4")
        duration = r["end"] - r["start"]
        cmd = [
            ffmpeg, "-y",
            "-ss", f"{r['start']:.3f}",
            "-i", input_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-c:a", "aac",
            "-movflags", "+faststart",
            seg,
        ]
        log.info(f"[{tag}] ffmpeg segment {i}: {r['start']:.3f}s–{r['end']:.3f}s → {seg}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg segment {i} failed: {result.stderr[-300:]}")
        seg_paths.append(seg)

    if len(seg_paths) == 1:
        Path(seg_paths[0]).rename(output_path)
        return

    # Concatenate segments
    concat_file = str(tmp / f"{Path(output_path).stem}_concat.txt")
    with open(concat_file, "w") as f:
        for seg in seg_paths:
            f.write(f"file '{seg}'\n")

    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", output_path]
    log.info(f"[{tag}] ffmpeg concat: {len(seg_paths)} segments → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-300:]}")

    for seg in seg_paths:
        Path(seg).unlink(missing_ok=True)
    Path(concat_file).unlink(missing_ok=True)


# ── Whisper ───────────────────────────────────────────────────────────────────

def transcribe(input_path: str) -> list:
    """Returns [{start, end, text, words: [{word, start, end}]}, ...] with word-level timestamps."""
    try:
        model = get_whisper_model()
    except ImportError:
        raise RuntimeError("openai-whisper not installed. Run: pip3 install openai-whisper")

    with _whisper_lock:
        result = model.transcribe(input_path, word_timestamps=True)
    segments = []
    for seg in result["segments"]:
        words = [
            {"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
            for w in (seg.get("words") or [])
        ]
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "words": words,
        })
    return segments


# ── Claude ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")


def claude_edit(
    transcript: list,
    duration: float,
    intended_dialogue: str,
    shot_summary: str,
    rules: str,
) -> list:
    import google.generativeai as genai

    def fmt_transcript(segments):
        if not segments:
            return "  (no speech detected)"
        lines = []
        for s in segments:
            lines.append(f"  [{s['start']:.2f}s–{s['end']:.2f}s] \"{s['text']}\"")
            for w in s.get("words", []):
                lines.append(f"    {w['start']:.2f}s–{w['end']:.2f}s  \"{w['word']}\"")
        return "\n".join(lines)

    transcript_block = fmt_transcript(transcript)

    default_rules = (
        "- Cut filler words and sounds (uh, um, hmm, okay so, etc.)\n"
        "- Cut pauses longer than 0.8 seconds\n"
        "- Cut incoherent speech or AI-generation gibberish\n"
        "- Keep all meaningful dialogue that matches the intended script"
    )

    intended_block = (
        intended_dialogue.strip()
        if intended_dialogue.strip()
        else "(none — use best judgement from transcript)"
    )

    user_prompt = f"""## Intended Dialogue  (target script — what SHOULD be in this video)
{intended_block}

Find where this dialogue occurs in the transcript. Keep those segments. Cut everything else — especially gibberish, filler, or words not in the intended dialogue.

## Editing Rules
{rules.strip() if rules.strip() else default_rules}

## Shot Context
{shot_summary.strip() if shot_summary.strip() else "(none)"}

## Transcript  (what Whisper actually heard, with timestamps)
{transcript_block}

## Video duration: {duration:.3f}s

Output JSON with the time ranges to KEEP."""

    api_key = os.environ.get("GEMINI_API_KEY", "")
    log.info(f"[Gemini] Sending request — key prefix: {api_key[:12] if api_key else '(MISSING!)'}")
    log.info(f"[Gemini] User prompt:\n{user_prompt}")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(user_prompt)
    log.info(f"[Gemini] Raw response:\n{response.text}")

    raw = response.text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(raw)
    keep = data.get("keep", [])

    # Clamp to valid range
    keep = [
        {"start": max(0.0, r["start"]), "end": min(duration, r["end"])}
        for r in keep
        if r["end"] > r["start"]
    ]
    return keep


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/clear")
def clear_jobs():
    count = len(jobs)
    jobs.clear()
    log.info(f"[/clear] Cleared {count} job(s) from memory")
    return {"cleared": count}


@app.get("/status")
def get_status(ids: str = ""):
    """Return status for specific job IDs (comma-separated ?ids=id1,id2) or all jobs if omitted."""
    requested = set(ids.split(",")) if ids else None
    filtered = {
        job_id: job for job_id, job in jobs.items()
        if requested is None or job_id in requested
    }
    log.info(f"[/status] returning {len(filtered)} job(s)" + (f" (filtered from {len(jobs)} total)" if requested else ""))
    return {
        job_id: {
            "sequence": job["sequence"],
            "original_name": job["original_name"],
            "status": job["status"],
            "status_detail": job.get("status_detail", ""),
            "original_duration": job.get("original_duration"),
            "kept_duration": job.get("kept_duration"),
            "transcript": job.get("transcript"),
            "error": job["error"],
        }
        for job_id, job in filtered.items()
    }


@app.post("/jobs/{job_id}/render")
def submit_timeline(job_id: str, body: dict, background_tasks: BackgroundTasks):
    """Manual mode: caller submits keep ranges; server renders with ffmpeg."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "awaiting_timeline":
        raise HTTPException(409, f"Job is not awaiting a timeline (current status: {job['status']})")

    keep_ranges = body.get("keep", [])
    if not isinstance(keep_ranges, list) or not all(
        isinstance(r, dict) and "start" in r and "end" in r for r in keep_ranges
    ):
        raise HTTPException(422, "Body must be {\"keep\": [{\"start\": float, \"end\": float}, ...]}")

    jobs[job_id]["transcript"] = None  # no longer needed
    set_status(job_id, "rendering", "Rendering with submitted timeline…")
    log.info(f"[{job_id[:8]}] Manual render requested — {len(keep_ranges)} segment(s): {keep_ranges}")
    background_tasks.add_task(
        _render_job, job_id,
        job["upload_file"], job["output_file"], keep_ranges,
    )
    return {"status": "rendering"}


def _render_job(job_id: str, input_path: str, output_path: str, keep_ranges: list):
    try:
        render_with_ffmpeg(keep_ranges, input_path, output_path, job_id[:8])
        if not Path(output_path).exists():
            raise RuntimeError("Output file missing after render")
        actual = get_video_duration(output_path)
        jobs[job_id]["kept_duration"] = actual
        log.info(f"[{job_id[:8]}] Manual render done — actual duration {actual:.2f}s")
        set_status(job_id, "done", "")
    except Exception as e:
        log.exception(f"[{job_id[:8]}] Manual render failed")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.get("/videos/{job_id}")
def serve_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(425, "Video not ready yet")
    output_path = job["output_file"]
    if not Path(output_path).exists():
        raise HTTPException(500, "Output file missing")
    return FileResponse(
        output_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )
