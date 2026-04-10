import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
import redis as redis_lib
from dotenv import load_dotenv
import firestore_client
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

load_dotenv()

JOB_TTL_SECONDS  = 2 * 60 * 60   # active/stuck jobs expire after 2 h
DONE_TTL_SECONDS = 30 * 60        # done/error jobs linger 30 min for polling & download

# Each Whisper worker thread loads its own model instance via thread-local
# storage (see get_whisper_model). 4 workers = 4 independent model copies,
# ~580 MB total for the base model — safe to run in parallel.
WHISPER_WORKERS = 4
_whisper_executor = ThreadPoolExecutor(max_workers=WHISPER_WORKERS)

# Render executor: handles Gemini + ffmpeg only (no Whisper, never blocks).
_render_executor = ThreadPoolExecutor(max_workers=8)

# ── Session scheduler ─────────────────────────────────────────────────────────

class SessionScheduler:
    """
    Round-robin scheduler across sessions. Owns per-session job queues and
    controls what enters _whisper_executor — at most WHISPER_WORKERS jobs
    in-flight at once. New batches (has_more=True) append to the session deque
    seamlessly; the round-robin picks them up on the next available slot.

    Flow:
        submit(session_id, args)
            → appends to session deque
            → calls _try_dispatch()

        _try_dispatch()
            → while in_flight < WHISPER_WORKERS and there are pending jobs:
                pick next session round-robin
                submit 1 job to _whisper_executor
                attach _on_done callback

        _on_done(future)
            → decrement in_flight
            → check if session is exhausted + fully submitted → remove from rotation
            → call _try_dispatch() to fill the freed slot immediately
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._session_queues: dict[str, list] = {}   # session_id → list of job arg tuples
        self._session_order: list[str] = []           # round-robin rotation
        self._in_flight = 0

    def submit(self, session_id: str, job_args: tuple):
        with self._lock:
            if session_id not in self._session_queues:
                self._session_queues[session_id] = []
                self._session_order.append(session_id)
            self._session_queues[session_id].append(job_args)
        self._try_dispatch()

    def _try_dispatch(self):
        while True:
            with self._lock:
                if self._in_flight >= WHISPER_WORKERS:
                    return
                job = self._pick_next()
                if job is None:
                    return
                self._in_flight += 1

            _whisper_executor.submit(_whisper_stage, *job).add_done_callback(self._on_done)

    def _pick_next(self) -> tuple | None:
        """Round-robin across sessions with pending jobs. Must be called under self._lock."""
        for _ in range(len(self._session_order)):
            sid = self._session_order[0]
            self._session_order.append(self._session_order.pop(0))  # rotate
            if self._session_queues.get(sid):
                return self._session_queues[sid].pop(0)
        return None

    def _on_done(self, future):
        with self._lock:
            self._in_flight -= 1
            # Clean up sessions whose queue is empty and all batches have arrived.
            # Checking _redis here is safe — _on_done runs in an executor thread.
            exhausted = [
                sid for sid in list(self._session_order)
                if not self._session_queues.get(sid)
                and _redis is not None
                and session_get_meta(sid).get("all_received", False)
            ]
            for sid in exhausted:
                self._session_queues.pop(sid, None)
                if sid in self._session_order:
                    self._session_order.remove(sid)
                log.info(f"[scheduler] session {sid[:8]} fully processed — removed from rotation")
        self._try_dispatch()

    def remove_session(self, session_id: str):
        """Called when a session is fully submitted and its queue is drained."""
        with self._lock:
            self._session_queues.pop(session_id, None)
            if session_id in self._session_order:
                self._session_order.remove(session_id)


scheduler = SessionScheduler()

# ── Directories ───────────────────────────────────────────────────────────────

UPLOAD_DIR   = Path("/tmp/autovideoedit/uploads")
OUTPUT_DIR   = Path("/tmp/autovideoedit/outputs")
TIMELINE_DIR = Path("/tmp/autovideoedit/timelines")
LOG_FILE     = Path(__file__).parent / "server.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("autovideoedit")

AE_BIN: str = ""
_thread_local = threading.local()  # each Whisper worker thread holds its own model instance
_redis: redis_lib.Redis | None = None


# ── Redis helpers ─────────────────────────────────────────────────────────────

def _rkey(job_id: str) -> str:
    return f"job:{job_id}"


def job_get(job_id: str) -> dict | None:
    assert _redis is not None
    raw = _redis.get(_rkey(job_id))
    return json.loads(raw) if raw else None


def job_set(job_id: str, data: dict):
    assert _redis is not None
    ttl = DONE_TTL_SECONDS if data.get("status") in ("done", "error") else JOB_TTL_SECONDS
    _redis.setex(_rkey(job_id), ttl, json.dumps(data))


def job_update(job_id: str, **fields):
    data = job_get(job_id)
    if data is None:
        log.error(f"job_update: job {job_id[:8]} not found in Redis (expired?), skipping update: {fields}")
        return
    data.update(fields)
    job_set(job_id, data)


def job_delete(job_id: str):
    assert _redis is not None
    _redis.delete(_rkey(job_id))


def jobs_get_many(ids: list[str]) -> dict[str, dict]:
    assert _redis is not None
    if not ids:
        return {}
    values = _redis.mget([_rkey(jid) for jid in ids])
    return {jid: json.loads(v) for jid, v in zip(ids, values) if v}


def jobs_get_all() -> dict[str, dict]:
    assert _redis is not None
    result = {}
    for key in _redis.scan_iter("job:*"):
        raw = _redis.get(key)
        if raw:
            jid = key.removeprefix("job:")
            result[jid] = json.loads(raw)
    return result


# ── Session index helpers ─────────────────────────────────────────────────────

def _skey(session_id: str) -> str:
    return f"session:{session_id}"

def _smkey(session_id: str) -> str:
    return f"session_meta:{session_id}"


def session_add_jobs(session_id: str, job_ids: list[str]):
    """SADD job_ids to the session set. Refreshes TTL to JOB_TTL_SECONDS."""
    assert _redis is not None
    _redis.sadd(_skey(session_id), *job_ids)
    _redis.expire(_skey(session_id), JOB_TTL_SECONDS)


def session_get_jobs(session_id: str) -> list[str]:
    """Return all job_ids registered to this session."""
    assert _redis is not None
    return list(_redis.smembers(_skey(session_id)))


def session_set_meta(session_id: str, **fields):
    """Merge fields into session metadata (stored as JSON string)."""
    assert _redis is not None
    key = _smkey(session_id)
    raw = _redis.get(key)
    data = json.loads(raw) if raw else {}
    data.update(fields)
    _redis.setex(key, JOB_TTL_SECONDS, json.dumps(data))


def session_get_meta(session_id: str) -> dict:
    assert _redis is not None
    raw = _redis.get(_smkey(session_id))
    return json.loads(raw) if raw else {}


# ── Startup / shutdown ────────────────────────────────────────────────────────

def get_whisper_model():
    # Each executor thread loads its own model on first use, then reuses it forever.
    # Thread-local storage persists for the lifetime of the thread — ThreadPoolExecutor
    # keeps threads alive until shutdown(), so this loads exactly once per worker.
    if not hasattr(_thread_local, "model"):
        import whisper
        log.info(f"[Whisper] Loading model on thread {threading.current_thread().name}")
        _thread_local.model = whisper.load_model("base")
    return _thread_local.model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global AE_BIN, _redis

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Logging to {LOG_FILE}")

    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url:
        _redis = redis_lib.from_url(redis_url, decode_responses=True)
        _redis.ping()
        log.info(f"Redis connected: {redis_url.split('@')[-1]}")
    else:
        log.error("REDIS_URL not set — job store unavailable")

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

    firestore_client.init_firestore()

    yield

    _whisper_executor.shutdown(wait=False)
    _render_executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_videos(
    files: list[UploadFile] = File(...),
    sequences: list[int] = Form(...),
    job_ids: list[str] = Form(...),
    intended_dialogues: list[str] = Form(None),
    shot_summaries: list[str] = Form(None),
    rules: str = Form(""),
    mode: str = Form("auto"),
    session_id: str = Form(...),  # caller generates one UUID per episode; same for all batches
    has_more: bool = Form(True),  # False on the last batch — tells backend all jobs are now known
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

    n = len(files)
    dialogues = list(intended_dialogues) if intended_dialogues else [""] * n
    summaries = list(shot_summaries) if shot_summaries else [""] * n
    if len(dialogues) != n or len(summaries) != n:
        raise HTTPException(400, "intended_dialogues / shot_summaries count must match file count when provided")

    if _redis is None:
        raise HTTPException(503, "Redis unavailable — cannot process uploads")

    # ── Pre-upload snapshot ───────────────────────────────────────────────────
    existing = jobs_get_many(job_ids)
    log.info(f"")
    log.info(f"╔══════════════════════════════════════════════════════════════╗")
    log.info(f"║                   UPLOAD REQUEST RECEIVED                   ║")
    log.info(f"╠══════════════════════════════════════════════════════════════╣")
    log.info(f"║ session_id      : {session_id}")
    log.info(f"║ Incoming batch  : {len(files)} file(s)  mode={mode}")
    log.info(f"║ Incoming job IDs: {job_ids}")
    log.info(f"║ Already in Redis ({len(existing)} of {len(job_ids)} known):")
    for jid, j in existing.items():
        log.info(f"║   [{jid[:8]}] status={j['status']:<20} name={j.get('original_name', '?')}")
    log.info(f"╚══════════════════════════════════════════════════════════════╝")
    log.info(f"=== NEW UPLOAD BATCH === {len(files)} file(s), mode={mode}, rules={rules.strip()[:100] or '(default)'}")

    for file, seq, job_id, intended_dialogue, summary in zip(files, sequences, job_ids, dialogues, summaries):
        safe_filename = Path(file.filename).name if file.filename else "video.mp4"
        upload_path = UPLOAD_DIR / f"{job_id}_{safe_filename}"
        output_path = OUTPUT_DIR / f"{job_id}_{safe_filename}"

        log.info(f"[{job_id[:8]}] Receiving file '{safe_filename}' seq={seq}")

        bytes_written = 0
        async with aiofiles.open(upload_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await out.write(chunk)
                bytes_written += len(chunk)

        log.info(f"[{job_id[:8]}] Saved to {upload_path} ({bytes_written / 1024:.1f} KB)")

        job_set(job_id, {
            "original_name": file.filename,
            "sequence": seq,
            "status": "pending",
            "status_detail": "Waiting to start…",
            "upload_file": str(upload_path),
            "output_file": str(output_path),
            "shot_summary": summary,
            "intended_dialogue": intended_dialogue,
            "session_id": session_id,
            "original_duration": None,
            "kept_duration": None,
            "transcript": None,
            "mode": mode,
            "error": None,
            "created_at": time.time(),
        })
        scheduler.submit(
            session_id,
            (job_id, str(upload_path), str(output_path), intended_dialogue, summary, rules, mode),
        )
        log.info(f"[{job_id[:8]}] Queued via scheduler (session={session_id[:8]})")

    # ── Session index + has_more tracking ────────────────────────────────────
    session_add_jobs(session_id, job_ids)   # SADD this batch into session:{session_id} set

    if not has_more:
        # All batches received — record the final total so /status can report "X of Y done"
        total = _redis.scard(_skey(session_id))
        session_set_meta(session_id, all_received=True, total_submitted=int(total))
        log.info(f"[session={session_id[:8]}] Last batch — {total} job(s) total for this session")
    else:
        log.info(f"[session={session_id[:8]}] Batch accepted, more batches expected")

    log.info(f"=== BATCH ACCEPTED === job_ids: {job_ids}")
    return {"job_ids": job_ids}


# ── Pipeline ─────────────────────────────────────────────────────────────────

def set_status(job_id: str, status: str, detail: str = ""):
    data = job_get(job_id) or {}
    prev = data.get("status", "?")
    data["status"] = status
    data["status_detail"] = detail
    job_set(job_id, data)
    log.info(f"[{job_id[:8]}] STATUS  {prev} → {status}  {('| ' + detail) if detail else ''}")


def _whisper_stage(
    job_id: str,
    input_path: str,
    output_path: str,
    intended_dialogue: str,
    shot_summary: str,
    rules: str,
    mode: str = "auto",
):
    """Stage 1: runs inside _whisper_executor. Transcribes, then chains to _render_stage."""
    tag = job_id[:8]
    log.info(f"[{tag}] ─────────────────────────────────────────────")
    log.info(f"[{tag}] PIPELINE START  file='{Path(input_path).name}'  mode={mode}")
    log.info(f"[{tag}] intended_dialogue : {intended_dialogue.strip()[:200] or '(none)'}")
    log.info(f"[{tag}] shot_summary      : {shot_summary.strip()[:200] or '(none)'}")
    log.info(f"[{tag}] rules             : {rules.strip()[:200] or '(default)'}")

    if not AE_BIN:
        msg = "auto-editor not installed. Run: brew install auto-editor"
        log.error(f"[{tag}] ABORT — {msg}")
        job_update(job_id, status="error", error=msg)
        return

    try:
        log.info(f"[{tag}] STEP 1/3 — Whisper transcription")
        set_status(job_id, "transcribing", "Transcribing audio with Whisper…")
        transcript = transcribe(input_path)
        duration = get_video_duration(input_path)
        job_update(job_id, original_duration=duration)
        log.info(f"[{tag}] Whisper done — {len(transcript)} segment(s), video duration={duration:.2f}s")
        if transcript:
            for seg in transcript:
                log.info(f"[{tag}]   [{seg['start']:.2f}s–{seg['end']:.2f}s] \"{seg['text']}\"")
                for w in seg.get("words", []):
                    log.info(f"[{tag}]     word {w['start']:.2f}s–{w['end']:.2f}s \"{w['word']}\"")
        else:
            log.warning(f"[{tag}]   No speech detected in audio")

        if mode == "manual":
            job_update(job_id, transcript=transcript)
            set_status(job_id, "awaiting_timeline", "Transcript ready — waiting for keep ranges")
            log.info(f"[{tag}] Manual mode — paused, awaiting keep ranges from caller")
            return

        # Whisper done — hand off to render executor immediately.
        # This frees this Whisper worker to start the next job in the scheduler queue.
        _render_executor.submit(
            _render_stage, job_id, input_path, output_path,
            transcript, duration, intended_dialogue, shot_summary, rules, mode,
        )

    except Exception as e:
        log.exception(f"[{tag}] WHISPER STAGE FAILED: {e}")
        job_update(job_id, status="error", error=str(e))


def _render_stage(
    job_id: str,
    input_path: str,
    output_path: str,
    transcript: list,
    duration: float,
    intended_dialogue: str,
    shot_summary: str,
    rules: str,
    mode: str = "auto",
):
    """Stage 2: runs inside _render_executor. Gemini + ffmpeg, fully parallel across jobs."""
    tag = job_id[:8]
    try:
        # ── Step 2: Gemini decides what to keep ───────────────────────────
        log.info(f"[{tag}] STEP 2/3 — Gemini AI editing")
        set_status(job_id, "ai_editing", "Gemini is reviewing dialogue and editing…")
        keep_ranges = claude_edit(transcript, duration, intended_dialogue, shot_summary, rules)
        estimated_kept = sum(r["end"] - r["start"] for r in keep_ranges)
        log.info(f"[{tag}] Gemini decision — {len(keep_ranges)} keep range(s): {keep_ranges}")
        pct = f"{100*estimated_kept/duration:.1f}%" if duration > 0 else "N/A"
        log.info(f"[{tag}] Kept {estimated_kept:.2f}s of {duration:.2f}s ({pct} retained)")

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
            job_update(job_id, status="error", error=msg)
            return

        actual_duration = get_video_duration(output_path)
        job_update(job_id, kept_duration=actual_duration)

        full_transcript = " ".join(s["text"] for s in transcript) if transcript else "(no speech)"
        cut_ranges = keep_to_cut_ranges(keep_ranges, duration)
        total_cut = sum(e - s for s, e in cut_ranges)

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
        kept_pct = f"{100*actual_duration/duration:.1f}%" if duration > 0 else "N/A"
        cut_pct  = f"{100*total_cut/duration:.1f}%"      if duration > 0 else "N/A"
        log.info(f"[{tag}] ║ Kept        : {actual_duration:.2f}s  ({kept_pct} retained)")
        log.info(f"[{tag}] ║ Trimmed     : {total_cut:.2f}s  ({cut_pct} removed)")
        log.info(f"[{tag}] ║ Output      : {Path(output_path).name}")
        log.info(f"[{tag}] ╚══════════════════════════════════════════════╝")

        set_status(job_id, "done", "")

    except Exception as e:
        log.exception(f"[{tag}] RENDER STAGE FAILED: {e}")
        job_update(job_id, status="error", error=str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_duration(input_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "json", input_path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[-200:]}")
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"ffprobe output unparseable: {e} — stdout: {result.stdout[:200]}")


def keep_to_cut_ranges(keep_ranges: list, duration: float) -> list:
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
        shutil.move(seg_paths[0], output_path)
        return

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
    try:
        model = get_whisper_model()
    except ImportError:
        raise RuntimeError("openai-whisper not installed. Run: pip3 install openai-whisper")

    # Runs directly in a _whisper_executor thread — no inner submit needed.
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


# ── Gemini ────────────────────────────────────────────────────────────────────

# System prompt is fetched from Firestore (config/system_prompt.content)
# and cached for PROMPT_CACHE_TTL seconds. Update in Firestore → propagates automatically.


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
        system_instruction=firestore_client.get_system_prompt()
    )
    log.info(f"[Gemini] System instruction:\n{firestore_client.get_system_prompt()}")
    response = model.generate_content(user_prompt)
    log.info(f"[Gemini] Raw response:\n{response.text}")

    raw = response.text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Remove first line (```json or ```) and last line if it's closing ```
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        raw = "\n".join(lines[start:end]).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini returned invalid JSON: {e} — raw: {raw[:300]}")

    keep = data.get("keep", [])
    if not isinstance(keep, list):
        raise RuntimeError(f"Gemini JSON missing 'keep' list — got: {type(keep)}")

    keep = [
        {"start": max(0.0, float(r["start"])), "end": min(duration, float(r["end"]))}
        for r in keep
        if isinstance(r, dict) and "start" in r and "end" in r and float(r["end"]) > float(r["start"])
    ]
    return keep


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/clear")
def clear_jobs():
    if _redis is None:
        raise HTTPException(503, "Redis unavailable")
    keys = list(_redis.scan_iter("job:*"))
    count = len(keys)
    if keys:
        _redis.delete(*keys)
    log.info(f"[/clear] Cleared {count} job(s) from Redis")
    return {"cleared": count}


@app.get("/status")
def get_status(ids: str = "", session_id: str = ""):
    """
    Filter priority:
      ?ids=id1,id2        → return exactly those job IDs (most precise, use this)
      ?session_id=abc     → SMEMBERS session set + MGET — O(session size), not O(all jobs)
      (none)              → return all jobs in Redis (admin/debug only)
    """
    if ids:
        filtered = jobs_get_many([i for i in ids.split(",") if i])
        meta = {}
    elif session_id:
        # O(session size): one SMEMBERS + one MGET — no full scan
        job_ids_in_session = session_get_jobs(session_id)
        filtered = jobs_get_many(job_ids_in_session)
        meta = session_get_meta(session_id)
    else:
        filtered = jobs_get_all()
        meta = {}

    log.info(f"[/status] returning {len(filtered)} job(s) (ids={bool(ids)}, session_id={bool(session_id)})")

    # Flat dict — same shape the frontend always expected: { job_id: { ... }, ... }
    result: dict = {
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

    # When queried by session_id, inject metadata under "_session".
    # UUID job IDs never equal "_session" so there is no collision risk.
    if session_id:
        status_counts: dict = {}
        for job in filtered.values():
            s = job["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
        result["_session"] = {
            "total_submitted": meta.get("total_submitted"),  # None until has_more=False arrives
            "all_received": meta.get("all_received", False), # True after last batch
            "counts": status_counts,                         # {"done": 12, "transcribing": 3, ...}
        }

    return result


@app.post("/jobs/{job_id}/render")
def submit_timeline(job_id: str, body: dict):
    job = job_get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "awaiting_timeline":
        raise HTTPException(409, f"Job is not awaiting a timeline (current status: {job['status']})")

    keep_ranges = body.get("keep", [])
    if not isinstance(keep_ranges, list) or not all(
        isinstance(r, dict) and "start" in r and "end" in r for r in keep_ranges
    ):
        raise HTTPException(422, "Body must be {\"keep\": [{\"start\": float, \"end\": float}, ...]}")

    job_update(job_id, transcript=None)
    set_status(job_id, "rendering", "Rendering with submitted timeline…")
    log.info(f"[{job_id[:8]}] Manual render requested — {len(keep_ranges)} segment(s): {keep_ranges}")
    _render_executor.submit(_render_job, job_id, job["upload_file"], job["output_file"], keep_ranges)
    return {"status": "rendering"}


def _render_job(job_id: str, input_path: str, output_path: str, keep_ranges: list):
    try:
        render_with_ffmpeg(keep_ranges, input_path, output_path, job_id[:8])
        if not Path(output_path).exists():
            raise RuntimeError("Output file missing after render")
        actual = get_video_duration(output_path)
        job_update(job_id, kept_duration=actual)
        log.info(f"[{job_id[:8]}] Manual render done — actual duration {actual:.2f}s")
        set_status(job_id, "done", "")
    except Exception as e:
        log.exception(f"[{job_id[:8]}] Manual render failed")
        job_update(job_id, status="error", error=str(e))


@app.get("/videos/{job_id}")
def serve_video(job_id: str):
    job = job_get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
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
