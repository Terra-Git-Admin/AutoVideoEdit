"""
Firestore + GCS system prompt loader.

Flow:
    1. Firestore  — system-prompts/EUrmPS1aV62HFVB13tle
                    → story-board.smart_video_trimmer
                    → GCS object path  (e.g. "system-prompts/story-board/SP_SMART_VIDEO_TRIMMER.txt")

    2. GCS bucket comics-master-store
                    → download that object → prompt text string

    3. Cache the text for PROMPT_CACHE_TTL seconds.
       Update the .txt file in GCS → propagates to all jobs within cache TTL, no restart needed.
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("autovideoedit")

PROMPT_CACHE_TTL = 1200  # seconds (20 min) — refresh from GCS every 20 min

_db  = None
_gcs = None
_prompt_cache: dict = {"text": None, "fetched_at": 0.0}


def init_firestore():
    """
    Call once from the FastAPI lifespan. Initialises both Firestore and GCS clients.
    Resolves a relative GOOGLE_APPLICATION_CREDENTIALS path to absolute automatically.
    """
    global _db, _gcs

    # Resolve relative credential path to absolute so it works from any cwd
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred_path and not Path(cred_path).is_absolute():
        resolved = Path(__file__).parent / cred_path
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)
        log.info(f"[Firestore] Credentials resolved → {resolved}")

    project_id = os.environ.get("FIRESTORE_PROJECT_ID", "")
    if not project_id:
        raise RuntimeError("FIRESTORE_PROJECT_ID not set in environment")

    database_id = os.environ.get("FIRESTORE_DATABASE_ID", "(default)")
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    if not bucket_name:
        raise RuntimeError("GCS_BUCKET_NAME not set in environment")

    from google.cloud import firestore, storage

    _db  = firestore.Client(project=project_id, database=database_id)
    _gcs = storage.Client(project=project_id)

    log.info(f"[Firestore] Connected — project={project_id}  database={database_id}")
    log.info(f"[GCS]       Connected — bucket={bucket_name}")


def get_system_prompt() -> str:
    """
    Returns the system prompt text, serving from cache when fresh.
    On cache miss / expiry: reads the GCS path from Firestore, downloads the file,
    caches the result, and returns it.
    """
    now = time.time()
    if _prompt_cache["text"] and now - _prompt_cache["fetched_at"] < PROMPT_CACHE_TTL:
        return _prompt_cache["text"]

    if _db is None or _gcs is None:
        raise RuntimeError("Firestore/GCS not initialised — call init_firestore() first")

    # Step 1: read GCS path from Firestore
    doc = _db.collection("system-prompts").document("EUrmPS1aV62HFVB13tle").get()
    if not doc.exists:
        raise RuntimeError("Firestore document system-prompts/EUrmPS1aV62HFVB13tle not found")

    gcs_path = doc.to_dict().get("story-board", {}).get("smart_video_trimmer", "")
    if not gcs_path:
        raise RuntimeError("Firestore story-board.smart_video_trimmer is empty or missing")

    log.info(f"[GCS] Downloading prompt from gs://{os.environ['GCS_BUCKET_NAME']}/{gcs_path}")

    # Step 2: download the .txt file from GCS
    bucket = _gcs.bucket(os.environ["GCS_BUCKET_NAME"])
    blob   = bucket.blob(gcs_path)
    if not blob.exists():
        raise RuntimeError(f"GCS object not found: gs://{os.environ['GCS_BUCKET_NAME']}/{gcs_path}")

    text = blob.download_as_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError(f"GCS prompt file is empty: {gcs_path}")

    # Step 3: cache and return
    _prompt_cache["text"]       = text
    _prompt_cache["fetched_at"] = now
    log.info(f"[GCS] System prompt cached ({len(text)} chars, path={gcs_path})")
    return text
