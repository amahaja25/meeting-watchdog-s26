from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import subprocess, sys, json, os, urllib.request, urllib.parse, asyncio, traceback
from sean_assets.prompt import prompt as GEMINI_PROMPT

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Path to bundled jurisdiction data
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__) or ".", "data")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# ========================= YouTube endpoints ================================

@app.get("/api/transcript")
def get_transcript(video_id: str):
    """Return raw transcript JSON (segments) for a video id using existing script output.
    If transcript JSON doesn't exist yet, invoke transcript_import.py to generate it.
    """
    transcripts_json = os.path.join("transcripts", f"{video_id}.json")
    if not os.path.exists(transcripts_json):
        # run the script to create it
        cmd = [sys.executable, "transcript_import.py", video_id]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Execution failed: {e}")
        if completed.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Script error: {completed.stderr or completed.stdout}")
        if not os.path.exists(transcripts_json):
            raise HTTPException(status_code=500, detail="Transcript JSON not generated")
    try:
        with open(transcripts_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read transcript: {e}")
    return JSONResponse(content=data)

@app.get("/api/extract")
def extract_summary(video_id: str, force: bool = False):
    """Run extraction via transcript_import.py --extract and return JSON (or raw).

    Caching: If extraction file already exists and force is False, reuse it.
    Returns an additional key `cached`: true if reused.
    """
    extraction_path = os.path.join("extraction", f"{video_id}.txt")
    ran = False
    if force or not os.path.exists(extraction_path):
        cmd = [sys.executable, "transcript_import.py", "--extract", video_id]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Execution failed: {e}")
        if completed.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Script error: {completed.stderr or completed.stdout}")
        ran = True
        if not os.path.exists(extraction_path):
            raise HTTPException(status_code=500, detail="Extraction file not generated")
    try:
        with open(extraction_path, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read failure: {e}")
    try:
        parsed = json.loads(txt)
        return JSONResponse(content={"status":"ok","data":parsed, "cached": (not ran)})
    except Exception:
        return JSONResponse(content={"status":"raw","data":txt, "cached": (not ran)})

@app.get("/api/playlist")
def get_playlist(playlist_id: str, max_items: int = 50):
    """Fetch playlist metadata + items (YouTube Data API v3)."""
    api_key = os.environ.get("YT_API_KEY", "").strip()
    if not api_key:
        key_path = os.path.join("sean_assets", "yt_apikey.txt")
        if os.path.exists(key_path):
            try:
                with open(key_path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if "=" in raw:
                    raw = raw.split("=",1)[1]
                api_key = raw.strip().strip('"').strip("'")
            except Exception:
                pass
    if not api_key:
        raise HTTPException(status_code=500, detail="No YT API key (env YT_API_KEY or sean_assets/yt_apikey.txt)")

    playlist_title = None
    channel_title = None
    meta_params = {
        "part": "snippet",
        "id": playlist_id,
        "key": api_key
    }
    meta_url = "https://www.googleapis.com/youtube/v3/playlists?" + urllib.parse.urlencode(meta_params)
    try:
        with urllib.request.urlopen(meta_url, timeout=15) as resp:
            meta_data = json.loads(resp.read().decode("utf-8", errors="replace"))
        items_md = meta_data.get("items", [])
        if items_md:
            snip = items_md[0].get("snippet", {})
            playlist_title = snip.get("title")
            channel_title = snip.get("channelTitle")
    except Exception:
        pass

    items = []
    page_token = None
    fetched = 0
    while fetched < max_items:
        params = {
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": min(50, max_items - fetched),
            "key": api_key
        }
        if page_token:
            params["pageToken"] = page_token
        url = "https://www.googleapis.com/youtube/v3/playlistItems?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Playlist fetch failed: {e}")
        for it in data.get("items", []):
            snip = it.get("snippet", {})
            rid = snip.get("resourceId", {})
            vid = rid.get("videoId")
            if not vid:
                continue
            title = snip.get("title")
            pos = snip.get("position")
            thumbs = snip.get("thumbnails", {})
            thumb = (thumbs.get("medium") or thumbs.get("high") or thumbs.get("default") or {}).get("url")
            items.append({
                "video_id": vid,
                "title": title,
                "position": pos,
                "thumbnail_url": thumb
            })
        fetched = len(items)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return JSONResponse(content={
        "playlist": {"id": playlist_id, "title": playlist_title, "channel_title": channel_title},
        "items": items
    })

# ========================= Granicus endpoints ===============================

@app.get("/api/granicus/jurisdictions")
def granicus_jurisdictions():
    """Return sorted list of jurisdictions from bundled data/subdomains.json."""
    path = os.path.join(DATA_DIR, "subdomains.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="subdomains.json not found in data/")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subs = data.get("subdomains", {})
    result = sorted(
        [{"subdomain": k, "url": v.get("url", ""), "organization": v.get("organization", "")}
         for k, v in subs.items()],
        key=lambda x: x["subdomain"],
    )
    return JSONResponse(content={"jurisdictions": result, "total": len(result)})

@app.get("/api/granicus/videos")
def granicus_videos(subdomain: str):
    """Return video list for a jurisdiction from data/latest/{subdomain}.json."""
    path = os.path.join(DATA_DIR, "latest", f"{subdomain}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No video data for '{subdomain}'")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/api/granicus/transcribe")
async def granicus_transcribe(
    request: Request,
    subdomain: str,
    video_id: str,
    model: str = os.environ.get("WHISPER_MODEL", "large-v2"),
    diarize: bool = False,
):
    """SSE endpoint: transcribe a Granicus video, streaming progress events.

    Event types:
        progress  — {step, percent, message}
        complete  — {video_key}
        error     — {message}
    """
    from granicus import run_full_pipeline

    video_key = f"g_{subdomain}_{video_id}"

    # Fast-path: if transcript already cached, return immediately
    cached_json = os.path.join("transcripts", f"{video_key}.json")
    if os.path.exists(cached_json):
        async def _cached():
            yield _sse("progress", {"step": "complete", "percent": 100, "message": "Transcript already cached"})
            yield _sse("complete", {"video_key": video_key})
        return StreamingResponse(_cached(), media_type="text/event-stream")

    # Run pipeline in a thread, bridging progress to an asyncio queue
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _progress_cb(step, percent, message):
        asyncio.run_coroutine_threadsafe(
            queue.put(("progress", {"step": step, "percent": percent, "message": message})),
            loop,
        )

    async def _run_pipeline():
        try:
            result_key = await loop.run_in_executor(
                None,
                lambda: run_full_pipeline(
                    subdomain, video_id,
                    model=model, diarize=diarize,
                    progress_cb=_progress_cb,
                ),
            )
            await queue.put(("complete", {"video_key": result_key}))
        except Exception as exc:
            await queue.put(("error", {"message": str(exc)}))

    task = asyncio.ensure_future(_run_pipeline())

    async def _event_stream():
        keepalive_timeout = 120  # seconds
        while True:
            try:
                event_type, data = await asyncio.wait_for(queue.get(), timeout=keepalive_timeout)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            yield _sse(event_type, data)

            if event_type in ("complete", "error"):
                break

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# ========================= Utility endpoints ================================

@app.get("/api/prompt")
def get_prompt():
    """Return the current Gemini extraction prompt text."""
    return JSONResponse(content={"prompt": GEMINI_PROMPT})

@app.get("/api/model")
def get_model():
    """Return the model name used for extraction (env GEMINI_MODEL or default)."""
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro").strip()
    return JSONResponse(content={"model": model})
