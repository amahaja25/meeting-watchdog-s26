"""
Granicus video transcription pipeline for the Meeting Watchdog web app.

Ported from granicus-transcriber/transcribe.py, replacing all rich console
output with a progress_cb(step, percent, message) callback for SSE streaming.

Usage (standalone):
    python granicus.py houston 4650
    python granicus.py https://houston.granicus.com/videos/4650

Usage (library):
    from granicus import run_full_pipeline
    video_key = run_full_pipeline("houston", "4650", progress_cb=my_cb)
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any

# ---------------------------------------------------------------------------
# PyTorch 2.6+ compatibility — applied just before whisperx is used
# ---------------------------------------------------------------------------
_torch_patched = False

def _patch_torch_load():
    """Force weights_only=False on ALL torch.load calls.

    Must be called before importing whisperx or loading any model.
    Safe because we only load trusted HuggingFace models.
    """
    global _torch_patched
    if _torch_patched:
        return
    _torch_patched = True

    import torch
    import torch.serialization

    # Change the default on the actual function object
    for fn in (torch.load, torch.serialization.load):
        if hasattr(fn, "__kwdefaults__") and isinstance(fn.__kwdefaults__, dict):
            fn.__kwdefaults__["weights_only"] = False

    # Also wrap it to force weights_only=False even when callers pass it explicitly
    _orig = torch.serialization.load

    def _forced_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)

    torch.serialization.load = _forced_load
    torch.load = _forced_load
    print("torch: patched torch.load to force weights_only=False", file=sys.stderr)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRANICUS_URL_PATTERN = re.compile(
    r"https?://([a-zA-Z0-9-]+)\.granicus\.com/videos/(\d+)"
)
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v2")
DEFAULT_BATCH_SIZE = 16
TRANSCRIPTS_DIR = Path("transcripts")
AUDIO_CACHE_DIR = TRANSCRIPTS_DIR / "audio_cache"

HF_TOKEN_PATHS = [
    Path.home() / ".huggingface" / "token",
    Path.home() / ".cache" / "huggingface" / "token",
    Path("hf_token.txt"),
]

# Type alias for the progress callback
ProgressCB = Callable[[str, Optional[float], str], None]


def _noop_cb(step: str, percent: Optional[float], message: str) -> None:
    """Default silent callback."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_granicus_url(url: str) -> tuple[str, str]:
    """Parse a Granicus URL or ``subdomain:video_id`` string.

    Returns (subdomain, video_id) as strings.
    """
    url = url.strip()

    # Try URL pattern
    m = GRANICUS_URL_PATTERN.match(url)
    if m:
        return m.group(1), m.group(2)

    # Try subdomain:video_id shorthand
    if ":" in url and not url.startswith("http"):
        parts = url.split(":", 1)
        if parts[0] and parts[1].isdigit():
            return parts[0], parts[1]

    raise ValueError(f"Could not parse Granicus input: {url}")


def detect_device() -> tuple[str, str]:
    """Auto-detect the best available compute device and dtype."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # CTranslate2 (used by WhisperX) does not support MPS yet
            return "cpu", "int8"
        return "cpu", "int8"
    except Exception:
        return "cpu", "int8"


def get_hf_token() -> Optional[str]:
    """Resolve HuggingFace token from env vars or well-known file locations."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    for p in HF_TOKEN_PATHS:
        if p.exists():
            try:
                tok = p.read_text().strip()
                if tok:
                    return tok
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Video stream discovery
# ---------------------------------------------------------------------------

def get_video_stream_url(subdomain: str, video_id: str) -> Optional[str]:
    """Scrape the Granicus player page for the underlying video stream URL."""
    import urllib.request

    player_url = f"https://{subdomain}.granicus.com/videos/{video_id}/player"
    try:
        req = urllib.request.Request(
            player_url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        patterns = [
            r'"(https://[^"]+\.m3u8[^"]*)"',
            r"'(https://[^']+\.m3u8[^']*)'",
            r'"(https://[^"]+\.mp4[^"]*)"',
            r"'(https://[^']+\.mp4[^']*)'",
            r'src="(https://[^"]+(?:\.m3u8|\.mp4)[^"]*)"',
        ]
        for pat in patterns:
            m = re.search(pat, html)
            if m:
                return m.group(1)

        m = re.search(r'data-(?:video-)?(?:src|url)="([^"]+)"', html)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def download_audio(
    subdomain: str,
    video_id: str,
    output_dir: Path,
    progress_cb: ProgressCB = _noop_cb,
) -> Path:
    """Download audio from Granicus video.

    Strategy:
    1. Parse player HTML for stream URL, download with ffmpeg.
    2. Fall back to yt-dlp if (1) fails.

    Returns path to the downloaded audio file.
    """
    video_url = f"https://{subdomain}.granicus.com/videos/{video_id}"
    player_url = f"https://{subdomain}.granicus.com/videos/{video_id}/player"
    audio_path = output_dir / f"{subdomain}_{video_id}.mp3"

    # Check for existing cached audio
    for ext in (".mp3", ".m4a", ".wav", ".webm", ".opus"):
        existing = output_dir / f"{subdomain}_{video_id}{ext}"
        if existing.exists() and existing.stat().st_size > 10_000:
            progress_cb("download", 100, f"Audio cached ({existing.stat().st_size / 1024 / 1024:.1f} MB)")
            return existing

    progress_cb("download", 0, f"Downloading audio from {video_url}")

    # --- Strategy 1: ffmpeg from stream URL ---
    stream_url = get_video_stream_url(subdomain, video_id)
    if stream_url:
        progress_cb("download", 5, "Found stream URL, downloading via ffmpeg")
        ok = _download_with_ffmpeg(stream_url, audio_path, progress_cb)
        if ok:
            progress_cb("download", 100, f"Audio downloaded ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return audio_path
        progress_cb("download", 10, "ffmpeg failed, trying yt-dlp fallback")
    else:
        progress_cb("download", 5, "No stream URL found, trying yt-dlp")

    # --- Strategy 2: yt-dlp fallback ---
    _download_with_ytdlp(player_url, audio_path, progress_cb)

    # yt-dlp may produce a different extension
    if audio_path.exists():
        return audio_path
    for ext in (".mp3", ".m4a", ".wav", ".webm"):
        alt = audio_path.with_suffix(ext)
        if alt.exists():
            return alt

    raise RuntimeError(
        f"Could not download audio from {video_url}. "
        "Make sure ffmpeg and yt-dlp are installed."
    )


def _ffprobe_duration(stream_url: str) -> Optional[float]:
    """Use ffprobe to get stream duration in seconds. Returns None on failure."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            stream_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _download_with_ffmpeg(stream_url: str, audio_path: Path, progress_cb: ProgressCB) -> bool:
    """Download audio from a stream URL using ffmpeg. Returns True on success."""
    try:
        # First, probe duration so we can show meaningful progress
        duration = _ffprobe_duration(stream_url)

        cmd = [
            "ffmpeg", "-i", stream_url,
            "-vn", "-acodec", "libmp3lame", "-ab", "128k",
            "-y",
            "-progress", "pipe:1",
            str(audio_path),
        ]
        # Merge stderr into stdout so we can read everything in one stream
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        last_pct = 5
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break

            # ffmpeg progress output: out_time_ms or out_time_us depending on version
            if "out_time_us=" in line:
                try:
                    time_us = int(line.split("=")[1].strip())
                    time_sec = time_us / 1_000_000
                    if duration and duration > 0:
                        pct = min(95, max(5, time_sec / duration * 100))
                        if pct > last_pct:
                            last_pct = pct
                            progress_cb("download", pct, f"Downloading via ffmpeg ({time_sec:.0f}s / {duration:.0f}s)")
                    else:
                        progress_cb("download", None, f"Downloading via ffmpeg ({time_sec:.0f}s elapsed)")
                except Exception:
                    pass
            elif "out_time_ms=" in line:
                try:
                    time_ms = int(line.split("=")[1].strip())
                    time_sec = time_ms / 1_000_000
                    if duration and duration > 0:
                        pct = min(95, max(5, time_sec / duration * 100))
                        if pct > last_pct:
                            last_pct = pct
                            progress_cb("download", pct, f"Downloading via ffmpeg ({time_sec:.0f}s / {duration:.0f}s)")
                    else:
                        progress_cb("download", None, f"Downloading via ffmpeg ({time_sec:.0f}s elapsed)")
                except Exception:
                    pass
            # Also try to grab Duration from stderr (now merged) as fallback
            elif duration is None and "Duration:" in line:
                try:
                    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)", line)
                    if m:
                        h, mn, s = map(int, m.groups())
                        duration = h * 3600 + mn * 60 + s
                except Exception:
                    pass

        proc.wait()
        if audio_path.exists() and audio_path.stat().st_size > 10_000:
            return True
    except Exception:
        pass
    return False


def _download_with_ytdlp(player_url: str, audio_path: Path, progress_cb: ProgressCB) -> None:
    """Download audio using yt-dlp (blocking, with timeout)."""
    import threading
    import time

    cmd = [
        "yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "0",
        "-o", str(audio_path.with_suffix("")),
        "--no-playlist", "--newline", "--progress",
        player_url,
    ]

    result_code: list[Optional[int]] = [None]
    error_msg: list[Optional[str]] = [None]

    def _run():
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                line = line.strip()
                m = re.search(r"(\d+\.?\d*)%", line)
                if m:
                    pct = float(m.group(1))
                    progress_cb("download", pct, "Downloading via yt-dlp")
            p.wait()
            result_code[0] = p.returncode
        except Exception as exc:
            error_msg[0] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    timeout = 180
    started = time.time()
    while t.is_alive():
        if time.time() - started > timeout:
            raise TimeoutError("yt-dlp timed out")
        time.sleep(1)

    t.join(timeout=5)
    if error_msg[0]:
        raise RuntimeError(error_msg[0])
    if result_code[0] and result_code[0] != 0:
        raise subprocess.CalledProcessError(result_code[0], "yt-dlp")


# ---------------------------------------------------------------------------
# WhisperX transcription
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    language: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    diarize: bool = False,
    hf_token: Optional[str] = None,
    progress_cb: ProgressCB = _noop_cb,
) -> dict:
    """Transcribe an audio file using WhisperX, returning the result dict."""
    _patch_torch_load()
    import torch
    import whisperx

    if device is None or compute_type is None:
        det_dev, det_ct = detect_device()
        device = device or det_dev
        compute_type = compute_type or det_ct

    if device == "cpu":
        batch_size = min(batch_size, 4)

    progress_cb("transcribe", 0, f"Loading WhisperX model ({model_name}) on {device}")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    progress_cb("transcribe", 10, "Loading audio")
    audio = whisperx.load_audio(str(audio_path))
    audio_duration = len(audio) / 16000

    progress_cb("transcribe", 15, f"Transcribing ({audio_duration / 60:.0f} min of audio)")
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    detected_lang = result.get("language", language or "en")
    progress_cb("transcribe", 70, f"Transcription done (lang={detected_lang})")

    # --- Alignment ---
    progress_cb("align", 0, "Aligning transcript for word-level timestamps")
    try:
        model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )
        progress_cb("align", 100, "Alignment complete")
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        progress_cb("align", 100, f"Alignment skipped: {exc}")

    # Clean up main model
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Diarization (optional) ---
    if diarize:
        if not hf_token:
            progress_cb("diarize", 100, "Diarization skipped (no HF token)")
        else:
            progress_cb("diarize", 0, "Running speaker diarization")
            try:
                from whisperx.diarize import DiarizationPipeline

                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)

                speakers = {seg.get("speaker") for seg in result.get("segments", []) if "speaker" in seg}
                progress_cb("diarize", 100, f"Diarization complete ({len(speakers)} speakers)")

                del diarize_model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                progress_cb("diarize", 100, f"Diarization failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Segment normalisation & saving
# ---------------------------------------------------------------------------

def normalize_segments(whisperx_segments: list[dict]) -> list[dict]:
    """Convert WhisperX segments ``{text, start, end}`` to the format used by
    the YouTube pipeline: ``{text, start, duration}``.
    """
    out = []
    for seg in whisperx_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        out.append({
            "text": text,
            "start": round(start, 3),
            "duration": round(max(end - start, 0), 3),
        })
    return out


def save_granicus_transcript(
    subdomain: str,
    video_id: str,
    segments: list[dict],
) -> tuple[str, str, str]:
    """Save transcript in the same three-file format as the YouTube pipeline.

    Returns ``(json_path, txt_path, timed_path)`` strings.
    """
    key = f"g_{subdomain}_{video_id}"
    os.makedirs("transcripts", exist_ok=True)
    base = os.path.join("transcripts", key)
    json_path = base + ".json"
    txt_path = base + ".txt"
    timed_path = base + "_timed.txt"

    # JSON segments (same format as YouTube)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # Plain text
    parts = [s["text"].replace("\n", " ").strip() for s in segments if s.get("text")]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(" ".join(parts) + "\n")

    # Timed text ([seconds] text)
    with open(timed_path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.get("text") or "").strip().replace("\n", " ")
            if not text:
                continue
            ts = int(seg.get("start", 0))
            f.write(f"[{ts}] {text}\n")

    return json_path, txt_path, timed_path


# ---------------------------------------------------------------------------
# Full orchestrator
# ---------------------------------------------------------------------------

def run_full_pipeline(
    subdomain: str,
    video_id: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    language: Optional[str] = None,
    diarize: bool = False,
    progress_cb: ProgressCB = _noop_cb,
) -> str:
    """Run the full download-transcribe-save pipeline.

    Returns the ``video_key`` string (``g_{subdomain}_{video_id}``) which can
    then be passed to ``/api/extract`` unchanged.
    """
    video_key = f"g_{subdomain}_{video_id}"

    # Check for cached transcript
    cached_json = os.path.join("transcripts", f"{video_key}.json")
    if os.path.exists(cached_json):
        progress_cb("complete", 100, "Transcript already cached")
        return video_key

    # Ensure audio cache dir exists
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download
    progress_cb("download", 0, "Starting audio download")
    audio_path = download_audio(subdomain, video_id, AUDIO_CACHE_DIR, progress_cb)

    # 2. Transcribe
    progress_cb("transcribe", 0, "Starting transcription")
    hf_token = get_hf_token() if diarize else None
    result = transcribe_audio(
        audio_path,
        model_name=model,
        batch_size=batch_size,
        language=language,
        diarize=diarize,
        hf_token=hf_token,
        progress_cb=progress_cb,
    )

    # 3. Normalize & save
    progress_cb("saving", 50, "Saving transcript files")
    raw_segments = result.get("segments", [])
    segments = normalize_segments(raw_segments)
    save_granicus_transcript(subdomain, video_id, segments)
    progress_cb("saving", 100, f"Saved {len(segments)} segments")

    return video_key


# ---------------------------------------------------------------------------
# CLI entry-point (for testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _cli_cb(step, pct, msg):
        pct_str = f"{pct:.0f}%" if pct is not None else "---"
        print(f"  [{step}] {pct_str} | {msg}")

    args = sys.argv[1:]
    if not args:
        print("Usage: python granicus.py <url_or_subdomain:video_id>")
        sys.exit(1)

    raw = args[0]
    if ":" in raw and not raw.startswith("http"):
        sd, vid = raw.split(":", 1)
    elif raw.startswith("http"):
        sd, vid = parse_granicus_url(raw)
    else:
        if len(args) < 2:
            print("Usage: python granicus.py <subdomain> <video_id>")
            sys.exit(1)
        sd, vid = args[0], args[1]

    model = args[2] if len(args) > 2 else DEFAULT_MODEL
    key = run_full_pipeline(sd, vid, model=model, progress_cb=_cli_cb)
    print(f"\nDone! video_key = {key}")
