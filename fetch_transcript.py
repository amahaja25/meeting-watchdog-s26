#!/usr/bin/env python3
"""
Fetch a YouTube video transcript (English preferred) and save to transcripts/v_<video_id>.txt

Usage:
    python fetch_transcript.py VIDEO_ID_OR_URL

Exit codes:
 1 usage error
 2 invalid video id
 3 transcripts disabled
 4 no transcript found
 5 could not retrieve transcript
 6 api exception
 7 unexpected error
 8 empty transcript
"""
from __future__ import annotations
import sys
import re
import os
from typing import List, Optional
from youtube_transcript_api import (  # type: ignore
    YouTubeTranscriptApi,  # type: ignore
    TranscriptsDisabled,  # type: ignore
    NoTranscriptFound,  # type: ignore
    CouldNotRetrieveTranscript,  # type: ignore
    YouTubeTranscriptApiException,  # type: ignore
)

YOUTUBE_ID_RE = re.compile(r'(?:v=|youtu\.be/|shorts/|embed/|^)([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)')
PREFERRED_LANGS = ["en", "en-US", "en-GB"]

def extract_video_id(raw: str) -> str:
    raw = raw.strip()
    m = YOUTUBE_ID_RE.search(raw)
    if not m:
        raise ValueError(f"Could not parse YouTube video ID from input: {raw}")
    return m.group(1)

def fetch_segments(video_id: str, debug: bool = False) -> List[dict]:
    """Return list of segment dicts for the video.

    Strategy:
      1. Direct attempt for preferred English variants.
      2. Fallback: enumerate transcripts, pick natural English first, else
         translate a translatable transcript to English.
    Raises:
      TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript,
      YouTubeTranscriptApiException, RuntimeError (if unexpected state)
    """
    # 1. Direct attempt (fast path) if get_transcript exists.
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            if debug: print("[debug] trying get_transcript with preferred languages", file=sys.stderr)
            return YouTubeTranscriptApi.get_transcript(video_id, languages=PREFERRED_LANGS)  # type: ignore[attr-defined]
        except TypeError:
            # Older version may not accept languages kw.
            try:
                if debug: print("[debug] retry get_transcript without languages param", file=sys.stderr)
                return YouTubeTranscriptApi.get_transcript(video_id)  # type: ignore[attr-defined]
            except Exception as e:
                if debug: print(f"[debug] get_transcript(no languages) failed: {e}", file=sys.stderr)
        except TranscriptsDisabled:
            raise
        except NoTranscriptFound as e:
            if debug: print(f"[debug] direct get_transcript reported NoTranscriptFound: {e}", file=sys.stderr)
        except Exception as e:
            if debug: print(f"[debug] direct get_transcript failed: {e}", file=sys.stderr)

    # 2. Enumerate available transcripts (may not exist in older lib versions)
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        try:
            if debug: print("[debug] listing transcripts", file=sys.stderr)
            t_list = YouTubeTranscriptApi.list_transcripts(video_id)  # type: ignore[attr-defined]
        except Exception as e:
            if debug: print(f"[debug] list_transcripts failed: {e}", file=sys.stderr)
            t_list = []
    else:
        t_list = []
        if debug: print("[debug] list_transcripts unavailable; probing languages individually", file=sys.stderr)
        candidate_langs = PREFERRED_LANGS + ["en", "en-CA", "en-AU"]
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            for lang in candidate_langs:
                try:
                    if debug: print(f"[debug] probe language {lang}", file=sys.stderr)
                    return YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])  # type: ignore[attr-defined]
                except NoTranscriptFound:
                    continue
                except Exception as e:
                    if debug: print(f"[debug] probe {lang} failed: {e}", file=sys.stderr)
                    continue
        raise RuntimeError("Could not obtain transcript via direct or language probe (no listing API)")

    # Prefer an existing English-language transcript (natural, not translated)
    for tr in t_list:
        try:
            if getattr(tr, "language_code", "").startswith("en") and not getattr(tr, "is_generated", False):
                if debug: print(f"[debug] using natural English transcript {getattr(tr,'language_code','')} (non-generated)", file=sys.stderr)
                return tr.fetch()
        except Exception as e:
            if debug: print(f"[debug] natural English candidate failed fetch: {e}", file=sys.stderr)
            continue

    # Accept generated English if present
    for tr in t_list:
        try:
            if getattr(tr, "language_code", "").startswith("en"):
                if debug: print(f"[debug] using generated English transcript {getattr(tr,'language_code','')}", file=sys.stderr)
                return tr.fetch()
        except Exception as e:
            if debug: print(f"[debug] generated English candidate failed fetch: {e}", file=sys.stderr)
            continue

    # Else attempt translation from a translatable transcript
    for tr in t_list:
        if getattr(tr, "is_translatable", False):
            try:
                if debug: print(f"[debug] attempting translate from {getattr(tr,'language_code','?')} -> en", file=sys.stderr)
                return tr.translate("en").fetch()
            except Exception as e:
                if debug: print(f"[debug] translate attempt failed: {e}", file=sys.stderr)
                continue

    raise RuntimeError("No English (direct, generated, or translatable) transcript available")

def build_text(segments: List[dict]) -> str:
    lines = []
    for seg in segments:
        text = seg.get("text", "").replace("\n", " ").strip()
        if text:
            lines.append(text)
    return " ".join(lines)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python fetch_transcript.py [--debug] VIDEO_ID_OR_URL", file=sys.stderr)
        sys.exit(1)
    debug = False
    args = [a for a in sys.argv[1:] if a]
    if args[0] == "--debug":
        if len(args) != 2:
            print("Usage: python fetch_transcript.py [--debug] VIDEO_ID_OR_URL", file=sys.stderr)
            sys.exit(1)
        debug = True
        raw_input = args[1]
    else:
        raw_input = args[0]
    try:
        video_id = extract_video_id(raw_input)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        segments = fetch_segments(video_id, debug=debug)
        text = build_text(segments)
    except TranscriptsDisabled:
        print("ERROR: Transcripts are disabled for this video.", file=sys.stderr)
        sys.exit(3)
    except NoTranscriptFound as e:
        print(f"ERROR: No transcript found: {e}", file=sys.stderr)
        sys.exit(4)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(4)
    except CouldNotRetrieveTranscript as e:
        print(f"ERROR: Could not retrieve transcript: {e}", file=sys.stderr)
        sys.exit(5)
    except YouTubeTranscriptApiException as e:
        print(f"ERROR: API exception: {e}", file=sys.stderr)
        sys.exit(6)
    except Exception as e:
        print(f"ERROR: Unexpected failure: {e.__class__.__name__}: {e}", file=sys.stderr)
        sys.exit(7)

    if not text.strip():
        print("ERROR: Retrieved transcript is empty.", file=sys.stderr)
        sys.exit(8)

    out_dir = "transcripts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"v_{video_id}.txt")
    header = [
        f"# video_id: {video_id}",
        f"# segments: {len(segments)}",
        "# note: concatenated plain text transcript\n",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
        f.write(text + "\n")

    print(f"Saved transcript -> {out_path} (chars: {len(text)})")
    if debug:
        print("[debug] done", file=sys.stderr)

if __name__ == "__main__":  # pragma: no cover
    main()
