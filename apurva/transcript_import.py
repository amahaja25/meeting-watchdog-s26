from youtube_transcript_api import YouTubeTranscriptApi
import sys
import os
import json
import http.client
import urllib.parse
from typing import List

DEFAULT_VIDEO_ID = "bRgnmYwcOGQ"

def save_transcript(video_id: str):
    api = YouTubeTranscriptApi()
    segments = api.fetch(video_id)
    # Normalize segments into plain dicts (in case library returns objects)
    norm_segments = []
    for s in segments:
        if isinstance(s, dict):
            text_val = s.get("text", "")
            start_val = s.get("start", 0.0)
            dur_val = s.get("duration", 0.0)
        else:  # object with attributes
            text_val = getattr(s, "text", "")
            start_val = getattr(s, "start", 0.0)
            dur_val = getattr(s, "duration", 0.0)
        norm_segments.append({
            "text": (text_val or ""),
            "start": float(start_val) if isinstance(start_val, (int, float)) else 0.0,
            "duration": float(dur_val) if isinstance(dur_val, (int, float)) else 0.0,
        })
    parts = []
    for s in norm_segments:
        t = s.get("text", "")
        t = (t or "").strip().replace("\n", " ")
        if t:
            parts.append(t)
    text = " ".join(parts)
    os.makedirs("transcripts", exist_ok=True)
    base = os.path.join("transcripts", f"{video_id}")
    txt_path = base + ".txt"
    json_path = base + ".json"
    timed_path = base + "_timed.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(norm_segments, f, ensure_ascii=False, indent=2)
    # Timed lines: [floor(start)] text
    with open(timed_path, "w", encoding="utf-8") as f:
        for seg in norm_segments:
            raw_text = (seg.get("text", "") or "").strip().replace("\n", " ")
            if not raw_text:
                continue
            ts = int(seg.get("start", 0.0) or 0)
            f.write(f"[{ts}] {raw_text}\n")
    print(f"Saved transcript text -> {txt_path} (chars: {len(text)})")
    print(f"Saved transcript JSON -> {json_path} (segments: {len(norm_segments)})")
    print(f"Saved timed transcript -> {timed_path}")

def _call_gemini_pro(api_key: str, model: str, prompt_text: str, transcript_text: str) -> str:
    # Minimal REST call to Gemini (generative language) style endpoint.
    # We avoid external deps; adjust endpoint as needed for your key provider.
    host = "generativelanguage.googleapis.com"
    path = f"/v1beta/models/{model}:generateContent?key={urllib.parse.quote(api_key)}"
    body = {
        "contents": [
            {"parts": [{"text": prompt_text + "\n\nTRANSCRIPT:\n" + transcript_text[:180000]}]}
        ]
    }
    payload = json.dumps(body)
    conn = http.client.HTTPSConnection(host, timeout=60)
    conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read().decode("utf-8", errors="replace")
    if resp.status != 200:
        return f"ERROR: Gemini API {resp.status} {resp.reason}\n{data}"
    try:
        parsed = json.loads(data)
        # Extract first candidate text
        candidates = parsed.get("candidates") or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts and isinstance(parts, list):
                return parts[0].get("text", data)
        return data
    except Exception:
        return data

def main():
    args = sys.argv[1:]
    extract = False
    vid = DEFAULT_VIDEO_ID
    for a in list(args):
        if a == "--extract":
            extract = True
            args.remove(a)
    if args:
        vid = args[0]
    # Skip YouTube fetch if transcript JSON already exists (e.g. Granicus pre-generated)
    if not os.path.exists(os.path.join("transcripts", f"{vid}.json")):
        save_transcript(vid)
    if not extract:
        return
    # Build structured transcript string with timecodes (reuse timed file logic)
    base = os.path.join("transcripts", vid)
    json_path = base + ".json"
    if not os.path.exists(json_path):
        print("No JSON transcript to extract from", file=sys.stderr)
        return
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    lines: List[str] = []
    for seg in segments:
        t = int(seg.get("start", 0))
        text_seg = (seg.get("text", "") or "").strip()
        if not text_seg:
            continue
        lines.append(f"[{t}] {text_seg}")
    structured_text = "\n".join(lines)
    # Load prompt
    try:
        from sean_assets.prompt import prompt as base_prompt
    except Exception:
        base_prompt = "You are an analyst. Summarize the transcript."  # fallback
    # Obtain API key (env first, then file)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        key_path = os.path.join("sean_assets", "apikey.txt")
        if not os.path.exists(key_path):
            print("No API key found (env GEMINI_API_KEY or sean_assets/apikey.txt); skipping Gemini extraction.")
            return
        with open(key_path, "r", encoding="utf-8") as f:
            raw_key = f.read().strip()
        # Accept formats like: gemini_api_key="KEY" or just KEY
        if "=" in raw_key:
            raw_key = raw_key.split("=", 1)[1]
        api_key = raw_key.strip().strip('"').strip("'")
    if not api_key:
        print("API key empty after parsing; skipping Gemini extraction.")
        return
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    print(f"Calling Gemini model {model} ...")
    result_text = _call_gemini_pro(api_key, model, base_prompt, structured_text)
    os.makedirs("extraction", exist_ok=True)
    out_path = os.path.join("extraction", f"{vid}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"Saved extraction -> {out_path}")

if __name__ == "__main__":
    main()