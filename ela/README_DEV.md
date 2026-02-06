# Meeting Watchdog (Dev Notes)

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here  # or put in .env (not yet auto-loaded)
uvicorn app.main:app --reload
```
Visit: http://127.0.0.1:8000/

## API
POST /api/analyze
Body: {"model": "gemini-2.5-lite" | "gemini-2.5-pro"}

Dynamic video: Enter a YouTube URL/ID in the UI and click Load, then Analyze. Backend will attempt to fetch the transcript (english preferred) using youtube-transcript-api.

Response (stub w/out API key):
```json
{
  "raw": {
    "headline": "(stub) Council marks 50th anniversary of gay rights ordinance",
    "summary": "The Minneapolis City Council commemorated the 50th anniversary ...",
    "explanation": "Chosen because...",
    "other_items": ["Recognition of ordinance authors"],
    "model_used": "gemini-2.5-lite",
    "stub": true
  }
}
```
If API key present, raw.parsed may contain structured JSON extracted from model output and raw.raw_text contains original.

### Transcript Debug
POST /api/debug/transcript
Body: {"youtube_url": "<url or id>"}
Returns diagnostic info: detected video_id, module file path, presence of get_transcript, list of transcript-related attributes, sample lines or fetch_error.

## Prompt Editing
Edit `prompt.py` variable `prompt`. Keep JSON schema instructions synced with frontend expectations (keys: headline, summary, explanation, other_items).

## Future Enhancements
- Timecode extraction and YouTube seeking
- Chunking large transcripts
- Persisting analyses
- Authentication & rate limiting

