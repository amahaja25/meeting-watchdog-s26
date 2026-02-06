# Meeting Watchdog — Lizzy's Copy

A web app that fetches YouTube meeting transcripts and analyzes them using Google's Gemini AI.

## How to Run

### Step 1: Open your terminal

On Mac, open the **Terminal** app (search for "Terminal" in Spotlight).

**Important:** Make sure you `cd` into **your** folder, not the top-level project:

From the repo root:

```
cd lizzy
```

### Step 2: Create a virtual environment

This keeps the app's dependencies separate from the rest of your system. You only need to do this once.

```
python3 -m venv .venv
```

### Step 3: Activate the virtual environment

You need to do this every time you open a new terminal to run the app.

```
source .venv/bin/activate
```

You'll know it worked if you see `(.venv)` at the beginning of your terminal line.

### Step 4: Install dependencies

You only need to do this once (or again if `requirements.txt` changes).

```
pip install -r requirements.txt
```

### Step 5: Set up API keys

The app needs API keys to use its full features. Run these in your terminal before starting the app:

```
export GEMINI_API_KEY=your_gemini_key_here
export YT_API_KEY=your_youtube_api_key_here
```

Replace `your_gemini_key_here` and `your_youtube_api_key_here` with your actual keys.

### Step 6: Start the app

```
uvicorn app:app --reload
```

### Step 7: Open the app

Go to **http://127.0.0.1:8000** in your web browser.

## Granicus Support

The app can also analyze municipal meeting videos hosted on Granicus. Toggle between YouTube and Granicus mode using the buttons in the header.

### Additional requirements

Granicus videos don't have transcripts, so the app transcribes them locally using [WhisperX](https://github.com/m-bain/whisperX). Most dependencies are covered by `pip install -r requirements.txt`, but **ffmpeg** is a system-level tool that must be installed separately:

```
# If you have Homebrew:
brew install ffmpeg

# If you don't have Homebrew, install it first:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg
```

### WhisperX model selection

By default the app uses the `large-v2` model, which is the most accurate but slow on CPU (~30-60 min for a 2-hour meeting). Set the `WHISPER_MODEL` environment variable to use a faster model:

```
WHISPER_MODEL=base uvicorn app:app --reload
```

Available models (fastest to most accurate):
| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `tiny` | Fastest | Low | Good for quick testing |
| `base` | Fast | Moderate | Reasonable quality, ~10x faster than large |
| `small` | Medium | Good | Good balance |
| `medium` | Slow | Very good | |
| `large-v2` | Slowest | Best | Default; best for production use |
| `large-v3` | Slowest | Best | Alternative to large-v2 |

### Optional: Speaker diarization

To identify individual speakers, set a HuggingFace token:

```
export HF_TOKEN=your_huggingface_token_here
```

### Processing time expectations

- **Download**: 1-3 minutes depending on video length
- **Transcription**: ~10-60 minutes depending on video length and model size
- **Gemini extraction**: ~10-30 seconds

## Stopping the App

Press `Ctrl + C` in the terminal to stop the server.
