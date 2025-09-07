# Advanced Video Processor

An advanced video processing tool that can automatically create condensed videos from transcripts with seamless AI-powered transitions.

## Features

*   **Jump-cut Mode**: Creates a fast-paced video by stitching together essential segments from a transcript.
*   **Seamless Mode**: Generates smooth AI-powered transitions between segments using Google's Generative AI and Fal AI.
*   **Manual Cut Mode**: Allows for precise video cutting based on `xxx` timestamps in a transcript.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Keys:**
    Create a `.env` file in the root directory and add your API keys:
    ```
    FAL_KEY="your_fal_api_key"
    GEMINI_API_KEY="your_google_api_key"
    ```

## Usage

The main script is `scripts/video_cutter2.py`. You can run it with different modes.

### Seamless Mode (Default)

This mode generates a condensed video with AI transitions.

```bash
python scripts/video_cutter2.py --video my_video.mp4 --transcript my_transcript.srt
```

### Jump-cut Mode

This mode creates a condensed video with simple cuts (no transitions).

```bash
python scripts/video_cutter2.py --mode jumpcut --video my_video.mp4 --transcript my_transcript.srt
```

### Manual Cut Mode

This mode processes the first `xxx` timestamp in the transcript and creates `before` and `after` video files.

```bash
python scripts/video_cutter2.py --mode manualcut --video my_video.mp4 --transcript my_transcript.srt
```