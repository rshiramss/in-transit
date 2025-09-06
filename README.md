# ElevenLabs Transcription Service

A simple Python service for transcribing videos and audio files using ElevenLabs' Speech-to-Text API.

## Features

- Transcribe video files (MP4, MOV, AVI, MKV, etc.)
- Transcribe audio files (MP3, WAV, etc.)
- Support for multiple languages
- Export transcripts in multiple formats (TXT, JSON, SRT, VTT)
- Simple and easy-to-use interface

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get your ElevenLabs API key:**
   - Visit [ElevenLabs](https://elevenlabs.io/)
   - Sign up or log in to your account
   - Go to your profile settings to get your API key

3. **Set your API key:**
   ```bash
   export ELEVENLABS_API_KEY="your_api_key_here"
   ```

## Usage

### Quick Start

Run the example script:
```bash
python example_transcription.py
```

This will give you an interactive menu to transcribe your files.

### Using the Service Directly

```python
from transcription_service import ElevenLabsTranscriptionService

# Initialize the service
service = ElevenLabsTranscriptionService()

# Transcribe a video file
result = service.transcribe_video("path/to/your/video.mp4")

# Get the transcript text
transcript = result.get('text', '')

# Save in different formats
service.save_transcript(result, "output.txt", "txt")
service.save_transcript(result, "output.json", "json")
service.save_transcript(result, "output.srt", "srt")
service.save_transcript(result, "output.vtt", "vtt")
```

### Supported File Formats

**Video:** MP4, MOV, AVI, MKV, and other major video formats
**Audio:** MP3, WAV, M4A, and other common audio formats

### Supported Languages

The service supports 99+ languages. Specify the language code when transcribing:

```python
# Transcribe in Spanish
result = service.transcribe_video("video.mp4", language="es")

# Transcribe in French
result = service.transcribe_video("video.mp4", language="fr")
```

### Output Formats

- **TXT**: Plain text transcript
- **JSON**: Full API response with metadata
- **SRT**: SubRip subtitle format
- **VTT**: WebVTT subtitle format

## API Reference

### ElevenLabsTranscriptionService

#### `__init__(api_key=None)`
Initialize the service with your API key.

#### `transcribe_video(video_path, language="en")`
Transcribe a video file.

#### `transcribe_audio(audio_path, language="en")`
Transcribe an audio file.

#### `save_transcript(transcript_data, output_path, format="txt")`
Save transcription results to a file.

## Error Handling

The service includes comprehensive error handling for:
- Missing API keys
- File not found errors
- Network request failures
- Invalid file formats

## Example Output

```
TRANSCRIPTION RESULT:
==================================================
Hello, this is a sample transcription of my video. 
The ElevenLabs API has successfully converted the 
spoken audio into text with high accuracy.
==================================================
```

## Troubleshooting

1. **"API key is required" error:**
   - Make sure you've set the `ELEVENLABS_API_KEY` environment variable
   - Verify your API key is correct

2. **"File not found" error:**
   - Check that the file path is correct
   - Make sure the file exists and is accessible

3. **"Transcription request failed" error:**
   - Check your internet connection
   - Verify your API key has sufficient credits
   - Ensure the file format is supported

## License

This project is for educational and personal use. Please respect ElevenLabs' terms of service.
