import os
import requests
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dotenv import load_dotenv
from text_process import TranscriptProcessor

# Load environment variables from .env file
load_dotenv()


class ElevenLabsTranscriptionService:
    """
    A simple service to transcribe videos using ElevenLabs' Speech-to-Text API.
    """
    
    def __init__(self, api_key: Optional[str] = None, enable_condensing: bool = True):
        """
        Initialize the transcription service.
        
        Args:
            api_key: Your ElevenLabs API key. If not provided, will look for ELEVENLABS_API_KEY env var.
            enable_condensing: Whether to automatically condense transcripts to 25% using AI.
        """
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Set ELEVENLABS_API_KEY in .env file, environment variable, or pass api_key parameter.")
        
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        self.enable_condensing = enable_condensing
        if enable_condensing:
            try:
                self.text_processor = TranscriptProcessor()
            except ValueError as e:
                print(f"Warning: Could not initialize text processor: {e}")
                print("Condensing will be disabled.")
                self.enable_condensing = False
    
    def transcribe_video(self, video_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe a video file using ElevenLabs' Speech-to-Text API.
        
        Args:
            video_path: Path to the video file to transcribe
            language: Language code for transcription (default: "en")
            
        Returns:
            Dictionary containing transcription results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Prepare the request
        url = f"{self.base_url}/speech-to-text"
        
        # Read the video file
        with open(video_path, 'rb') as video_file:
            files = {
                'file': (os.path.basename(video_path), video_file, 'video/mp4')
            }
            
            # Remove Content-Type from headers for multipart form data
            headers = {k: v for k, v in self.headers.items() if k != "Content-Type"}
            
            data = {
                'model_id': 'scribe_v1'
            }
            
            try:
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Transcription request failed: {str(e)}")
    
    def transcribe_audio(self, audio_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe an audio file using ElevenLabs' Speech-to-Text API.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Language code for transcription (default: "en")
            
        Returns:
            Dictionary containing transcription results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Prepare the request
        url = f"{self.base_url}/speech-to-text"
        
        # Read the audio file
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': (os.path.basename(audio_path), audio_file, 'audio/mpeg')
            }
            
            # Remove Content-Type from headers for multipart form data
            headers = {k: v for k, v in self.headers.items() if k != "Content-Type"}
            
            data = {
                'model_id': 'scribe_v1'
            }
            
            try:
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Transcription request failed: {str(e)}")
    
    def save_transcript(self, transcript_data: Dict[str, Any], output_path: str) -> str:
        """
        Save transcription results to an SRT file and optionally create condensed version.
        
        Args:
            transcript_data: Transcription results from transcribe_video or transcribe_audio
            output_path: Path where to save the SRT transcript
            
        Returns:
            Path to the condensed SRT file if condensing is enabled, otherwise the original output_path
        """
        # Save original SRT
        self._save_srt(transcript_data, output_path)
        print(f"Original transcript saved: {output_path}")
        
        # Create condensed version if enabled
        if self.enable_condensing:
            try:
                condensed_path = self.text_processor.process_srt_file(output_path)
                print(f"Condensed transcript saved: {condensed_path}")
                return condensed_path
            except Exception as e:
                print(f"Error creating condensed transcript: {e}")
                print("Returning original transcript path.")
                return output_path
        
        return output_path
    
    def _save_srt(self, transcript_data: Dict[str, Any], output_path: str) -> None:
        """Save transcript in SRT subtitle format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            words = transcript_data.get('words', [])
            if not words:
                # Fallback for simple text
                text = transcript_data.get('text', '')
                f.write(f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n\n")
                return
            
            # Group words into subtitle chunks with sentence boundary awareness
            target_chunk_size = 18  # Increased for better coherence
            max_chunk_duration = 8.0  # Maximum 8 seconds per chunk
            subtitle_num = 1
            
            i = 0
            while i < len(words):
                chunk = []
                chunk_start_time = words[i].get('start', 0)
                
                # Build chunk respecting sentence boundaries and duration
                while (i < len(words) and 
                       len(chunk) < target_chunk_size and 
                       (words[i].get('end', chunk_start_time) - chunk_start_time) < max_chunk_duration):
                    
                    word_data = words[i]
                    chunk.append(word_data)
                    
                    # Check if this word ends a sentence (contains sentence-ending punctuation)
                    word_text = word_data.get('text', '')
                    if (len(chunk) >= target_chunk_size * 0.7 and  # At least 70% of target size
                        any(punct in word_text for punct in ['.', '!', '?', ':'])):
                        i += 1
                        break
                    
                    i += 1
                
                if not chunk:
                    break
                    
                start_time = chunk[0].get('start', 0)
                end_time = chunk[-1].get('end', start_time + 5)
                
                start_srt = self._seconds_to_srt_time(start_time)
                end_srt = self._seconds_to_srt_time(end_time)
                
                text = ' '.join([word.get('text', '') for word in chunk])
                
                f.write(f"{subtitle_num}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text}\n\n")
                subtitle_num += 1
    
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    
    



def main():
    """
    Example usage of the ElevenLabsTranscriptionService.
    """
    # Example usage
    try:
        # Initialize the service (make sure to set ELEVENLABS_API_KEY environment variable)
        service = ElevenLabsTranscriptionService()
        
        # Example: Transcribe a video file
        video_path = "../media/videoplayback.mp4"  # Replace with your video file path
        if os.path.exists(video_path):
            print(f"Transcribing video: {video_path}")
            result = service.transcribe_video(video_path)
            print("Transcription completed!")
            print(f"Text: {result.get('text', 'No text found')}")
            
            # Save transcript as SRT (and create condensed version if enabled)
            final_path = service.save_transcript(result, "../output/transcript.srt")
            
            if final_path != "../output/transcript.srt":
                print("Condensed transcript created with 25% of most important content!")
            else:
                print("Original transcript saved as SRT file!")
        else:
            print(f"Video file not found: {video_path}")
            print("Please provide a valid video file path.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to set your ELEVENLABS_API_KEY in the .env file or as an environment variable.")


if __name__ == "__main__":
    main()
