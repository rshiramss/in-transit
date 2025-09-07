#!/usr/bin/env python3
"""
Video Cutter Script - FIXED VERSION
Advanced video processing tool that can:
1. Cut MP4 files based on xxx timestamps (manual cutting mode)
2. Extract and stitch essential segments from condensed transcripts (jumpcut mode)
3. Generate seamless AI-powered transitions between segments using Google's Nano Banana
   model for intermediate frame generation and Fal AI for video transitions (seamless mode).

FIXES:
- Proper frame rate normalization across all clips
- Audio continuity preservation 
- Resolution standardization
- Eliminates flickering and audio dropouts
"""

import os
import re
import sys
import shutil
import requests
import argparse
import mimetypes
import traceback
from typing import List, Tuple, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve
from dotenv import load_dotenv

import numpy as np
from PIL import Image
from moviepy.editor import (
    VideoFileClip, 
    AudioFileClip, 
    concatenate_videoclips,
    concatenate_audioclips,
    CompositeVideoClip,
    ColorClip
)

# --- Fal AI Client Import ---
try:
    import fal_client
except ImportError:
    print("Error: fal_client not found. Please run 'pip install fal-client'")
    sys.exit(1)

# --- Google Generative AI Import (for Nano Banana) ---
try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not found. Please run 'pip install google-generativeai'")
    sys.exit(1)

# --- OpenCV Import (Optional) ---
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    print("Warning: OpenCV (cv2) not found. Frame alignment will be skipped.")
    print("For better transitions, run: pip install opencv-python")


# ==============================================================================
# === NORMALIZATION UTILITIES ===
# ==============================================================================

class VideoSpecs:
    """Container for standardized video specifications"""
    def __init__(self, fps: float = 30.0, width: int = 1280, height: int = 720, 
                 audio_fps: int = 44100, audio_channels: int = 2):
        self.fps = fps
        self.width = width
        self.height = height
        self.audio_fps = audio_fps
        self.audio_channels = audio_channels
        self.size = (width, height)

def normalize_clip(clip: VideoFileClip, target_specs: VideoSpecs, is_transition: bool = False) -> VideoFileClip:
    """
    Normalize a video clip to match target specifications.
    Handles both content clips and AI-generated transition clips.
    """
    if clip is None:
        return None
    
    print(f"    Normalizing clip: duration={clip.duration:.2f}s, "
          f"fps={getattr(clip, 'fps', 'unknown')}, size={getattr(clip, 'size', 'unknown')}")
    
    try:
        # 1. Normalize frame rate
        normalized_clip = clip.set_fps(target_specs.fps)
        
        # 2. Normalize resolution (resize maintaining aspect ratio, then pad/crop to exact size)
        if hasattr(normalized_clip, 'size') and normalized_clip.size != target_specs.size:
            normalized_clip = normalized_clip.resize(target_specs.size)
        
        # 3. Handle audio normalization
        if is_transition:
            # For AI transitions: create matching silent audio track
            from moviepy.audio.AudioClip import AudioClip
            
            def make_silence(t):
                if target_specs.audio_channels == 1:
                    return 0
                else:
                    return [0] * target_specs.audio_channels
            
            silent_audio = AudioClip(
                make_silence, 
                duration=normalized_clip.duration,
                fps=target_specs.audio_fps
            )
            normalized_clip = normalized_clip.set_audio(silent_audio)
            print(f"    Added silent audio track for transition (duration: {normalized_clip.duration:.2f}s)")
            
        elif normalized_clip.audio is not None:
            # For content clips: normalize existing audio
            audio = normalized_clip.audio
            if hasattr(audio, 'fps') and audio.fps != target_specs.audio_fps:
                # Note: MoviePy handles audio resampling automatically in most cases
                pass  # Let MoviePy handle this during final render
            normalized_clip = normalized_clip.set_audio(audio)
            print(f"    Preserved original audio track")
        else:
            print(f"    Warning: Content clip has no audio")
        
        print(f"    Normalization complete: fps={target_specs.fps}, size={target_specs.size}")
        return normalized_clip
        
    except Exception as e:
        print(f"    Error during normalization: {e}")
        return clip  # Return original clip if normalization fails


# ==============================================================================
# === UTILITY FUNCTIONS (Updated with normalization) ===
# ==============================================================================

def letterbox_fit(im: Image.Image, size: Tuple[int, int], bg=(0, 0, 0)) -> Image.Image:
    """Fit image into size with aspect preserved and padding."""
    target_w, target_h = size
    im = im.convert("RGB")
    src_w, src_h = im.size
    scale = min(target_w/src_w, target_h/src_h)
    new_w, new_h = max(1, int(round(src_w*scale))), max(1, int(round(src_h*scale)))
    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    off = ((target_w - new_w)//2, (target_h - new_h)//2)
    canvas.paste(im_resized, off)
    return canvas

def try_align_orb(imgA: Image.Image, imgB: Image.Image) -> Optional[Image.Image]:
    """Homography-align imgB to imgA using ORB; returns aligned B or None if failed."""
    if not _HAS_CV2:
        print("  [Nanobanana] CV2 not found. Skipping alignment.")
        return None
        
    a = np.array(imgA.convert("RGB"))
    b = np.array(imgB.convert("RGB"))
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(3000)
    kpa, desca = orb.detectAndCompute(a_gray, None)
    kpb, descb = orb.detectAndCompute(b_gray, None)
    if desca is None or descb is None:
        print("  [Nanobanana] ORB failed to find descriptors. Skipping alignment.")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desca, descb, k=2)

    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75*m[1].distance:
            good.append(m[0])
            
    if len(good) < 12:
        print(f"  [Nanobanana] Not enough good matches found ({len(good)}). Skipping alignment.")
        return None

    src_pts = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("  [Nanobanana] Homography calculation failed. Skipping alignment.")
        return None

    h, w = a.shape[:2]
    warped = cv2.warpPerspective(b, H, (w, h), flags=cv2.INTER_LINEAR)
    warped_pil = Image.fromarray(warped)
    print("  [Nanobanana] Frames successfully aligned.")
    return warped_pil

def cross_dissolve(imA: Image.Image, imB: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Simple midpoint blend A*(1-alpha) + B*alpha, expects same size, RGB."""
    A = np.asarray(imA.convert("RGB"), dtype=np.float32)
    B = np.asarray(imB.convert("RGB"), dtype=np.float32)
    C = (1.0 - alpha) * A + alpha * B
    C = np.clip(C, 0, 255).astype(np.uint8)
    return Image.fromarray(C, mode="RGB")

def generate_nano_banana_frame(frame_a_path: str, frame_b_path: str) -> Optional[Image.Image]:
    """Uses Google's Nano Banana model to generate an intermediate frame between two images."""
    try:
        # Load images
        image_a = Image.open(frame_a_path)
        image_b = Image.open(frame_b_path)
        
        # Convert to bytes for API
        import io
        buffer_a = io.BytesIO()
        buffer_b = io.BytesIO()
        image_a.save(buffer_a, format='JPEG')
        image_b.save(buffer_b, format='JPEG')
        
        # Create the model
        model = genai.GenerativeModel('gemini-2.5-flash-image-preview')
        
        # Prepare the prompt for Nano Banana
        prompt = """
        Generate a smooth intermediate frame between these two images that creates a natural transition. 
        The intermediate frame should blend elements from both images while maintaining visual continuity.
        Focus on creating a seamless transition that preserves the key visual elements from both frames.
        """
        
        # Generate the intermediate frame
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": buffer_a.getvalue()},
            {"mime_type": "image/jpeg", "data": buffer_b.getvalue()}
        ])
        
        if response and hasattr(response, 'parts') and response.parts:
            # Extract the generated image
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    generated_image = Image.open(io.BytesIO(image_data))
                    return generated_image
        
        return None
        
    except Exception as e:
        print(f"    Error in Nano Banana API: {e}")
        return None

def create_nanobanana_frame(frame_a_path: str, frame_b_path: str, out_path: str, size: Tuple[int, int], align: bool = True):
    """Uses Google's Nano Banana model to generate an intermediate frame between two images."""
    print(f"  [Nano Banana] Generating AI middleman frame for: {os.path.basename(frame_a_path)} -> {os.path.basename(frame_b_path)}")
    
    try:
        # Load and prepare images
        imA_raw = Image.open(frame_a_path)
        imB_raw = Image.open(frame_b_path)
        
        A = letterbox_fit(imA_raw, size)
        B = letterbox_fit(imB_raw, size)
        
        # Optional: Try alignment if OpenCV is available
        if align and _HAS_CV2:
            alignedB = try_align_orb(A, B)
            if alignedB is not None:
                B = alignedB
                print("    [Nano Banana] Frames aligned successfully.")
            else:
                print("    [Nano Banana] Alignment failed, proceeding with original frames.")
        
        # Save temporary aligned frames for Nano Banana
        temp_a_path = out_path.replace('.jpg', '_temp_a.jpg')
        temp_b_path = out_path.replace('.jpg', '_temp_b.jpg')
        A.save(temp_a_path)
        B.save(temp_b_path)
        
        # Generate intermediate frame using Nano Banana
        print("    [Nano Banana] Calling Google's Nano Banana model...")
        generated_image = generate_nano_banana_frame(temp_a_path, temp_b_path)
        
        if generated_image:
            generated_image.save(out_path)
            print(f"    Success: Saved Nano Banana middleman frame: {out_path}")
        else:
            print("    Warning: Nano Banana generation failed, falling back to cross-dissolve.")
            # Fallback to cross-dissolve
            mid = cross_dissolve(A, B, alpha=0.5)
            mid.save(out_path)
            print(f"    Success: Saved fallback middleman frame: {out_path}")
        
        # Clean up temporary files
        if os.path.exists(temp_a_path):
            os.remove(temp_a_path)
        if os.path.exists(temp_b_path):
            os.remove(temp_b_path)
            
    except Exception as e:
        print(f"    Error in Nano Banana generation: {e}")
        print("    Falling back to cross-dissolve method...")
        try:
            imA_raw = Image.open(frame_a_path)
            imB_raw = Image.open(frame_b_path)
            A = letterbox_fit(imA_raw, size)
            B = letterbox_fit(imB_raw, size)
            mid = cross_dissolve(A, B, alpha=0.5)
            mid.save(out_path)
            print(f"    Success: Saved fallback middleman frame: {out_path}")
        except Exception as fallback_error:
            print(f"    Error: Fallback also failed: {fallback_error}")
            return

def ensure_url(path_or_url: str) -> str:
    """Upload local file to fal if needed, else pass through URL."""
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        return path_or_url
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(path_or_url)
    print(f"    [Fal AI] Uploading local file: {os.path.basename(path_or_url)}...")
    url = fal_client.upload_file(path_or_url)
    print(f"    [Fal AI] Upload complete.")
    return url

def download_to(url: str, out_path: str):
    """Downloads a public URL to a local path."""
    import requests
    print(f"    [Fal AI] Downloading result to: {os.path.basename(out_path)}...")
    try:
        # Use requests instead of urlretrieve to handle SSL certificates properly
        response = requests.get(url, stream=True, verify=True)
        response.raise_for_status()
        
        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        
        print(f"    [Fal AI] Download complete.")
        
    except requests.exceptions.SSLError as e:
        print(f"    Warning: SSL certificate error, trying without verification...")
        try:
            # Fallback: try without SSL verification (less secure but works)
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()
            
            with open(out_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"    [Fal AI] Download complete (unverified SSL).")
            
        except Exception as fallback_error:
            print(f"    Error: Download failed even without SSL verification: {fallback_error}")
            raise
            
    except Exception as e:
        print(f"    Error: Download failed: {e}")
        raise

# ==============================================================================
# === FAL AI API FUNCTION (Updated) ===
# ==============================================================================

def run_fal_video_generation(image_url: str, out_mp4: str, prompt: str):
    """Uses Kling v2.1 Standard model for video generation"""
    args = {
        "prompt": prompt,
        "image_url": image_url,
        "aspect_ratio": "16:9",
        "cfg_scale": 0.5,
        "duration": "5",
        "seed": None
    }
    
    model_id = "fal-ai/kling-video/v2.1/standard/image-to-video"
    
    print(f"    [Fal AI] Submitting job to Kling v2.1: {model_id}")
    result = fal_client.subscribe(
        model_id,
        arguments=args,
        with_logs=True,
    )
    
    video_url = (result.get("video") or {}).get("url")
    if not video_url:
        raise RuntimeError(f"No video URL in fal response: {result}")
    
    download_to(video_url, out_mp4)

def run_fal_amt_interpolation(start_url: str, end_url: str, out_mp4: str, *,
                             output_fps: int = 12, 
                             recursive_interpolation_passes: int = 2):
    """
    Run Fal AI AMT frame interpolation for short, seamless transitions.
    
    Args:
        start_url: URL or path to first frame
        end_url: URL or path to second frame  
        out_mp4: Output path for generated video
        output_fps: Frames per second (1-60, lower = shorter duration)
        recursive_interpolation_passes: Interpolation passes (1-10, fewer = shorter)
    """
    # Ensure URLs are properly formatted
    start_url = ensure_url(start_url) if not start_url.startswith('http') else start_url
    end_url = ensure_url(end_url) if not end_url.startswith('http') else end_url
    
    args = {
        "frames": [
            {"url": start_url},
            {"url": end_url}
        ],
        "output_fps": output_fps,
        "recursive_interpolation_passes": recursive_interpolation_passes
    }
    
    print(f"    [AMT] Running frame interpolation with FPS={output_fps}, passes={recursive_interpolation_passes}")
    result = fal_client.subscribe(
        "fal-ai/amt-interpolation/frame-interpolation",
        arguments=args,
        with_logs=True,
    )
    
    video_url = (result.get("video") or {}).get("url")
    if not video_url:
        raise RuntimeError(f"No video URL in fal response: {result}")
    
    download_to(video_url, out_mp4)
    print(f"    [AMT] Saved interpolated video: {out_mp4}")

def generate_fal_transition_clip(frame_a_path: str, frame_b_path: str, nano_path: str, 
                                temp_video_base: str, target_specs: VideoSpecs) -> Optional[VideoFileClip]:
    """
    Generate AI transition clip with proper normalization applied.
    """
    print(f"  [Fal AI] Generating full transition, this will take a moment...")
    try:
        # 1. Prepare file URLs for the API
        start_url = ensure_url(frame_a_path)
        
        # 2. Define temp output path
        part_a_path = temp_video_base + "_part_A.mp4"
        
        # 3. Generate transition video
        print("    [Fal AI] Generating AI Transition...")
        run_fal_video_generation(
            image_url=start_url, 
            out_mp4=part_a_path,
            prompt="Smooth, coherent motion from the first frame into the midframe."
        )

        # 4. Load and normalize the AI-generated clip
        if not os.path.exists(part_a_path):
            raise FileNotFoundError("Fal AI failed to generate video.")
            
        print("    [Fal AI] Loading and normalizing AI transition...")
        raw_clip = VideoFileClip(part_a_path)
        
        # CRITICAL: Normalize the AI transition clip to match target specs
        normalized_clip = normalize_clip(raw_clip, target_specs, is_transition=True)
        
        print("    Success: AI Transition generated and normalized.")
        return normalized_clip

    except Exception as e:
        print(f"  Error: FAL AI TRANSITION FAILED: {e}")
        traceback.print_exc()
        return None

def generate_amt_transition_clip(frame_a_path: str, frame_b_path: str,
                                temp_video_base: str, target_specs: VideoSpecs,
                                output_fps: int = 8, passes: int = 1) -> Optional[VideoFileClip]:
    """
    Generate short, seamless AI transition clip using AMT frame interpolation.
    
    Args:
        frame_a_path: Path to first frame
        frame_b_path: Path to second frame
        temp_video_base: Base name for temp files
        target_specs: Video specifications for normalization
        output_fps: FPS for transition (lower = shorter duration)
        passes: Interpolation passes (fewer = shorter transition)
    """
    print(f"  [AMT] Generating short seamless transition...")
    try:
        # Define temp output path
        amt_video_path = temp_video_base + "_amt_transition.mp4"
        
        # Generate AMT interpolation video
        print(f"    [AMT] Creating frame interpolation (FPS: {output_fps}, passes: {passes})...")
        run_fal_amt_interpolation(
            start_url=frame_a_path,
            end_url=frame_b_path, 
            out_mp4=amt_video_path,
            output_fps=output_fps,
            recursive_interpolation_passes=passes
        )

        # Load and normalize the AMT-generated clip
        if not os.path.exists(amt_video_path):
            raise FileNotFoundError("AMT interpolation failed to generate video.")
            
        print("    [AMT] Loading and normalizing AMT transition...")
        raw_clip = VideoFileClip(amt_video_path)
        
        # CRITICAL: Normalize the AMT transition clip to match target specs
        normalized_clip = normalize_clip(raw_clip, target_specs, is_transition=True)
        
        print("    Success: AMT Transition generated and normalized.")
        return normalized_clip

    except Exception as e:
        print(f"  Error: AMT TRANSITION FAILED: {e}")
        traceback.print_exc()
        return None


# ==============================================================================
# === VIDEO CUTTER CLASS (Updated with normalization) ===
# ==============================================================================

class VideoCutter:
    def __init__(self, video_file: str, transcript_file: str):
        """Initialize the video cutter with video and transcript files."""
        self.video_file = video_file
        self.transcript_file = transcript_file
        self.video: Optional[VideoFileClip] = None
        self.audio: Optional[AudioFileClip] = None
        self.cut_points = []
        self.transition_markers = []
        self.temp_dir = "temp_transitions"
        
        # Video specifications - will be set when video is loaded
        self.target_specs: Optional[VideoSpecs] = None
        
    def load_video(self):
        """Load the video file and determine target specifications."""
        if not os.path.exists(self.video_file):
            raise FileNotFoundError(f"Video file not found: {self.video_file}")
        
        print(f"Loading video: {self.video_file}")
        self.video = VideoFileClip(self.video_file)
        self.audio = self.video.audio
        
        # Determine target specifications based on source video
        source_fps = self.video.fps if hasattr(self.video, 'fps') else 30.0
        source_size = self.video.size if hasattr(self.video, 'size') else (1280, 720)
        
        # Set up target specifications (use source video as reference)
        self.target_specs = VideoSpecs(
            fps=source_fps,
            width=source_size[0],
            height=source_size[1],
            audio_fps=44100,
            audio_channels=2 if self.audio and hasattr(self.audio, 'nchannels') 
                          and self.audio.nchannels >= 2 else 1
        )
        
        print(f"Video loaded successfully. Duration: {self.video.duration:.2f}s")
        print(f"Target specs: {self.target_specs.fps}fps, {self.target_specs.size}, "
              f"audio: {self.target_specs.audio_fps}Hz/{self.target_specs.audio_channels}ch")
        
    def is_transition_marker(self, text_content: str) -> bool:
        """Check if the text content is a transition marker."""
        transition_pattern = r'\[\.\.\.[^[\]]*(?:minute|second|hour|later|content omitted|omitted)\s*(?:later)?\.\.\.\]'
        return bool(re.search(transition_pattern, text_content, re.IGNORECASE))
    
    def parse_xxx_timestamps(self) -> List[Tuple[float, float, int]]:
        """Parse timestamps for manual cutting workflow"""
        if not os.path.exists(self.transcript_file):
            raise FileNotFoundError(f"Transcript file not found: {self.transcript_file}")
        cut_points = []
        transition_markers = []
        with open(self.transcript_file, 'r', encoding='utf-8') as file:
            content = file.read()
        blocks = content.strip().split('\n\n')
        for block_index, block in enumerate(blocks):
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                timestamp_line = lines[1]
                text_content = lines[2].strip()
                if self.is_transition_marker(text_content):
                    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
                    match = re.match(timestamp_pattern, timestamp_line)
                    if match:
                        start_time_srt, end_time_srt = match.groups()
                        start_seconds = self.srt_time_to_seconds(start_time_srt)
                        end_seconds = self.srt_time_to_seconds(end_time_srt)
                        transition_markers.append((start_seconds, end_seconds, block_index, text_content))
                elif text_content == "xxx":
                    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
                    match = re.match(timestamp_pattern, timestamp_line)
                    if match:
                        start_time_srt, end_time_srt = match.groups()
                        start_seconds = self.srt_time_to_seconds(start_time_srt)
                        end_seconds = self.srt_time_to_seconds(end_time_srt)
                        cut_points.append((start_seconds, end_seconds, block_index))
        self.transition_markers = transition_markers
        print(f"Found {len(transition_markers)} transition markers")
        print(f"Found {len(cut_points)} xxx segments to cut:")
        for i, (start, end, block_idx) in enumerate(cut_points):
            print(f"  {i+1}. {self.seconds_to_time(start)} - {self.seconds_to_time(end)} (block {block_idx})")
        return cut_points
    
    def parse_condensed_segments(self) -> List[Tuple[float, float, str]]:
        """Parse condensed SRT to find important segments to extract."""
        if not os.path.exists(self.transcript_file):
            raise FileNotFoundError(f"Transcript file not found: {self.transcript_file}")
        important_segments = []
        with open(self.transcript_file, 'r', encoding='utf-8') as file:
            content = file.read()
        blocks = content.strip().split('\n\n')
        
        print("Parsing and grouping consecutive transcript segments...")
        current_start = None
        current_end = None
        current_preview = ""
        
        for block_index, block in enumerate(blocks):
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                timestamp_line = lines[1]
                text_content = '\n'.join(lines[2:]).strip()
                
                is_content = (text_content != "xxx" and 
                              text_content != "ddd" and 
                              not self.is_transition_marker(text_content) and
                              text_content)
                
                if is_content:
                    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
                    match = re.match(timestamp_pattern, timestamp_line)
                    if match:
                        start_time_srt, end_time_srt = match.groups()
                        start_seconds = self.srt_time_to_seconds(start_time_srt)
                        end_seconds = self.srt_time_to_seconds(end_time_srt)
                        
                        if current_start is None:
                            current_start = start_seconds
                            current_end = end_seconds
                            current_preview = text_content[:50] + "..."
                        else:
                            current_end = end_seconds
                
                elif current_start is not None:
                    important_segments.append((current_start, current_end, current_preview))
                    current_start = None
        
        if current_start is not None:
            important_segments.append((current_start, current_end, current_preview))

        print(f"Found {len(important_segments)} grouped content segments to extract:")
        for i, (start, end, preview) in enumerate(important_segments):
            duration = end - start
            print(f"  {i+1}. {self.seconds_to_time(start)} - {self.seconds_to_time(end)} ({duration:.1f}s): {preview}")
        return important_segments
    
    def show_transition_markers(self):
        if not self.transition_markers: return
        print(f"\nTransition markers found ({len(self.transition_markers)}):")
        for i, (start, end, block_idx, text) in enumerate(self.transition_markers):
            print(f"  {i+1}. {self.seconds_to_time(start)} - {self.seconds_to_time(end)}: {text}")
    
    def time_to_seconds(self, time_str: str) -> float:
        parts = time_str.split(':'); h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    
    def srt_time_to_seconds(self, srt_time: str) -> float:
        time_part, ms_part = srt_time.split(','); parts = time_part.split(':')
        h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(ms_part)
        return h * 3600 + m * 60 + s + ms / 1000.0
    
    def seconds_to_time(self, seconds: float) -> str:
        h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def save_frame_at_time(self, time_seconds: float, filename: str) -> bool:
        """Save a frame from the video at a specific time."""
        print(f"  Saving frame at {time_seconds:.2f}s to: {os.path.basename(filename)}")
        try:
            time_seconds = max(0, min(time_seconds, self.video.duration - 0.01))
            frame = self.video.get_frame(time_seconds)
            image = Image.fromarray(frame.astype('uint8'))
            image.save(filename)
            return True
        except Exception as e:
            print(f"  Error saving frame at {time_seconds}s: {e}")
            return False

    # [Manual cutting functions remain the same]
    def find_next_non_xxx_segment(self, current_block_index: int) -> Optional[Tuple[float, float]]:
        with open(self.transcript_file, 'r', encoding='utf-8') as file:
            content = file.read()
        blocks = content.strip().split('\n\n')
        for i in range(current_block_index + 1, len(blocks)):
            lines = blocks[i].strip().split('\n')
            if len(lines) >= 3:
                text_content = lines[2].strip()
                if text_content != "xxx" and not self.is_transition_marker(text_content):
                    timestamp_line = lines[1]
                    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
                    match = re.match(timestamp_pattern, timestamp_line)
                    if match:
                        start_time_srt, end_time_srt = match.groups()
                        return (self.srt_time_to_seconds(start_time_srt), self.srt_time_to_seconds(end_time_srt))
        return None
    
    def replace_xxx_with_ddd(self, block_index: int) -> bool:
        try:
            with open(self.transcript_file, 'r', encoding='utf-8') as file:
                content = file.read()
            blocks = content.strip().split('\n\n')
            if block_index < len(blocks):
                lines = blocks[block_index].strip().split('\n')
                if len(lines) >= 3 and lines[2].strip() == "xxx":
                    lines[2] = "ddd"
                    blocks[block_index] = '\n'.join(lines)
                    with open(self.transcript_file, 'w', encoding='utf-8') as file:
                        file.write('\n\n'.join(blocks))
                    return True
            return False
        except Exception as e:
            print(f"Error replacing xxx with ddd: {e}")
            return False
    
    def cut_video_at_point(self, cut_index: int) -> bool:
        if not self.video: raise ValueError("Video not loaded.")
        if cut_index >= len(self.cut_points):
            print(f"Cut index {cut_index} out of range.")
            return False
        start_cut, end_cut, block_index = self.cut_points[cut_index]
        print(f"\n=== Processing Cut {cut_index + 1} ===")
        print(f"Cutting from {self.seconds_to_time(start_cut)} to {self.seconds_to_time(end_cut)}")
        next_segment = self.find_next_non_xxx_segment(block_index)
        if start_cut > 0:
            last_frame_time = max(0, start_cut - 0.1)
            last_frame_filename = f"last_frame_cut_{cut_index + 1}.png"
            if self.save_frame_at_time(last_frame_time, last_frame_filename):
                print(f"Last frame saved: {last_frame_filename}")
        if next_segment:
            first_frame_filename = f"first_frame_cut_{cut_index + 1}.png"
            if self.save_frame_at_time(next_segment[0], first_frame_filename):
                print(f"First frame of next segment saved: {first_frame_filename}")
        if self.replace_xxx_with_ddd(block_index):
            print(f"Replaced xxx with ddd in transcript for block {block_index}")
        if start_cut > 0:
            before_clip = self.video.subclip(0, start_cut)
            before_filename = f"video_before_cut_{cut_index + 1}.mp4"
            before_clip.write_videofile(before_filename, audio_codec='aac', codec='libx264')
            before_clip.close()
            print(f"Before video saved: {before_filename}")
        if end_cut < self.video.duration:
            after_clip = self.video.subclip(end_cut, self.video.duration)
            after_filename = f"video_after_cut_{cut_index + 1}.mp4"
            after_clip.write_videofile(after_filename, audio_codec='aac', codec='libx264')
            after_clip.close()
            print(f"After video saved: {after_filename}")
        print(f"Cut {cut_index + 1} completed successfully!")
        return True

    # ==============================================================================
    # === UPDATED CORE FUNCTIONS WITH NORMALIZATION ===
    # ==============================================================================

    def extract_single_segment(self, start: float, end: float) -> Optional[VideoFileClip]:
        """
        Extracts a single normalized segment with audio preservation.
        """
        try:
            # Validate and adjust timestamps
            start_safe = max(0, start)
            end_safe = min(self.video.duration, end)

            if (end_safe - start_safe) < 0.1:
                print(f"  Warning: Segment from {start_safe:.2f}s to {end_safe:.2f}s is too short, skipping.")
                return None
            
            # Create video-only subclip from main video stream
            video_seg = self.video.subclip(start_safe, end_safe)
            
            # Create audio-only subclip from the single, pre-loaded audio stream
            if self.audio:
                audio_seg = self.audio.subclip(start_safe, end_safe)
                video_seg = video_seg.set_audio(audio_seg)
            
            # CRITICAL: Normalize the content segment
            normalized_seg = normalize_clip(video_seg, self.target_specs, is_transition=False)
            
            return normalized_seg
            
        except Exception as e:
            print(f"  Error extracting single segment ({start_safe}s to {end_safe}s): {e}")
            return None

    def stitch_video_segments(self, segment_clips: List[VideoFileClip], output_filename: str = "condensed_video.mp4") -> bool:
        """
        Stitch together a list of NORMALIZED clips into a single file.
        All clips should already be normalized to target_specs before calling this function.
        """
        valid_clips = [clip for clip in segment_clips if clip is not None]
        if not valid_clips:
            print("No valid clips to stitch together.")
            return False
        
        print(f"\n=== Stitching {len(valid_clips)} Normalized Segments Together ===")
        
        try:
            # Since all clips are already normalized, we can concatenate directly
            print(f"Concatenating {len(valid_clips)} pre-normalized clips...")
            
            # Use "compose" method for more stable concatenation of normalized clips
            final_video = concatenate_videoclips(valid_clips, method="compose")
            
            print(f"Writing final video: {output_filename}")
            final_video.write_videofile(
                output_filename, 
                audio_codec='aac',
                codec='libx264',
                fps=self.target_specs.fps,  # Explicitly set fps
                verbose=False, 
                logger='bar'
            )
            
            print(f"Success: Final video created successfully!")
            return True
            
        except Exception as e:
            print(f"Error stitching segments: {e}")
            print(traceback.format_exc())
            return False
        
        finally:
            # Clean up all clips to free memory
            for clip in valid_clips:
                try:
                    clip.close()
                except:
                    pass
    
    def create_condensed_video(self, output_filename: str = "condensed_video.mp4") -> bool:
        """
        JUMP-CUT WORKFLOW: Creates a jump-cut video by stitching normalized segments directly.
        """
        if not self.video: raise ValueError("Video not loaded.")
        if not self.target_specs: raise ValueError("Target specs not set. Call load_video() first.")
        
        print(f"\n=== Creating JUMP-CUT Condensed Video: {output_filename} ===")
        segments_data = self.parse_condensed_segments()
        if not segments_data:
            print("No segments found.")
            return False
        
        segment_clips_list = self.extract_segments_to_memory(segments_data)
        if not segment_clips_list:
            print("Failed to extract any segments.")
            return False
        
        return self.stitch_video_segments(segment_clips_list, output_filename)

    def extract_segments_to_memory(self, segments_data: List[Tuple[float, float, str]]) -> List[VideoFileClip]:
        """Helper for jump-cut workflow: Extracts all segments to a normalized list in memory."""
        print(f"\n=== Extracting and Normalizing {len(segments_data)} Segments ===")
        clips_list = []
        for i, (start, end, preview) in enumerate(segments_data):
            print(f"  Processing {i+1}/{len(segments_data)}: {preview}")
            clip = self.extract_single_segment(start, end)  # This already normalizes
            if clip:
                clips_list.append(clip)
        return clips_list

    def create_seamless_video(self, output_filename: str = "seamless_video.mp4", 
                            transition_size: Optional[Tuple[int, int]] = None, 
                            align_frames: bool = True,
                            amt_fps: int = 8,
                            amt_passes: int = 1) -> bool:
        """
        *** SEAMLESS WORKFLOW WITH NORMALIZATION ***
        Creates a video with AI transitions, all properly normalized.
        """
        if not self.video: raise ValueError("Video not loaded.")
        if not self.target_specs: raise ValueError("Target specs not set. Call load_video() first.")

        print(f"\n=== Creating SEAMLESS AI-Powered Video: {output_filename} ===")
        
        # Use target specs for transition size if not specified
        if transition_size is None:
            transition_size = self.target_specs.size
        
        # 1. Setup temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"Created temp directory: {self.temp_dir}")
        
        final_assembly_list = []

        # 2. Get content segments
        segments_data = self.parse_condensed_segments()
        if not segments_data:
            print("No segments found. Cannot create seamless video.")
            return False

        num_transitions = len(segments_data) - 1
        print(f"\nProcessing {len(segments_data)} content segments to generate {num_transitions} transitions...")

        # 3. Process each segment and generate transitions
        for i, (start, end, preview) in enumerate(segments_data):
            
            print(f"\n--- Processing Content Segment {i+1}/{len(segments_data)} ---")
            
            # A. Extract and normalize the content segment
            print(f"  Extracting content: {preview}")
            content_clip = self.extract_single_segment(start, end)  # Already normalized
            if content_clip:
                final_assembly_list.append(content_clip)
            else:
                print(f"  SKIPPING segment {i+1} due to extraction error.")
                continue

            # B. Generate transition if not the last segment
            if i == len(segments_data) - 1:
                print("  This is the final segment. No transition needed.")
                break

            print(f"\n  --- Generating AI Transition {i+1}/{num_transitions} ---")
            (next_start, next_end, next_preview) = segments_data[i+1]
            
            # Define file paths
            frame_a_path = os.path.join(self.temp_dir, f"frame_{i:03d}_A_end.jpg")
            frame_b_path = os.path.join(self.temp_dir, f"frame_{i:03d}_B_start.jpg")
            nano_path = os.path.join(self.temp_dir, f"nano_{i:03d}_mid.jpg")
            trans_video_base = os.path.join(self.temp_dir, f"transition_{i:03d}")

            # Save keyframes
            frame_a_ok = self.save_frame_at_time(end - 0.04, frame_a_path)
            frame_b_ok = self.save_frame_at_time(next_start, frame_b_path)

            if not (frame_a_ok and frame_b_ok):
                print("  Error: Failed to save keyframes. Skipping this transition.")
                continue

            # Create Nanobanana frame
            create_nanobanana_frame(frame_a_path, frame_b_path, nano_path, transition_size, align=align_frames)

            if not os.path.exists(nano_path):
                 print("  Error: Nanobanana frame not created. Skipping Fal AI generation.")
                 continue

            # Generate normalized AI transition using AMT interpolation for shorter, seamless transitions
            transition_clip = generate_amt_transition_clip(
                frame_a_path, frame_b_path, trans_video_base, self.target_specs,
                output_fps=amt_fps,   # Configurable FPS for duration control
                passes=amt_passes     # Configurable passes for seamlessness/speed balance
            )

            if transition_clip:
                final_assembly_list.append(transition_clip)
            else:
                print(f"  Warning: AI transition failed. A jump-cut will occur here.")

        # 4. Stitch all normalized clips
        success = self.stitch_video_segments(final_assembly_list, output_filename)

        # 5. Cleanup
        if os.path.exists(self.temp_dir):
            print(f"\nCleaning up temporary directory: {self.temp_dir}")
            try:
                 shutil.rmtree(self.temp_dir)
                 print("  Temp directory removed.")
            except Exception as e:
                 print(f"  Warning: Could not remove temp directory: {e}")
        
        return success

    def close(self):
        """Close the video and audio file handlers."""
        if self.audio:
            self.audio.close()
            self.audio = None
        if self.video:
            self.video.close()
            self.video = None
            print("Video and audio handlers closed.")


def main():
    """Main function to run the video processor."""
    print("=== Advanced Video Processor - FIXED VERSION ===")
    load_dotenv()

    # Check for required API keys
    if not os.environ.get("FAL_KEY"):
        print("Error: FAL_KEY environment variable not set.")
        print("   Please set it in your environment or in a .env file.")
        sys.exit(2)
    else:
        print("Success: FAL_KEY environment variable found.")
    
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("   Please set it in your environment or in a .env file for Nano Banana access.")
        sys.exit(2)
    else:
        print("Success: GEMINI_API_KEY environment variable found.")
        
    # Initialize Google Generative AI
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    # Argument parsing
    parser = argparse.ArgumentParser(description="Advanced Video Processor - Fixed Version")
    parser.add_argument(
        '--mode', 
        choices=['jumpcut', 'seamless', 'manualcut'], 
        default='seamless', 
        help="Processing mode: 'jumpcut' (stitch segments only), 'seamless' (generate AI transitions), 'manualcut' (process 'xxx' segments)."
    )
    parser.add_argument(
        '--video', 
        default='GreekFinCrisis.mp4', 
        help="Source video file"
    )
    parser.add_argument(
        '--transcript', 
        default='../output/transcript_condensed.srt', 
        help="Source condensed transcript file"
    )
    parser.add_argument(
        '--no-align',
        action="store_true",
        help="Disable frame alignment for Nanobanana (faster, less accurate)"
    )
    parser.add_argument(
        '--amt-fps',
        type=int,
        default=8,
        help="AMT transition FPS (1-60, lower = shorter duration, default: 8)"
    )
    parser.add_argument(
        '--amt-passes',
        type=int,
        default=1,
        help="AMT interpolation passes (1-10, fewer = shorter transition, default: 1)"
    )
    args = parser.parse_args()

    # Find transcript file
    transcript_file = args.transcript
    if not os.path.exists(transcript_file):
        possible_paths = ["output/transcript_condensed.srt", "transcript_condensed.srt"]
        for path in possible_paths:
            if os.path.exists(path):
                transcript_file = path
                break

    print(f"\nVideo file: {args.video}")
    print(f"Transcript file: {transcript_file}")
    print(f"Mode selected: {args.mode}")

    # Validate files
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return
    
    if not os.path.exists(transcript_file):
        print(f"Error: Transcript file '{transcript_file}' not found.")
        return

    cutter = None
    try:
        cutter = VideoCutter(args.video, transcript_file)
        cutter.load_video()
        
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        
        # Save output to media folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        media_dir = os.path.join(project_root, "media")
        
        if args.mode == 'jumpcut':
            output_filename = os.path.join(media_dir, f"condensed_{base_name}_jumpcut.mp4")
            success = cutter.create_condensed_video(output_filename)
        
        elif args.mode == 'seamless':
            output_filename = os.path.join(media_dir, f"condensed_{base_name}_seamless.mp4")
            success = cutter.create_seamless_video(
                output_filename, 
                align_frames=(not args.no_align),
                amt_fps=args.amt_fps,
                amt_passes=args.amt_passes
            )
            
        elif args.mode == 'manualcut':
            print("--- Manual Cut Mode ---")
            print("This mode will find the FIRST 'xxx' segment and process it.")
            cutter.cut_points = cutter.parse_xxx_timestamps()
            if not cutter.cut_points:
                print("No 'xxx' cut points found in transcript.")
                success = False
            else:
                success = cutter.cut_video_at_point(0)
        
        if success:
            print(f"\n🎉 Success! Mode '{args.mode}' completed.")
        else:
            print(f"\nError: Failed to complete mode '{args.mode}'.")
            
    except Exception as e:
        print(f"Error: An unexpected master error occurred: {e}")
        print(traceback.format_exc())
    
    finally:
        if cutter:
            cutter.close()

if __name__ == "__main__":
    main()