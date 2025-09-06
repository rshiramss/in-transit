#!/usr/bin/env python3
"""
Video Cutter Script
Cuts MP4 files based on xxx timestamps from cleaned transcript files.
Handles transition markers like [...1 minute later...] and skips them during processing.
Processes one cut at a time and creates before/after video files.
"""

import os
import re
from typing import List, Tuple, Optional
from moviepy import VideoFileClip
import numpy as np
from PIL import Image

class VideoCutter:
    def __init__(self, video_file: str, transcript_file: str):
        """Initialize the video cutter with video and transcript files."""
        self.video_file = video_file
        self.transcript_file = transcript_file
        self.video = None
        self.cut_points = []
        self.transition_markers = []
        
    def load_video(self):
        """Load the video file."""
        if not os.path.exists(self.video_file):
            raise FileNotFoundError(f"Video file not found: {self.video_file}")
        
        print(f"Loading video: {self.video_file}")
        self.video = VideoFileClip(self.video_file)
        print(f"Video loaded successfully. Duration: {self.video.duration:.2f} seconds")
        
    def is_transition_marker(self, text_content: str) -> bool:
        """Check if the text content is a transition marker like [...1 minute later...]."""
        # Pattern to match transition markers
        transition_pattern = r'\[\.\.\.[^[\]]*(?:minute|second|hour|later|content omitted|omitted)\s*(?:later)?\.\.\.\]'
        return bool(re.search(transition_pattern, text_content, re.IGNORECASE))
    
    def parse_xxx_timestamps(self) -> List[Tuple[float, float, int]]:
        """Parse the SRT transcript file to find xxx timestamps and their positions."""
        if not os.path.exists(self.transcript_file):
            raise FileNotFoundError(f"Transcript file not found: {self.transcript_file}")
        
        cut_points = []
        transition_markers = []
        
        with open(self.transcript_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split content into subtitle blocks
        blocks = content.strip().split('\n\n')
        
        for block_index, block in enumerate(blocks):
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Parse timestamp line (second line)
                timestamp_line = lines[1]
                # Parse text content (third line)
                text_content = lines[2].strip()
                
                # Check if this is a transition marker
                if self.is_transition_marker(text_content):
                    # Parse SRT timestamp format: 00:00:00,000 --> 00:00:03,000
                    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
                    match = re.match(timestamp_pattern, timestamp_line)
                    
                    if match:
                        start_time_srt, end_time_srt = match.groups()
                        start_seconds = self.srt_time_to_seconds(start_time_srt)
                        end_seconds = self.srt_time_to_seconds(end_time_srt)
                        transition_markers.append((start_seconds, end_seconds, block_index, text_content))
                        print(f"Found transition marker: {text_content} at {self.seconds_to_time(start_seconds)}")
                
                # Check if this is an xxx segment
                elif text_content == "xxx":
                    # Parse SRT timestamp format: 00:00:00,000 --> 00:00:03,000
                    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
                    match = re.match(timestamp_pattern, timestamp_line)
                    
                    if match:
                        start_time_srt, end_time_srt = match.groups()
                        # Convert SRT format to seconds
                        start_seconds = self.srt_time_to_seconds(start_time_srt)
                        end_seconds = self.srt_time_to_seconds(end_time_srt)
                        cut_points.append((start_seconds, end_seconds, block_index))
        
        # Store transition markers for reference
        self.transition_markers = transition_markers
        
        print(f"Found {len(transition_markers)} transition markers")
        print(f"Found {len(cut_points)} xxx segments to cut:")
        for i, (start, end, block_idx) in enumerate(cut_points):
            print(f"  {i+1}. {self.seconds_to_time(start)} - {self.seconds_to_time(end)} (block {block_idx})")
        
        return cut_points
    
    def show_transition_markers(self):
        """Display all found transition markers."""
        if not self.transition_markers:
            print("No transition markers found.")
            return
        
        print(f"\nTransition markers found ({len(self.transition_markers)}):")
        for i, (start, end, block_idx, text) in enumerate(self.transition_markers):
            print(f"  {i+1}. {self.seconds_to_time(start)} - {self.seconds_to_time(end)}: {text}")
    
    def time_to_seconds(self, time_str: str) -> float:
        """Convert time string (HH:MM:SS) to seconds."""
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def srt_time_to_seconds(self, srt_time: str) -> float:
        """Convert SRT time string (HH:MM:SS,mmm) to seconds."""
        # Split time and milliseconds
        time_part, ms_part = srt_time.split(',')
        parts = time_part.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        milliseconds = int(ms_part)
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    
    
    def seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to time string (HH:MM:SS)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def save_frame_at_time(self, time_seconds: float, filename: str) -> bool:
        """Save a frame from the video at a specific time."""
        try:
            frame = self.video.get_frame(time_seconds)
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame.astype('uint8'))
            image.save(filename)
            return True
        except Exception as e:
            print(f"Error saving frame at {time_seconds}s: {e}")
            return False
    
    def find_next_non_xxx_segment(self, current_block_index: int) -> Optional[Tuple[float, float]]:
        """Find the next non-xxx and non-transition marker segment after the current block."""
        with open(self.transcript_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        blocks = content.strip().split('\n\n')
        
        for i in range(current_block_index + 1, len(blocks)):
            lines = blocks[i].strip().split('\n')
            if len(lines) >= 3:
                text_content = lines[2].strip()
                # Skip both xxx segments and transition markers
                if text_content != "xxx" and not self.is_transition_marker(text_content):
                    # Parse timestamp
                    timestamp_line = lines[1]
                    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
                    match = re.match(timestamp_pattern, timestamp_line)
                    if match:
                        start_time_srt, end_time_srt = match.groups()
                        start_seconds = self.srt_time_to_seconds(start_time_srt)
                        end_seconds = self.srt_time_to_seconds(end_time_srt)
                        return (start_seconds, end_seconds)
        return None
    
    def replace_xxx_with_ddd(self, block_index: int) -> bool:
        """Replace xxx with ddd in the transcript file for a specific block."""
        try:
            with open(self.transcript_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            blocks = content.strip().split('\n\n')
            
            if block_index < len(blocks):
                lines = blocks[block_index].strip().split('\n')
                if len(lines) >= 3 and lines[2].strip() == "xxx":
                    lines[2] = "ddd"
                    blocks[block_index] = '\n'.join(lines)
                    
                    # Write back to file
                    with open(self.transcript_file, 'w', encoding='utf-8') as file:
                        file.write('\n\n'.join(blocks))
                    return True
            return False
        except Exception as e:
            print(f"Error replacing xxx with ddd: {e}")
            return False
    
    def cut_video_at_point(self, cut_index: int) -> bool:
        """Cut the video at a specific xxx point and create before/after files."""
        if not self.video:
            raise ValueError("Video not loaded. Call load_video() first.")
        
        if cut_index >= len(self.cut_points):
            print(f"Cut index {cut_index} is out of range. Total cuts available: {len(self.cut_points)}")
            return False
        
        start_cut, end_cut, block_index = self.cut_points[cut_index]
        video_duration = self.video.duration
        
        print(f"\n=== Processing Cut {cut_index + 1} ===")
        print(f"Cutting from {self.seconds_to_time(start_cut)} to {self.seconds_to_time(end_cut)}")
        
        # Find the next non-xxx segment for frame capture
        next_segment = self.find_next_non_xxx_segment(block_index)
        
        # Save frames
        if start_cut > 0:
            # Save last frame of the segment before the cut
            last_frame_time = start_cut - 0.1  # Slightly before the cut
            if last_frame_time >= 0:
                last_frame_filename = f"last_frame_cut_{cut_index + 1}.png"
                if self.save_frame_at_time(last_frame_time, last_frame_filename):
                    print(f"✓ Last frame saved: {last_frame_filename}")
        
        if next_segment:
            next_start, next_end = next_segment
            # Save first frame of the next non-xxx segment
            first_frame_filename = f"first_frame_cut_{cut_index + 1}.png"
            if self.save_frame_at_time(next_start, first_frame_filename):
                print(f"✓ First frame of next segment saved: {first_frame_filename}")
        
        # Replace xxx with ddd in transcript
        if self.replace_xxx_with_ddd(block_index):
            print(f"✓ Replaced xxx with ddd in transcript for block {block_index}")
        
        # Create before video (from start to cut point)
        if start_cut > 0:
            before_clip = self.video.subclipped(0, start_cut)
            before_filename = f"video_before_cut_{cut_index + 1}.mp4"
            print(f"Creating before video: {before_filename}")
            before_clip.write_videofile(before_filename)
            before_clip.close()
            print(f"✓ Before video saved: {before_filename}")
        else:
            print("No before video needed (cut starts at beginning)")
        
        # Create after video (from cut end to video end)
        if end_cut < video_duration:
            after_clip = self.video.subclipped(end_cut, video_duration)
            after_filename = f"video_after_cut_{cut_index + 1}.mp4"
            print(f"Creating after video: {after_filename}")
            after_clip.write_videofile(after_filename)
            after_clip.close()
            print(f"✓ After video saved: {after_filename}")
        else:
            print("No after video needed (cut ends at video end)")
        
        print(f"✓ Cut {cut_index + 1} completed successfully!")
        return True
    
    def close(self):
        """Close the video file."""
        if self.video:
            self.video.close()
            self.video = None

def main():
    """Main function to run the video cutter."""
    # Configuration
    video_file = "videoplayback.mp4"  # Change this to your video file
    transcript_file = "transcript_cleaned.srt"
    
    print("=== Video Cutter ===")
    print(f"Video file: {video_file}")
    print(f"Transcript file: {transcript_file}")
    
    # Check if files exist
    if not os.path.exists(video_file):
        print(f"❌ Error: Video file '{video_file}' not found.")
        print("Please place your MP4 file in the current directory and update the video_file variable.")
        return
    
    if not os.path.exists(transcript_file):
        print(f"❌ Error: Transcript file '{transcript_file}' not found.")
        print("Please run the text processor first to create the cleaned transcript.")
        return
    
    try:
        # Initialize cutter
        cutter = VideoCutter(video_file, transcript_file)
        
        # Load video
        cutter.load_video()
        
        # Parse xxx timestamps
        cutter.cut_points = cutter.parse_xxx_timestamps()
        
        # Show transition markers
        cutter.show_transition_markers()
        
        if not cutter.cut_points:
            print("No xxx segments found in transcript. Nothing to cut.")
            return
        
        # Process first cut only
        print(f"\nProcessing first cut (1 of {len(cutter.cut_points)})...")
        success = cutter.cut_video_at_point(0)
        
        if success:
            print("\n🎉 Done!")
            print("First cut completed. Run the script again to process the next cut.")
        else:
            print("\n❌ Failed to process cut.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        if 'cutter' in locals():
            cutter.close()

if __name__ == "__main__":
    main()
