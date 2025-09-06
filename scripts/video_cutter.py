#!/usr/bin/env python3
"""
Video Cutter Script
Cuts MP4 files based on xxx timestamps from cleaned transcript files.
Processes one cut at a time and creates before/after video files.
"""

import os
import re
from typing import List, Tuple, Optional
from moviepy import VideoFileClip

class VideoCutter:
    def __init__(self, video_file: str, transcript_file: str):
        """Initialize the video cutter with video and transcript files."""
        self.video_file = video_file
        self.transcript_file = transcript_file
        self.video = None
        self.cut_points = []
        
    def load_video(self):
        """Load the video file."""
        if not os.path.exists(self.video_file):
            raise FileNotFoundError(f"Video file not found: {self.video_file}")
        
        print(f"Loading video: {self.video_file}")
        self.video = VideoFileClip(self.video_file)
        print(f"Video loaded successfully. Duration: {self.video.duration:.2f} seconds")
        
    def parse_xxx_timestamps(self) -> List[Tuple[float, float]]:
        """Parse the transcript file to find xxx timestamps."""
        if not os.path.exists(self.transcript_file):
            raise FileNotFoundError(f"Transcript file not found: {self.transcript_file}")
        
        cut_points = []
        
        with open(self.transcript_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Regex pattern to match timestamp format: 00:00:00 - 00:00:03: xxx
        pattern = r'(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2}):\s*xxx'
        matches = re.findall(pattern, content)
        
        for start_time, end_time in matches:
            start_seconds = self.time_to_seconds(start_time)
            end_seconds = self.time_to_seconds(end_time)
            cut_points.append((start_seconds, end_seconds))
        
        print(f"Found {len(cut_points)} xxx segments to cut:")
        for i, (start, end) in enumerate(cut_points):
            print(f"  {i+1}. {self.seconds_to_time(start)} - {self.seconds_to_time(end)}")
        
        return cut_points
    
    def time_to_seconds(self, time_str: str) -> float:
        """Convert time string (HH:MM:SS) to seconds."""
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    
    def seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to time string (HH:MM:SS)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def cut_video_at_point(self, cut_index: int) -> bool:
        """Cut the video at a specific xxx point and create before/after files."""
        if not self.video:
            raise ValueError("Video not loaded. Call load_video() first.")
        
        if cut_index >= len(self.cut_points):
            print(f"Cut index {cut_index} is out of range. Total cuts available: {len(self.cut_points)}")
            return False
        
        start_cut, end_cut = self.cut_points[cut_index]
        video_duration = self.video.duration
        
        print(f"\n=== Processing Cut {cut_index + 1} ===")
        print(f"Cutting from {self.seconds_to_time(start_cut)} to {self.seconds_to_time(end_cut)}")
        
        # Create before video (from start to cut point)
        if start_cut > 0:
            before_clip = self.video.subclip(0, start_cut)
            before_filename = f"video_before_cut_{cut_index + 1}.mp4"
            print(f"Creating before video: {before_filename}")
            before_clip.write_videofile(before_filename, verbose=False, logger=None)
            before_clip.close()
            print(f"✓ Before video saved: {before_filename}")
        else:
            print("No before video needed (cut starts at beginning)")
        
        # Create after video (from cut end to video end)
        if end_cut < video_duration:
            after_clip = self.video.subclip(end_cut, video_duration)
            after_filename = f"video_after_cut_{cut_index + 1}.mp4"
            print(f"Creating after video: {after_filename}")
            after_clip.write_videofile(after_filename, verbose=False, logger=None)
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
    video_file = "input_video.mp4"  # Change this to your video file
    transcript_file = "transcribed_text_cleaned.txt"
    
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
