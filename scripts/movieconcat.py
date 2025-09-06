#!/usr/bin/env python3
"""
Combine two video parts into one using moviepy.

Usage:
  python combine_videos.py --part1 first_to_nanobanana.mp4 \
                           --part2 nanobanana_to_second.mp4 \
                           --out final_transition.mp4
"""

import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips

def main():
    parser = argparse.ArgumentParser(description="Concatenate two video parts into one")
    parser.add_argument("--part1", required=True, help="Path to first video file")
    parser.add_argument("--part2", required=True, help="Path to second video file")
    parser.add_argument("--out", required=True, help="Output video filename (e.g., final.mp4)")
    args = parser.parse_args()

    # Load video parts
    clip1 = VideoFileClip(args.part1)
    clip2 = VideoFileClip(args.part2)

    # Concatenate
    final = concatenate_videoclips([clip1, clip2], method="compose")

    # Write output file
    final.write_videofile(args.out, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    main()
