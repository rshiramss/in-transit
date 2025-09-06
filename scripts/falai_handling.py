#!/usr/bin/env python3
"""
Make a transition video between two images using fal.ai (Wan-2.1 FLF2V)

Usage:
  python flf2v_transition.py \
    --start path/or/url/to/start.jpg \
    --end path/or/url/to/end.jpg \
    --out transition.mp4 \
    --prompt "Describe how the scene should evolve" \
    --frames 81 --fps 16 --resolution 720p --aspect auto

Setup:
  pip install fal-client
  export FAL_KEY="YOUR_FAL_API_KEY"
Docs:
  Model schema & params: https://fal.ai/models/fal-ai/wan-flf2v/api
"""

import os
import argparse
import mimetypes
import fal_client
import sys
from urllib.parse import urlparse

def is_url(s: str) -> bool:
    try:
        return urlparse(s).scheme in ("http", "https")
    except Exception:
        return False

def ensure_url(path_or_url: str) -> str:
    """Return a URL usable by fal: upload local files, pass through URLs."""
    if is_url(path_or_url):
        return path_or_url
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"File not found: {path_or_url}")
    # Hint content-type to keep extensions nice
    ctype, _ = mimetypes.guess_type(path_or_url)
    return fal_client.upload_file(path_or_url, content_type=ctype or "application/octet-stream")

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            # Stream model logs as they arrive
            msg = log.get("message")
            if msg:
                print(msg, file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Generate a transition video between two images using fal.ai Wan-2.1 FLF2V")
    ap.add_argument("--start", required=True, help="Path or URL to the starting image")
    ap.add_argument("--end", required=True, help="Path or URL to the ending image")
    ap.add_argument("--prompt", default="Smooth, coherent transition from the first image to the last, natural motion.",
                    help="Text prompt to guide the transition")
    ap.add_argument("--negative", default=None, help="Optional negative prompt")
    ap.add_argument("--frames", type=int, default=81, help="Number of frames (81–100). Default: 81")
    ap.add_argument("--fps", type=int, default=16, help="Frames per second (5–24). Default: 16")
    ap.add_argument("--resolution", choices=["480p", "720p"], default="720p", help="Output resolution")
    ap.add_argument("--aspect", choices=["auto", "16:9", "9:16", "1:1"], default="auto", help="Aspect ratio behavior")
    ap.add_argument("--steps", type=int, default=30, help="Num inference steps (quality/speed tradeoff)")
    ap.add_argument("--cfg", type=float, default=5.0, help="Classifier-free guidance scale")
    ap.add_argument("--shift", type=float, default=5.0, help="Shift parameter")
    ap.add_argument("--seed", type=int, default=None, help="Optional random seed")
    ap.add_argument("--acceleration", choices=["none", "regular"], default="regular", help="Acceleration level")
    ap.add_argument("--out", default="transition.mp4", help="Output MP4 filename")
    args = ap.parse_args()

    if not os.environ.get("FAL_KEY"):
        print("ERROR: FAL_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(2)

    # Turn local paths into temporary hosted URLs fal can fetch.
    start_url = ensure_url(args.start)
    end_url = ensure_url(args.end)

    # Build model arguments exactly per the model schema.
    model_args = {
        "prompt": args.prompt,
        "start_image_url": start_url,
        "end_image_url": end_url,
        "num_frames": args.frames,
        "frames_per_second": args.fps,
        "resolution": args.resolution,
        "num_inference_steps": args.steps,
        "guide_scale": args.cfg,
        "shift": args.shift,
        "enable_safety_checker": True,
        "enable_prompt_expansion": False,
        "acceleration": args.acceleration,
        "aspect_ratio": args.aspect,
    }
    if args.negative:
        model_args["negative_prompt"] = args.negative
    if args.seed is not None:
        model_args["seed"] = args.seed

    print("Submitting job to fal.ai …", file=sys.stderr)

    # Use subscribe() to stream logs and wait until complete.
    result = fal_client.subscribe(
        "fal-ai/wan-flf2v",
        arguments=model_args,
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    # Result schema: {"video": {"url": ...}, "seed": ...}
    video_info = result.get("video") or {}
    video_url = video_info.get("url")
    if not video_url:
        print(f"Unexpected response, no video URL. Full result:\n{result}", file=sys.stderr)
        sys.exit(1)

    # Download to --out
    import urllib.request
    print(f"Downloading video to {args.out} …", file=sys.stderr)
    urllib.request.urlretrieve(video_url, args.out)
    print(f"Done! Saved: {args.out}")

if __name__ == "__main__":
    main()
