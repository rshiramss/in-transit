#!/usr/bin/env python3
"""
nanobanana: create an in-between image and generate two fal.ai transition videos.

Usage:
  python nanobanana_transition.py \
    --first first.jpg \
    --second second.jpg \
    --out-mid nanobanana.jpg \
    --out-a first_to_nanobanana.mp4 \
    --out-b nanobanana_to_second.mp4 \
    --size 1280x720 --align \
    --prompt-a "Subtle reveal toward the midframe" \
    --prompt-b "Continue reveal into the target frame"

Notes:
  - If --align is provided and OpenCV is installed, the second image is homography-aligned to the first before blending.
  - If OpenCV isn't installed or alignment fails, a safe letterbox resize + center blend is used.
  - The script saves THREE files locally: the middle frame (jpg/png), and two .mp4 videos.
"""

import os
import re
import sys
import argparse
import mimetypes
from typing import Tuple, Optional

import numpy as np
from PIL import Image

# fal.ai client
import fal_client

try:
    import cv2  # Optional, for alignment
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ---------- utilities ----------
def parse_size(s: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", s)
    if not m:
        raise argparse.ArgumentTypeError("Size must be like 1280x720")
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Width/height must be > 0")
    return w, h

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
        return None
    a = np.array(imgA.convert("RGB"))
    b = np.array(imgB.convert("RGB"))
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(3000)
    kpa, desca = orb.detectAndCompute(a_gray, None)
    kpb, descb = orb.detectAndCompute(b_gray, None)
    if desca is None or descb is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desca, descb, k=2)

    # Lowe's ratio test
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75*m[1].distance:
            good.append(m[0])
    if len(good) < 12:
        return None

    src_pts = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    h, w = a.shape[:2]
    warped = cv2.warpPerspective(b, H, (w, h), flags=cv2.INTER_LINEAR)
    warped_pil = Image.fromarray(warped)
    return warped_pil

def cross_dissolve(imA: Image.Image, imB: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Simple midpoint blend A*(1-alpha) + B*alpha, expects same size, RGB."""
    A = np.asarray(imA.convert("RGB"), dtype=np.float32)
    B = np.asarray(imB.convert("RGB"), dtype=np.float32)
    C = (1.0 - alpha) * A + alpha * B
    C = np.clip(C, 0, 255).astype(np.uint8)
    return Image.fromarray(C, mode="RGB")

def ensure_url(path_or_url: str) -> str:
    """Upload local file to fal if needed, else pass through URL."""
    from urllib.parse import urlparse
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        return path_or_url
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(path_or_url)
    ctype, _ = mimetypes.guess_type(path_or_url)
    return fal_client.upload_file(path_or_url, content_type=ctype or "application/octet-stream")

def download_to(url: str, out_path: str):
    import urllib.request
    urllib.request.urlretrieve(url, out_path)


# ---------- fal wrappers ----------
def run_fal_video(start_url: str, end_url: str, out_mp4: str, *, prompt: str,
                  frames: int, fps: int, resolution: str, steps: int,
                  cfg: float, shift: float, seed: Optional[int], acceleration: str,
                  aspect: str):
    args = {
        "prompt": prompt,
        "start_image_url": start_url,
        "end_image_url": end_url,
        "num_frames": frames,
        "frames_per_second": fps,
        "resolution": resolution,                 # "480p" or "720p"
        "num_inference_steps": steps,
        "guide_scale": cfg,
        "shift": shift,
        "enable_safety_checker": True,
        "enable_prompt_expansion": False,
        "acceleration": acceleration,             # "regular" (default) or "none"
        "aspect_ratio": aspect,                   # "auto", "16:9", "9:16", "1:1"
    }
    if seed is not None:
        args["seed"] = seed

    result = fal_client.subscribe(
        "fal-ai/wan-flf2v",
        arguments=args,
        with_logs=True,
    )
    video_url = (result.get("video") or {}).get("url")
    if not video_url:
        raise RuntimeError(f"No video URL in fal response: {result}")
    download_to(video_url, out_mp4)


# ---------- main ----------
def main():
    p = argparse.ArgumentParser("nanobanana — make a middle frame and two fal.ai transitions")
    p.add_argument("--first", required=True, help="Path/URL to first image")
    p.add_argument("--second", required=True, help="Path/URL to second image")
    p.add_argument("--size", type=parse_size, default=(1280, 720), help="Working frame size WxH (default 1280x720)")
    p.add_argument("--align", action="store_true", help="Try ORB homography alignment (needs opencv-python)")

    p.add_argument("--out-mid", default="nanobanana.jpg", help="Output path for the middle frame")
    p.add_argument("--out-a", default="first_to_nanobanana.mp4", help="Output MP4 for first→nanobanana")
    p.add_argument("--out-b", default="nanobanana_to_second.mp4", help="Output MP4 for nanobanana→second")

    # fal generation controls
    p.add_argument("--prompt-a", default="Smooth, coherent motion from the first frame into the midframe.")
    p.add_argument("--prompt-b", default="Smooth, coherent motion from the midframe into the second frame.")
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--resolution", choices=["480p", "720p"], default="720p")
    p.add_argument("--aspect", choices=["auto", "16:9", "9:16", "1:1"], default="auto")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--acceleration", choices=["none", "regular"], default="regular")

    args = p.parse_args()

    if not os.environ.get("FAL_KEY"):
        print("ERROR: FAL_KEY environment variable not set.", file=sys.stderr)
        sys.exit(2)

    # Load and fit both images to a common working size
    if re.match(r"^https?://", args.first):
        im1 = Image.open(Image.open.__self__.fp if False else args.first)  # noqa: trick linter
    try:
        imA_raw = Image.open(args.first)
    except Exception:
        # allow URL via PIL's built-in opener
        from urllib.request import urlopen
        imA_raw = Image.open(urlopen(args.first))
    try:
        imB_raw = Image.open(args.second)
    except Exception:
        from urllib.request import urlopen
        imB_raw = Image.open(urlopen(args.second))

    A = letterbox_fit(imA_raw, args.size)
    B = letterbox_fit(imB_raw, args.size)

    # Optional alignment: align B to A before blending
    if args.align:
        alignedB = try_align_orb(A, B)
        if alignedB is not None:
            B = alignedB
        else:
            print("[nanobanana] Alignment skipped (cv2 missing or matching failed).", file=sys.stderr)

    # Build the middle frame (simple 50/50 cross-dissolve)
    mid = cross_dissolve(A, B, alpha=0.5)
    mid.save(args.out_mid)
    print(f"[nanobanana] Saved middle frame: {args.out_mid}")

    # Prepare URLs for fal (uploads if local)
    first_url = ensure_url(args.first)
    mid_url   = ensure_url(args.out_mid)
    second_url = ensure_url(args.second)

    # Generate two transition videos
    print("[nanobanana] Generating first → nanobanana …")
    run_fal_video(
        start_url=first_url, end_url=mid_url, out_mp4=args.out_a,
        prompt=args.prompt_a, frames=args.frames, fps=args.fps,
        resolution=args.resolution, steps=args.steps, cfg=args.cfg,
        shift=args.shift, seed=args.seed, acceleration=args.acceleration,
        aspect=args.aspect
    )
    print(f"[nanobanana] Saved: {args.out_a}")

    print("[nanobanana] Generating nanobanana → second …")
    run_fal_video(
        start_url=mid_url, end_url=second_url, out_mp4=args.out_b,
        prompt=args.prompt_b, frames=args.frames, fps=args.fps,
        resolution=args.resolution, steps=args.steps, cfg=args.cfg,
        shift=args.shift, seed=args.seed, acceleration=args.acceleration,
        aspect=args.aspect
    )
    print(f"[nanobanana] Saved: {args.out_b}")

    print("[nanobanana] Done.")

if __name__ == "__main__":
    main()
