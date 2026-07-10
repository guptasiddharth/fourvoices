"""Frame sampling from a clip via ffmpeg — uniform + scene-change frames.

Degrades gracefully: if ffmpeg or the clip is missing, returns [] and the
pipeline falls back to the stub description. Real runs on AMD Developer Cloud
have ffmpeg available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


def sample_frames(clip_path: str, n: int = 6) -> list[str]:
    """Extract ~n representative JPEG frames; prefer scene changes, fill uniformly.
    Returns a list of file paths (in a temp dir the caller may clean up)."""
    if not clip_path or not os.path.exists(clip_path) or not shutil.which("ffmpeg"):
        return []
    out_dir = tempfile.mkdtemp(prefix="vc_frames_")
    pattern = os.path.join(out_dir, "f_%03d.jpg")
    # scene-change frames (thresh 0.3); cap with -frames:v n
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", clip_path,
           "-vf", "select='gt(scene,0.3)',scale=768:-1", "-vsync", "vfr",
           "-frames:v", str(n), pattern]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir))
    if len(frames) >= 2:
        return frames[:n]
    # fallback: uniform sampling by fps
    for f in frames:
        os.remove(f)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", clip_path,
           "-vf", f"fps=1,scale=768:-1", "-frames:v", str(n), pattern]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir))[:n]
