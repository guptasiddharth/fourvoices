"""Frame sampling from a clip via ffmpeg — evenly across the FULL duration.

Key property: for a clip of any length (the hackathon allows up to 2 min), the
sampled frames span the entire timeline, so the caption reflects the whole clip
— not just its opening seconds. Degrades gracefully: if ffmpeg or the clip is
missing, returns [] and the pipeline falls back to the stub description.

Note: this is a visual sampler — it captures what is *seen*. Dialogue/audio is
not transcribed (see roadmap: optional Whisper pass for talky clips).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


def duration_sec(clip_path: str) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", clip_path],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return None


def sample_frames(clip_path: str, n: int = 8) -> list[str]:
    """Extract ~n JPEG frames spread evenly across the clip's full duration.
    Returns a list of file paths (in a temp dir the caller may clean up)."""
    if not clip_path or not os.path.exists(clip_path) or not shutil.which("ffmpeg"):
        return []
    n = max(1, n)
    out_dir = tempfile.mkdtemp(prefix="vc_frames_")
    pattern = os.path.join(out_dir, "f_%03d.jpg")

    dur = duration_sec(clip_path)
    if dur and dur > 0:
        # fps = n / duration → one frame every (duration/n) seconds, across the
        # WHOLE clip. With -frames:v n as a safety cap, the first n frames are
        # already time-spread (unlike fps=1, which would grab the first n seconds).
        fps = max(n / dur, 0.02)
        vf = f"fps={fps:.5f},scale=768:-2"
    else:
        vf = "fps=1,scale=768:-2"   # duration unknown → 1 fps, capped below

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", clip_path,
         "-vf", vf, "-vsync", "vfr", "-q:v", "2", "-frames:v", str(n), pattern],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                    if f.endswith(".jpg"))
    return frames[:n]
