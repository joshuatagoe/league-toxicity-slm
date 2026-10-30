"""Lay narration clips over the silent walkthrough: clip NN starts on section NN's title card (sections.json, from
cut.py). A clip longer than its section holds the section's last frame until it ends; a missing clip leaves that
section silent. The voice is normalized to a steady loudness. Writes walkthrough-voiced.mp4.

Usage: python3 voice.py <folder with 00.m4a, 01.m4a, ...> [video] [sections.json]
Any audio format ffmpeg reads works (m4a, mp3, wav, webm, ogg).
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
TAIL = 0.8  # seconds of quiet after a clip before the next section starts


def duration(path: Path) -> float:
    """Seconds. A browser recording (MediaRecorder webm) has no duration in its header, so ffprobe says N/A; then
    decode it to the null muxer and read the last timestamp ffmpeg reports."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    try:
        return float(out)
    except ValueError:
        log = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-f", "null", "-"], check=False, capture_output=True, text=True).stderr
        stamps = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", log)
        if not stamps:
            raise
        h, m, sec = stamps[-1]
        return int(h) * 3600 + int(m) * 60 + float(sec)


def clip_for(folder: Path, section_id: str) -> Path | None:
    found = sorted(p for p in folder.glob(f"{section_id}.*") if p.is_file())
    return found[0] if found else None


def main() -> None:
    folder = Path(sys.argv[1])
    video = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "walkthrough.mp4"
    info = json.loads((Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "sections.json").read_text())
    total = duration(video)
    starts = [s["start"] for s in info["sections"]] + [total]
    with tempfile.TemporaryDirectory() as tmp:
        pieces = []
        lead = starts[0]
        if lead > 0.05:  # anything before the first title card, silent
            pieces.append(("lead", 0.0, lead, None))
        for i, s in enumerate(info["sections"]):
            pieces.append((s["id"], starts[i], starts[i + 1], clip_for(folder, s["id"])))
        files, report = [], []
        for n, (sid, a, b, clip) in enumerate(pieces):
            video_len = b - a
            spoken = duration(clip) if clip else 0.0
            length = max(video_len, spoken + TAIL) if clip else video_len
            hold = max(0.0, length - video_len)
            out = Path(tmp) / f"p{n:02d}.mp4"
            vf = f"trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS" + (f",tpad=stop_mode=clone:stop_duration={hold:.3f}" if hold else "")
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video)]
            if clip:
                cmd += ["-i", str(clip), "-filter_complex",
                        f"[0:v]{vf}[v];[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,apad,atrim=0:{length:.3f}[a]"]  # fmt: skip
            else:
                cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-filter_complex",
                        f"[0:v]{vf}[v];[1:a]atrim=0:{length:.3f}[a]"]  # fmt: skip
            cmd += ["-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-r", "30",
                    "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-t", f"{length:.3f}", str(out)]  # fmt: skip
            subprocess.run(cmd, check=True)
            files.append(out)
            if sid != "lead":
                note = f"held {hold:.1f}s" if hold else (f"{video_len - spoken:.1f}s quiet" if clip else "no clip, silent")
                report.append(f"  {sid}: scene {video_len:5.1f}s, voice {spoken:5.1f}s -> {length:5.1f}s ({note})")
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{f}'\n" for f in files))
        target = video.with_name("walkthrough-voiced.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                        str(target)], check=True)  # fmt: skip
    print("\n".join(report))
    print(f"{target}: {duration(target):.1f}s, {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
