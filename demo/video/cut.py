"""Speed up the recording's waits 6x, keep the rest at 1x, and write walkthrough.mp4 plus sections.json (each title
card's start in the cut video), which voice.py uses to place the narration clips.

Usage: python3 cut.py [speed]   (writes walkthrough.mp4 and sections.json beside this file)
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    return float(out)


def segments(info: dict, total: float, offset: float, fast: float) -> list[tuple[float, float, float]]:
    out, at = [], 0.0
    for a, b in info["waits"]:
        a, b = max(a + offset, at), min(b + offset, total)
        if b - a < 1.0:
            continue
        out += [(at, a, 1.0), (a, b, fast)]
        at = b
    out.append((at, total, 1.0))
    return [s for s in out if s[1] - s[0] > 0.05]


def mapped(t: float, segs: list[tuple[float, float, float]]) -> float:
    """A raw-video time's place in the cut video."""
    out = 0.0
    for a, b, speed in segs:
        if t <= a:
            break
        out += (min(t, b) - a) / speed
    return out


def main() -> None:
    fast = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    info = json.loads((HERE / "waits.json").read_text())
    raw = HERE / "raw.webm"
    total = duration(raw)
    # The raw video runs longer than the script's clock at the end (the browser closing), not the start, so the
    # script's times map to the raw video as they are (checked on frames: the title cards land on their starts).
    offset = 0.0
    segs = segments(info, total, offset, fast)
    parts = [f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=(PTS-STARTPTS)/{s}[v{i}]" for i, (a, b, s) in enumerate(segs)]
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[out]"
    out = HERE / "walkthrough.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-filter_complex", graph, "-map", "[out]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-r", "30", str(out)], check=True)  # fmt: skip
    sections = [{"id": s["id"], "title": s["title"], "start": round(mapped(s["at"] + offset, segs), 2)}
                for s in info["scenes"] if s.get("id")]  # fmt: skip
    (HERE / "sections.json").write_text(json.dumps({"duration": round(duration(out), 2), "sections": sections}, indent=1))
    print(f"raw {total:.1f}s -> {out.name} {duration(out):.1f}s ({len(info['waits'])} waits at {fast}x)")
    for s in sections:
        print(f"  {s['start']:7.2f}s  {s['id']}  {s['title']}")


if __name__ == "__main__":
    main()
