"""Contact sheets for the QA pass (the demo-video skill): a frame every N seconds, 12 to a sheet, 3 across.

Usage: python3 sheet.py walkthrough.mp4 [every_seconds]   -> sheet0.png, sheet1.png, ... beside the video
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

W, H, ACROSS, PER = 640, 360, 3, 12


def main() -> None:
    video = Path(sys.argv[1])
    every = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps=1/{every},scale={W}:-1",
                        f"{tmp}/f%03d.png"], check=True)  # fmt: skip
        frames = sorted(Path(tmp).glob("f*.png"))
        for k in range(0, len(frames), PER):
            batch = frames[k : k + PER]
            rows = (len(batch) + ACROSS - 1) // ACROSS
            sheet = Image.new("RGB", (W * ACROSS, H * rows), "white")
            for i, f in enumerate(batch):
                sheet.paste(Image.open(f).resize((W, H)), ((i % ACROSS) * W, (i // ACROSS) * H))
            out = video.with_name(f"sheet{k // PER}.png")
            sheet.save(out)
            print(out, f"{k * every:.0f}s to {(k + len(batch)) * every:.0f}s")


if __name__ == "__main__":
    main()
