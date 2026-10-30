"""The narration studio (the demo-video skill): a local page that walks through the script one section at a time, plays
that section of the silent cut while you read, records your microphone, and builds the voiced video with voice.py.

    python3 studio.py --video walkthrough.mp4 --sections sections.json --script walkthrough-script.md [--clips clips]
    then open http://127.0.0.1:8770

The script's sections are its "## NN. Title …" headings, each followed by the words to read. Takes are saved as
<clips>/NN.<ext> (webm in Chrome and Firefox, m4a or mp4 in Safari); "Build" runs voice.py on them and serves
walkthrough-voiced.mp4. Standard library only; ffmpeg is needed only to build. It listens on 127.0.0.1, never wider.
"""

import argparse
import json
import mimetypes
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADING = re.compile(r"^## (\d\d)\. (.+)$", re.MULTILINE)
EXTENSIONS = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/mpeg": "mp3", "audio/wav": "wav",
              "video/webm": "webm", "video/mp4": "m4a"}  # fmt: skip
WORDS_PER_SECOND = 2.5


def parse_script(text: str) -> dict[str, dict]:
    """{id: {heading, text, words}} from the "## NN. …" sections; a heading's "(title card: …)" note is dropped."""
    found = {}
    marks = list(HEADING.finditer(text))
    for i, m in enumerate(marks):
        body = text[m.end() : marks[i + 1].start() if i + 1 < len(marks) else len(text)].strip()
        heading = re.split(r" \(title card:", m.group(2))[0].strip()
        found[m.group(1)] = {"heading": heading, "text": body, "words": len(body.split())}
    return found


def clip_path(clips: Path, section_id: str) -> Path | None:
    found = sorted(p for p in clips.glob(f"{section_id}.*") if p.is_file())
    return found[0] if found else None


class Studio:
    def __init__(self, args):
        self.video = Path(args.video).resolve()
        self.sections_file = Path(args.sections).resolve()
        self.script_file = Path(args.script).resolve()
        self.clips = Path(args.clips).resolve()
        self.clips.mkdir(parents=True, exist_ok=True)
        self.voiced = self.video.with_name("walkthrough-voiced.mp4")
        self.voice_py = Path(args.voice or HERE / "voice.py").resolve()

    def data(self) -> dict:
        info = json.loads(self.sections_file.read_text())
        script = parse_script(self.script_file.read_text(encoding="utf-8"))
        sections = info["sections"]
        out = []
        for i, s in enumerate(sections):
            if s["id"] not in script:
                continue  # a title card with no narration (the closing card)
            end = sections[i + 1]["start"] if i + 1 < len(sections) else info["duration"]
            clip = clip_path(self.clips, s["id"])
            part = script[s["id"]]
            out.append({
                "id": s["id"], "card": s["title"], "heading": part["heading"], "text": part["text"],
                "words": part["words"], "target": round(part["words"] / WORDS_PER_SECOND, 1),
                "start": s["start"], "end": end, "clip": f"/clips/{clip.name}" if clip else None,
            })  # fmt: skip
        return {"sections": out, "duration": info["duration"], "voiced": self.voiced.exists()}


def handler_for(studio: Studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet: one line per write, nothing per GET
            if self.command != "GET":
                sys.stderr.write("studio: " + fmt % args + "\n")

        def send_json(self, body, status=200):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def send_file(self, path: Path, kind: str | None = None):
            """A file, with byte ranges, so the browser can seek video and audio."""
            if not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            kind = kind or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            ranged = self.headers.get("Range", "")
            m = re.match(r"bytes=(\d*)-(\d*)$", ranged)
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.send_file(HERE / "studio.html", "text/html; charset=utf-8")
            elif path == "/data.json":
                self.send_json(studio.data())
            elif path == "/video":
                self.send_file(studio.video, "video/mp4")
            elif path == "/voiced":
                self.send_file(studio.voiced, "video/mp4")
            elif m := re.fullmatch(r"/clips/(\d\d\.[a-z0-9]+)", path):
                self.send_file(studio.clips / m.group(1))
            else:
                self.send_error(404)

        def do_PUT(self):
            m = re.fullmatch(r"/clips/(\d\d)", self.path)
            if not m:
                self.send_error(404)
                return
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            ext = EXTENSIONS.get(kind)
            length = int(self.headers.get("Content-Length") or 0)
            if not ext or not 0 < length < 200 * 1024 * 1024:
                self.send_json({"error": f"unsupported recording ({kind or 'no type'}, {length} bytes)"}, 400)
                return
            for old in studio.clips.glob(f"{m.group(1)}.*"):  # one take per section: the new one replaces the old
                old.unlink()
            target = studio.clips / f"{m.group(1)}.{ext}"
            target.write_bytes(self.rfile.read(length))
            self.send_json({"clip": f"/clips/{target.name}"})

        def do_DELETE(self):
            m = re.fullmatch(r"/clips/(\d\d)", self.path)
            if not m:
                self.send_error(404)
                return
            for old in studio.clips.glob(f"{m.group(1)}.*"):
                old.unlink()
            self.send_json({"clip": None})

        def do_POST(self):
            if self.path != "/build":
                self.send_error(404)
                return
            cmd = [sys.executable, str(studio.voice_py), str(studio.clips), str(studio.video), str(studio.sections_file)]
            try:
                run = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=900)
            except FileNotFoundError as e:
                self.send_json({"ok": False, "log": f"could not run voice.py: {e}"}, 500)
                return
            except subprocess.TimeoutExpired:
                self.send_json({"ok": False, "log": "voice.py took longer than 15 minutes"}, 500)
                return
            log = (run.stdout + run.stderr).strip()
            if run.returncode != 0 and "ffmpeg" in log.lower() and "not found" in log.lower():
                log += "\n\nffmpeg isn't installed or isn't on PATH. Install it (Windows: winget install ffmpeg; macOS: brew install ffmpeg), restart the terminal, and build again."
            self.send_json({"ok": run.returncode == 0, "log": log}, 200 if run.returncode == 0 else 500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="walkthrough.mp4", help="the silent cut from cut.py")
    parser.add_argument("--sections", default="sections.json", help="cut.py's section starts")
    parser.add_argument("--script", default="walkthrough-script.md", help="the narration script")
    parser.add_argument("--clips", default="clips", help="where takes are saved (NN.webm, NN.m4a, …)")
    parser.add_argument("--voice", default=None, help="voice.py (default: the one beside this file)")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    studio = Studio(args)
    for p in (studio.video, studio.sections_file, studio.script_file):
        if not p.is_file():
            sys.exit(f"studio: missing {p}")
    count = len(studio.data()["sections"])
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(studio))
    print(f"studio: {count} sections, takes in {studio.clips}\nstudio: open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
