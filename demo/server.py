"""Local demo: the untrained model, v1 and the public v2 adapter label the same chat lines side by side.

Usage (training environment, from the repo root):
  .venv-train/Scripts/python demo/server.py            # then open http://127.0.0.1:8780

Loads Qwen3-1.7B once and attaches two LoRA adapters: v2 from the public Hugging Face checkpoint
at its pinned commit, and v1 from models/ if it is there. The untrained answer comes from the same
weights with the adapters switched off. Prompt, chat template and greedy decoding are the ones
eval.py uses. On a GPU the weights load in bf16, as in eval.py; on a CPU (the hosted Space) in
--cpu-dtype, so a borderline line can get a different answer there (see the README).
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from labels import parse_label  # noqa: E402
from pins import BASE, BASE_REVISION  # noqa: E402
from pins import TUNED as V2_REPO  # noqa: E402
from pins import TUNED_REVISION as V2_REVISION  # noqa: E402
from slm import label  # noqa: E402

V1_DIR = ROOT / "models" / "qwen3-1.7b-v1-n10919"
RESULT_ROWS = [  # (run name, label on the page)
    ("qwen3-1.7b-base__slm__na", "Qwen3-1.7B, untrained"),
    ("qwen3-1.7b-v1-n10919__slm__na", "v1: 10,919 expert-labelled rows"),
    ("qwen3-1.7b-v2__slm__na", "v2: plus 676 teacher rows (public)"),
    ("claude-opus-5-5__zero_shot_p1__low", "Claude Opus 5.5, best prompt"),
    ("gpt-5.6-sol__few_shot__low", "GPT-5.6 Sol, best prompt"),
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


class Demo:
    def __init__(self, cpu_dtype: str = "float32"):
        self.device = "GPU" if torch.cuda.is_available() else "CPU"
        self.tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
        self.tokenizer.padding_side = "left"
        if self.device == "GPU":
            base = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=torch.bfloat16, device_map={"": 0})
        else:
            base = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=getattr(torch, cpu_dtype))
        self.model = PeftModel.from_pretrained(base, V2_REPO, revision=V2_REVISION, adapter_name="v2")
        self.models = ["untrained", "v2"]
        if (V1_DIR / "adapter_config.json").exists():
            self.model.load_adapter(str(V1_DIR), adapter_name="v1")
            self.models = ["untrained", "v1", "v2"]
        self.model.eval()
        self.lock = threading.Lock()
        self.sets = self._sets()

    def label(self, lines: list[str]) -> dict:
        out = {}
        with self.lock:
            start = time.perf_counter()
            for name in self.models:
                if name == "untrained":
                    with self.model.disable_adapter():
                        raws = label(self.tokenizer, self.model, lines)
                else:
                    self.model.set_adapter(name)
                    raws = label(self.tokenizer, self.model, lines)
                out[name] = [{"raw": r, "label": {1: "toxic", 0: "nontoxic"}.get(parse_label(r))} for r in raws]
            ms = (time.perf_counter() - start) * 1000
        return {"models": self.models, "answers": out, "device": self.device,
                "ms_per_line_per_model": round(ms / len(lines) / len(self.models), 1)}

    def _sets(self) -> list[dict]:
        cfg = json.loads((ROOT / "demo" / "examples.json").read_text(encoding="utf-8"))
        sets = []
        for s in cfg["sets"]:
            rows = {r["id"]: r for r in read_jsonl(ROOT / s["source"])}
            eval_set = Path(s["source"]).stem
            claude = {r["id"]: r for r in read_jsonl(ROOT / "runs" / f"claude-opus-5-5__zero_shot_p1__low__{eval_set}.jsonl")}
            sets.append({
                **{k: s[k] for k in ("key", "title", "source", "label_by", "note")},
                "lines": [{"id": i, "message": rows[i]["message"], "gold": "toxic" if rows[i]["label"] else "nontoxic",
                           "claude": {1: "toxic", 0: "nontoxic"}.get(claude[i]["pred"])} for i in s["ids"]],
            })
        return sets

    def results(self) -> dict:
        rows = []
        for run, name in RESULT_ROWS:
            spec = json.loads((ROOT / "runs" / f"{run}__E1000.meta.json").read_text())
            robust = json.loads((ROOT / "runs" / f"{run}__PRESSURE.meta.json").read_text())
            c = robust["confusion"]
            rows.append({"name": name, **{k: spec["scores"][k] for k in ("precision", "recall", "f1")},
                         "pressure": c["tp"] + c["tn"], "pressure_n": robust["scores"]["n"]})
        base = {r["id"]: r for r in read_jsonl(ROOT / "runs" / "qwen3-1.7b-base__slm__na__E1000.jsonl")}
        v2 = {r["id"]: r for r in read_jsonl(ROOT / "runs" / "qwen3-1.7b-v2__slm__na__E1000.jsonl")}
        flips = {
            "untrained_wrong_v2_right": sum(base[i]["pred"] != base[i]["gold"] and v2[i]["pred"] == v2[i]["gold"] for i in base),
            "untrained_right_v2_wrong": sum(base[i]["pred"] == base[i]["gold"] and v2[i]["pred"] != v2[i]["gold"] for i in base),
        }
        return {"rows": rows, "bar": {"precision": 0.80, "f1": 0.75}, "flips": flips,
                "checkpoint": {"repo": V2_REPO, "revision": V2_REVISION, "base": BASE, "base_revision": BASE_REVISION}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve beyond this machine (the Space does)")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--cpu-dtype", default="float32", choices=["float32", "bfloat16"], help="weights on a CPU-only machine")
    args = ap.parse_args()
    demo = Demo(args.cpu_dtype)
    info = {"models": demo.models, "device": demo.device, "sets": demo.sets, "results": demo.results()}
    files = {"/": ("demo/index.html", "text/html; charset=utf-8"),
             "/data_efficiency.svg": ("results/data_efficiency.svg", "image/svg+xml")}

    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, kind: str):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/info":
                return self.send(200, json.dumps(info).encode(), "application/json")
            if self.path in files:
                path, kind = files[self.path]
                return self.send(200, (ROOT / path).read_bytes(), kind)
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/label":
                return self.send(404, b"not found", "text/plain")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            lines = [str(l).strip()[:300] for l in body.get("lines", []) if str(l).strip()][:20]
            if not lines:
                return self.send(400, b'{"error": "no lines"}', "application/json")
            self.send(200, json.dumps(demo.label(lines)).encode(), "application/json")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"demo ready: http://{args.host}:{args.port}  models: {', '.join(demo.models)}  device: {demo.device}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
