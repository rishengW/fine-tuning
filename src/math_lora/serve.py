"""Lightweight inference HTTP server.

Loads a base model (and optional LoRA adapter), exposes ``POST /generate``
for chat-style requests, and ``GET /health`` for liveness checks.

This is intentionally minimal - production deployments should use vLLM or
TGI for batched, paged-attention serving. The point of this module is to
prove the trained adapter is loadable and addressable end-to-end without
depending on heavy infra.

Run::

    python -m math_lora.serve \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/adapter \
        --host 0.0.0.0 --port 8000

Request::

    POST /generate
    {
        "messages": [{"role": "user", "content": "What is 7 * 8?"}],
        "max_new_tokens": 256,
        "temperature": 0.0
    }
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_lora.logging_utils import get_logger

log = get_logger("math_lora.serve")


class _State:
    tokenizer: Any = None
    model: Any = None
    device: str = "cpu"


def _load(base_model: str, adapter: str | None) -> None:
    log.info("loading tokenizer + base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=True
    )
    if adapter:
        from peft import PeftModel

        log.info("attaching adapter: %s", adapter)
        model = PeftModel.from_pretrained(model, adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _State.tokenizer = tokenizer
    _State.model = model
    _State.device = device


def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("`messages` must be a non-empty list")
    max_new_tokens = int(payload.get("max_new_tokens", 256))
    temperature = float(payload.get("temperature", 0.0))

    text = _State.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _State.tokenizer(text, return_tensors="pt").to(_State.device)
    do_sample = temperature > 0.0
    with torch.no_grad():
        out = _State.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=_State.tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    response = _State.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {"response": response}


class _Handler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server interface
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
        else:
            self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server interface
        if self.path != "/generate":
            self._write_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._write_json(400, {"error": f"bad json: {exc}"})
            return
        try:
            result = _generate(payload)
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("generation failed")
            self._write_json(500, {"error": str(exc)})
            return
        self._write_json(200, result)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, N802
        log.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    p = argparse.ArgumentParser(description="math-lora inference server")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    _load(args.base_model, args.adapter)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    log.info("serving on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
