"""GPU Mode Switch — Production (Qwen) ↔ Training (Unsloth) via olares-cli."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("modeswitch")

OLARES_CLI = os.environ.get("OLARES_CLI", "olares-cli")
DATA_DIR = Path(os.environ.get("MODE_DATA_DIR", "/app/data"))
STATE_FILE = DATA_DIR / "state.json"

PROD_APP = os.environ.get("PROD_APP", "qwen3627b").strip()
TRAIN_APP = os.environ.get("TRAIN_APP", "unsloth").strip()
GPU_NODE = os.environ.get("GPU_NODE", "").strip()
GPU_DEVICE = os.environ.get("GPU_DEVICE", "").strip()
PROD_VRAM_GI = os.environ.get("PROD_VRAM_GI", "23").strip()
TRAIN_VRAM_GI = os.environ.get("TRAIN_VRAM_GI", "22").strip()
WATCH_TIMEOUT = os.environ.get("WATCH_TIMEOUT", "15m").strip()

# Comma-separated Market app names that must be Running when the mode settles.
# EmbeddingGemma / Router / kbmcp: resume WITHOUT compute-binding (never rebind Intel GPU).
DEFAULT_PROD_ENSURE = f"{PROD_APP},router,embeddinggemma,kbmcp"
DEFAULT_TRAIN_ENSURE = f"{TRAIN_APP},router,embeddinggemma,kbmcp"


def _parse_list(raw: str, default: str) -> list[str]:
    text = (raw or "").strip() or default
    seen: list[str] = []
    for part in text.split(","):
        name = part.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


PRODUCTION_ENSURE_RUNNING = _parse_list(
    os.environ.get("PRODUCTION_ENSURE_RUNNING", ""), DEFAULT_PROD_ENSURE
)
TRAINING_ENSURE_RUNNING = _parse_list(
    os.environ.get("TRAINING_ENSURE_RUNNING", ""), DEFAULT_TRAIN_ENSURE
)

# Apps that use the NVIDIA card via MemorySlice and need --compute-binding on resume.
GPU_BOUND_APPS = {PROD_APP, TRAIN_APP}

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="GPU Mode Switch", version="0.1.0")
_lock = asyncio.Lock()
_busy = False
_last_error = ""
_last_log: list[str] = []


class SwitchRequest(BaseModel):
    mode: str = Field(..., pattern="^(production|training)$")


def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"mode": "unknown", "updated_at": None}


def _save_state(mode: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(
            {"mode": mode, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
            indent=2,
        )
    )


def _is_running(state: str) -> bool:
    s = (state or "").lower()
    return s in ("running", "resumed") or s.endswith("running")


def _is_stopped(state: str) -> bool:
    s = (state or "").lower()
    return s in ("stopped", "suspended", "stop") or "stopp" in s or "suspend" in s


async def run_cli(*args: str, timeout: float = 900.0) -> tuple[int, str, str]:
    cmd = [OLARES_CLI, *args]
    logger.info("cli: %s", " ".join(cmd))
    _last_log.append("$ " + " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timeout after {timeout}s: {' '.join(cmd)}"
    out = (out_b or b"").decode(errors="replace")
    err = (err_b or b"").decode(errors="replace")
    if out.strip():
        _last_log.append(out.strip()[:2000])
    if err.strip():
        _last_log.append(err.strip()[:2000])
    del _last_log[:-40]
    return proc.returncode or 0, out, err


async def cli_json(*args: str) -> Any:
    code, out, err = await run_cli(*args, "-o", "json", timeout=120.0)
    if code != 0:
        raise RuntimeError(
            f"olares-cli {' '.join(args)} failed ({code}): {(err or out)[:500]}"
        )
    text = out.strip() or "null"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise RuntimeError(f"non-JSON from olares-cli {' '.join(args)}: {text[:300]}")


def binding_flag(vram_gi: str) -> list[str]:
    if not GPU_NODE or not GPU_DEVICE:
        raise RuntimeError(
            "GPU_NODE and GPU_DEVICE must be set (Settings → Accelerator / "
            "olares-cli settings compute list). Edit this app's envs after install."
        )
    return ["--compute-binding", f"{GPU_NODE}:{GPU_DEVICE}:{vram_gi}"]


def _normalize_status(name: str, data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        state = (
            data.get("state")
            or data.get("status")
            or data.get("finalState")
            or (data.get("data") or {}).get("state")
            or "unknown"
        )
        source = data.get("source") or (data.get("data") or {}).get("source")
        return {"name": name, "state": str(state), "source": source}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        row = data[0]
        return {
            "name": name,
            "state": str(row.get("state") or row.get("status") or "unknown"),
            "source": row.get("source"),
        }
    return {"name": name, "state": "unknown", "raw": str(data)[:200]}


async def app_state(name: str) -> dict[str, Any]:
    try:
        data = await cli_json("market", "status", name)
        return _normalize_status(name, data)
    except Exception as e:
        return {"name": name, "state": "unknown", "error": str(e)[:240]}


async def stop_app(name: str) -> None:
    code, out, err = await run_cli(
        "market", "stop", name, "--watch", f"--watch-timeout={WATCH_TIMEOUT}"
    )
    if code != 0:
        # Idempotent: already stopped is success for our purposes
        blob = (err or out).lower()
        if "already" in blob and "stop" in blob:
            return
        st = await app_state(name)
        if _is_stopped(st.get("state", "")):
            return
        raise RuntimeError(f"stop {name} failed ({code}): {(err or out)[:800]}")


async def resume_app(name: str, *, with_gpu: bool, vram_gi: str | None = None) -> None:
    args = ["market", "resume", name]
    if with_gpu:
        args += binding_flag(vram_gi or PROD_VRAM_GI)
    # Non-GPU apps (Router, EmbeddingGemma, kbmcp): plain resume — never touch binding
    args += ["--watch", f"--watch-timeout={WATCH_TIMEOUT}"]
    code, out, err = await run_cli(*args)
    if code != 0:
        blob = (err or out).lower()
        if "already" in blob and "run" in blob:
            return
        st = await app_state(name)
        if _is_running(st.get("state", "")):
            return
        raise RuntimeError(f"resume {name} failed ({code}): {(err or out)[:800]}")


async def ensure_running(names: list[str], *, gpu_primary: str, vram_gi: str) -> list[dict[str, Any]]:
    """Resume any stopped apps in the ensure list. GPU binding only for gpu_primary."""
    results: list[dict[str, Any]] = []
    for name in names:
        st = await app_state(name)
        state = st.get("state", "unknown")
        if _is_running(state):
            results.append({**st, "action": "already_running"})
            continue
        with_gpu = name == gpu_primary and name in GPU_BOUND_APPS
        logger.info("ensure_running: resuming %s (gpu_binding=%s)", name, with_gpu)
        await resume_app(name, with_gpu=with_gpu, vram_gi=vram_gi if with_gpu else None)
        st2 = await app_state(name)
        if not _is_running(st2.get("state", "")):
            raise RuntimeError(
                f"{name} did not reach running (state={st2.get('state')})"
            )
        results.append({**st2, "action": "resumed"})
    return results


async def wait_all_running(names: list[str], attempts: int = 30, delay: float = 5.0) -> list[dict[str, Any]]:
    last: list[dict[str, Any]] = []
    for _ in range(attempts):
        last = [await app_state(n) for n in names]
        if all(_is_running(x.get("state", "")) for x in last):
            return last
        await asyncio.sleep(delay)
    bad = [f"{x['name']}={x.get('state')}" for x in last if not _is_running(x.get("state", ""))]
    raise RuntimeError(f"timed out waiting for Running: {', '.join(bad)}")


async def switch_to(mode: str) -> dict[str, Any]:
    global _last_error
    t0 = time.monotonic()
    if mode == "production":
        # Stop Unsloth (frees GPU), resume Qwen with 23Gi, then ensure companions
        await stop_app(TRAIN_APP)
        await resume_app(PROD_APP, with_gpu=True, vram_gi=PROD_VRAM_GI)
        ensure_list = PRODUCTION_ENSURE_RUNNING
        gpu_primary = PROD_APP
        vram = PROD_VRAM_GI
        note = "KB ask online (Qwen on GPU)."
    else:
        await stop_app(PROD_APP)
        await resume_app(TRAIN_APP, with_gpu=True, vram_gi=TRAIN_VRAM_GI)
        ensure_list = TRAINING_ENSURE_RUNNING
        gpu_primary = TRAIN_APP
        vram = TRAIN_VRAM_GI
        note = "KB ask offline while Qwen is stopped (Training)."

    ensured = await ensure_running(ensure_list, gpu_primary=gpu_primary, vram_gi=vram)
    final = await wait_all_running(ensure_list)
    _save_state(mode)
    _last_error = ""
    return {
        "ok": True,
        "mode": mode,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "note": note,
        "ensured": ensured,
        "apps": final,
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/api/config")
async def api_config():
    return {
        "prod_app": PROD_APP,
        "train_app": TRAIN_APP,
        "gpu_node": GPU_NODE or None,
        "gpu_device": GPU_DEVICE or None,
        "prod_vram_gi": PROD_VRAM_GI,
        "train_vram_gi": TRAIN_VRAM_GI,
        "production_ensure_running": PRODUCTION_ENSURE_RUNNING,
        "training_ensure_running": TRAINING_ENSURE_RUNNING,
        "binding_configured": bool(GPU_NODE and GPU_DEVICE),
        "kb_ask_offline_in_training": True,
    }


@app.get("/api/status")
async def api_status():
    state = _load_state()
    prod = await app_state(PROD_APP)
    train = await app_state(TRAIN_APP)

    # Status for all unique ensure-list members
    names: list[str] = []
    for n in PRODUCTION_ENSURE_RUNNING + TRAINING_ENSURE_RUNNING + [PROD_APP, TRAIN_APP]:
        if n not in names:
            names.append(n)
    companions = {n: await app_state(n) for n in names}

    inferred = state.get("mode") or "unknown"
    ps = str(prod.get("state", "")).lower()
    ts = str(train.get("state", "")).lower()
    if _is_running(ps) and _is_stopped(ts):
        inferred = "production"
    elif _is_running(ts) and _is_stopped(ps):
        inferred = "training"

    ensure_now = (
        PRODUCTION_ENSURE_RUNNING if inferred == "production" else TRAINING_ENSURE_RUNNING
        if inferred == "training"
        else list(dict.fromkeys(PRODUCTION_ENSURE_RUNNING + TRAINING_ENSURE_RUNNING))
    )
    ensure_states = [companions[n] for n in ensure_now if n in companions]
    all_ok = bool(ensure_states) and all(
        _is_running(x.get("state", "")) for x in ensure_states
    )

    binding = None
    if GPU_NODE and GPU_DEVICE:
        binding = {
            "node": GPU_NODE,
            "device": GPU_DEVICE,
            "prod_gi": PROD_VRAM_GI,
            "train_gi": TRAIN_VRAM_GI,
        }

    return {
        "mode": inferred,
        "saved_mode": state.get("mode"),
        "updated_at": state.get("updated_at"),
        "busy": _busy,
        "last_error": _last_error,
        "prod": prod,
        "train": train,
        "ensure_list": ensure_now,
        "ensure_apps": ensure_states,
        "ensure_all_running": all_ok,
        "apps": companions,
        "gpu_binding": binding,
        "kb_ask_note": (
            "KB ask is offline in Training (Qwen suspended). "
            "search/embed via EmbeddingGemma still works."
            if inferred == "training"
            else "KB ask available when Qwen is running (Production)."
        ),
        "config": {
            "prod_app": PROD_APP,
            "train_app": TRAIN_APP,
            "gpu_node": GPU_NODE or None,
            "gpu_device": GPU_DEVICE or None,
            "production_ensure_running": PRODUCTION_ENSURE_RUNNING,
            "training_ensure_running": TRAINING_ENSURE_RUNNING,
        },
        "recent_log": _last_log[-12:],
    }


@app.get("/api/discover")
async def api_discover():
    compute = None
    try:
        compute = {"ok": True, "data": await cli_json("settings", "compute", "list")}
    except Exception as e:
        compute = {"ok": False, "error": str(e)[:400]}
    apps = None
    try:
        apps = await cli_json("market", "list", "--mine")
    except Exception as e:
        apps = {"error": str(e)[:400]}
    return {"compute": compute, "apps": apps}


@app.post("/api/switch")
async def api_switch(body: SwitchRequest):
    global _busy, _last_error
    if _lock.locked() or _busy:
        raise HTTPException(status_code=409, detail="A mode switch is already in progress")
    async with _lock:
        _busy = True
        try:
            result = await switch_to(body.mode)
            return result
        except Exception as e:
            _last_error = str(e)[:1000]
            logger.exception("switch failed")
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": _last_error, "mode": body.mode},
            )
        finally:
            _busy = False


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
