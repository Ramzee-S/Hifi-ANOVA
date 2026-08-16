"""HiFi Console server.

Run:  python -m gui3.server  [--port 8630] [--n 4000] [--no-warmup]
then open http://127.0.0.1:8630

One engine, one fit worker; designed for a single browser tab (the event
stream has one consumer). On startup the fatigue demo is loaded and a first
fit is run in the background to absorb the JAX compile (~20 s) before the
user touches anything.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .engine import ConsoleEngine

app = FastAPI(title="HiFi-ANOVA Console")
engine = ConsoleEngine()
HTML_PATH = Path(__file__).with_name("console.html")


@app.get("/")
async def index() -> HTMLResponse:
    # no-store: the console is a dev tool that changes every session —
    # a stale cached copy silently hides new features
    return HTMLResponse(HTML_PATH.read_text(),
                        headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    stop = asyncio.Event()

    async def pump() -> None:
        while not stop.is_set():
            try:
                ev = engine.events.get_nowait()
            except Exception:
                await asyncio.sleep(0.08)
                continue
            try:
                await sock.send_text(json.dumps({"type": "event", **ev}))
                if ev.get("event") in ("fit_done", "cancelled", "error",
                                       "verify_done", "loo_test_done",
                                       "residual_done", "residual_discarded"):
                    await sock.send_text(json.dumps(
                        {"type": "snapshot", **engine.snapshot()}))
            except Exception:
                return

    task = asyncio.create_task(pump())
    try:
        await sock.send_text(json.dumps({"type": "snapshot", **engine.snapshot()}))
        while True:
            msg = json.loads(await sock.receive_text())
            cmd = msg.get("cmd", "")
            reply = engine.dispatch(cmd, **(msg.get("args") or {}))
            await sock.send_text(json.dumps(
                {"type": "reply", "cmd": cmd, "seq": msg.get("seq"), **reply}))
            if cmd not in ("snapshot", "solo", "solo_pair", "export_report"):
                await sock.send_text(json.dumps(
                    {"type": "snapshot", **engine.snapshot()}))
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        task.cancel()


def _warmup(n: int) -> None:
    engine.dispatch("load_demo", n=n)
    engine.dispatch("fit")


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="HiFi-ANOVA Console server")
    ap.add_argument("--port", type=int, default=8630)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n", type=int, default=4000,
                    help="fatigue-demo sample size loaded at startup")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the startup demo fit (first FIT then pays JAX compile)")
    ap.add_argument("--no-warm-neighbors", action="store_true",
                    help="disable idle pre-compilation of neighboring K shapes")
    ap.add_argument("--no-auto-order2", action="store_true",
                    help="disable the default-fit d-rule (ALL pairs ≤4 vars / "
                         "TOP 5-10 / first-order above) on new datasets")
    ap.add_argument("--initial-try", default="balanced",
                    choices=["off", "fast", "balanced", "thorough"],
                    help="INITIAL TRY first-shot planner speed preset "
                         "(budget-ladder heuristic for the first quick fit; "
                         "'off' restores the d-rule/TOP behavior)")
    ap.add_argument("--fit-backend", default="auto",
                    choices=["auto", "numpy", "jax"],
                    help="array backend for fits + desk analytics: 'auto' "
                         "(default) = the NumPy exact core (same code, no "
                         "per-shape XLA compiles — statistically identical); "
                         "'jax' restores the previous behavior")
    ap.add_argument("--no-jax-cache", action="store_true",
                    help="disable the persistent JAX compilation cache")
    a = ap.parse_args()
    if not a.no_jax_cache:
        # persistent compile cache: shapes compiled in ANY earlier session load
        # from disk instead of recompiling (biggest win on GPU backends)
        from pathlib import Path as _P
        import jax
        cache = _P("~/.cache/hifi_anova_gui3/jax").expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    # warm-neighbor pre-compilation only helps the JAX backend; the engine
    # gates it on the ACTIVE backend per fit, so enabling it here just means
    # "warm whenever the desk is (toggled) on jax"
    engine.warm_neighbors = not a.no_warm_neighbors
    engine.auto_order2 = not a.no_auto_order2
    engine.plan_speed = a.initial_try  # INITIAL TRY planner (off = d-rule)
    engine.fit_backend = a.fit_backend  # numpy exact core by default ('auto')
    if not a.no_warmup:
        threading.Thread(target=_warmup, args=(a.n,), daemon=True).start()
    else:
        engine.dispatch("load_demo", n=a.n)
    print(f"HiFi-ANOVA Console → http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
