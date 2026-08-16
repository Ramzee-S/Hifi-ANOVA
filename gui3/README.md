# GUI3 — the HiFi Console (experimental)

The model as a mixing console: a local browser desk over the public
`hifi_anova` API. **Experimental / pre-alpha**, shipped as source (not in the
installed wheel).

## Run

```bash
pip install -e ".[gui3]"                 # fastapi + uvicorn (from a source checkout)
python -m gui3.server                    # default port 8630
# then open http://127.0.0.1:8630  (one browser tab per server)
```

Options: `--port`, `--n <demo sample size>` (default 4000), `--no-warmup`,
`--fit-backend auto|numpy|jax`. On startup a heteroscedastic fatigue demo is
loaded and a background fit absorbs the JAX compile (~20 s); warm homoscedastic
refits are sub-second, Stage-D (HETERO) takes a few seconds.

## Full guide

See **[`../docs/GUI_GUIDE.md`](../docs/GUI_GUIDE.md)** — the user-facing guide to
the desk: loading data, faders/basis/backend, mean/variance mutes, SCAN→ROUTE
interactions, the monitor scopes, the parity ladder, the COMPLEMENT bus, the
diagnostics and honesty lamps, and the propose-only selection aids. The
underlying statistics are documented in [`../docs/USER_GUIDE.md`](../docs/USER_GUIDE.md).

## Layers (for contributors)

- `engine.py` — headless `ConsoleEngine`; every desk action is
  `engine.dispatch(cmd, **args)`; single fit worker, cooperative cancel
  (a cancelled/failed fit never replaces the current view).
- `server.py` — FastAPI + one WebSocket; pushes snapshots + progress events.
- `console.html` — self-contained client (no build step, no CDN).
- Tests: `tests/test_gui3_engine.py` (tiny fits only, no browser).
