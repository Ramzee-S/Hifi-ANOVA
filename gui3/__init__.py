"""GUI3 — the HiFi Console (live prototype).

Layers:
  engine.py   headless ConsoleEngine wrapping the public hifi_anova API
  server.py   thin FastAPI/WebSocket host:  python -m gui3.server
  console.html  the desk (self-contained client)

Design: gui3p0_design_proposal.md. One fit worker per process (JAX
precision/x64 state is process-global).
"""
