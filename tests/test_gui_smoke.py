"""Headless GUI checks: wrap gui/hifi_anova_gui.py --smoke / --selftest.

Runs the widget-build smoke and the tiny end-to-end selftest in a subprocess
under a virtual display. Skipped when tkinter, an X display, or xvfb-run is
unavailable (e.g. on ws2). JAX is forced onto CPU so a busy GPU can never
flake the fit (cuSolver contention, X7C S2).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tkinter")

GUI = Path(__file__).resolve().parents[1] / "gui" / "hifi_anova_gui.py"

_HAS_DISPLAY = bool(os.environ.get("DISPLAY"))
_HAS_XVFB = shutil.which("xvfb-run") is not None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not GUI.exists(), reason="gui/ not present"),
    pytest.mark.skipif(not (_HAS_DISPLAY or _HAS_XVFB),
                       reason="no X display and no xvfb-run"),
]


def _run_gui(flag, timeout):
    cmd = [sys.executable, str(GUI), flag]
    if _HAS_XVFB:
        # Prefer a virtual display even when DISPLAY exists: keeps the run
        # deterministic and off the user's screen.
        cmd = ["xvfb-run", "-a"] + cmd
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["HIFI_ANOVA_X64"] = "1"
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def test_gui_smoke():
    r = _run_gui("--smoke", timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "SMOKE OK" in r.stdout, r.stdout[-2000:]


def test_gui_selftest():
    r = _run_gui("--selftest", timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "SELFTEST OK" in r.stdout, r.stdout[-2000:]
