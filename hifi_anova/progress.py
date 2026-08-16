"""Progress reporting + cooperative cancellation for long-running fits.

Designed for GUI / web back-ends that must show progress and let the user
abort a fit without killing the process. Both :func:`hifi_anova` and
:class:`HiFiANOVATrainer` accept two optional callables:

``progress(event: dict) -> None``
    Called at stage boundaries and the post-fit analysis phase with a small,
    JSON-friendly dict::

        {'event': str,        # 'fit_start'|'stage_start'|'stage_end'
                              # |'stage_progress'|'phase'|'done'
         'stage': str|None,   # 'A'|'B'|'C'|'D'|'mixed'|'analysis'|None
         'fraction': float,   # coarse 0..1 completion estimate
         'message': str,      # human-readable, optional
         'metrics': dict}     # stage metrics (e.g. {'rmse_val': ...}), optional

``should_stop() -> bool``
    Polled between stages and on each Stage-D outer iteration; when it returns
    truthy the fit aborts by raising :class:`HiFiCancelled`.

Both default to ``None``, so existing callers see no behavior change: no
progress is emitted and a fit is never cancelled.

The callbacks run synchronously on the fitting thread. A typical GUI runs the
fit on a worker thread, has ``progress`` marshal the event to the UI thread,
and has ``should_stop`` read a ``threading.Event`` toggled by a Cancel button.
"""

from typing import Optional, Callable, Dict, Any

ProgressCallback = Callable[[Dict[str, Any]], None]
ShouldStopCallback = Callable[[], bool]


class HiFiCancelled(RuntimeError):
    """Raised when a ``should_stop()`` callback requests cancellation.

    A cooperative signal, not a hard interrupt: it fires only at the next
    cancellation checkpoint (a stage boundary or a Stage-D outer iteration),
    so any work already in flight completes first.
    """


# Coarse, monotonically increasing completion estimates per stage. Deliberately
# approximate — stage costs vary with data and config — but enough to drive a
# progress bar. 'mixed' spans A+B; 'analysis' is the post-fit Sobol/CI phase.
_STAGE_FRACTION = {
    'A': 0.2, 'B': 0.45, 'mixed': 0.5, 'C': 0.7, 'D': 0.9,
    'analysis': 0.95, 'done': 1.0,
}


def stage_fraction(stage: Optional[str]) -> float:
    """Best-effort 0..1 completion for a stage label (0.0 if unknown/None)."""
    return float(_STAGE_FRACTION.get(stage, 0.0))


def make_event(event: str, stage: Optional[str] = None,
               message: Optional[str] = None,
               metrics: Optional[Dict[str, Any]] = None,
               fraction: Optional[float] = None) -> Dict[str, Any]:
    """Build a normalized progress event dict (see module docstring for schema)."""
    ev: Dict[str, Any] = {'event': event, 'stage': stage,
                          'fraction': stage_fraction(stage)
                          if fraction is None else float(fraction)}
    if message is not None:
        ev['message'] = message
    if metrics:
        ev['metrics'] = {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in metrics.items()}
    return ev
