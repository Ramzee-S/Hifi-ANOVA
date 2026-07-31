"""Publication-quality plotting library for HiFi-ANOVA.

Every function takes data + returns ``(fig, axes)`` (or a ``fig``). All figures
are designed for direct inclusion in papers: LaTeX-compatible fonts, clean axes,
a consistent palette, 300 DPI export.

This is a package split of the former single ``plots.py`` module; the public
API is unchanged — ``from hifi_anova.analysis.plots import <name>`` works for
every previously-exported name. Implementation lives in the ``_*`` submodules.
"""
from ._common import PALETTE, VAR_COLORS, apply_style, save_fig  # noqa: F401
from ._regpath import *        # noqa: F401,F403
from ._sobol import *          # noqa: F401,F403
from ._interactions import *   # noqa: F401,F403
from ._components import *      # noqa: F401,F403
from ._variance import *       # noqa: F401,F403
from ._diagnostics import *    # noqa: F401,F403

# Preserve the exact public surface of the former monolithic ``plots.py``. These
# names were importable from it as (unused) module-level imports; re-export them
# so `from hifi_anova.analysis.plots import gridspec` etc. keeps working.
import matplotlib.gridspec as gridspec  # noqa: F401,E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: F401,E402
from typing import Any  # noqa: F401,E402
