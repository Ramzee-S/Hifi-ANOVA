"""Self-contained numeric-CSV import for the HiFi Console.

GUI3 must stand alone (the `gui2/` tree is untracked and pruned from the
package), so the CSV parser it needs is vendored here rather than imported from
`gui2.services.dataset_io`. This is the essential subset of the gui2 "Numeric
Dataset Convention 1.0" that `engine.cmd_load_csv` actually uses:

- UTF-8 (BOM tolerated), exactly one header row, unique non-empty column names;
- ragged rows rejected (cell count must match the header);
- target column inferred (`y` / `target` / last column) unless named;
- every selected cell numeric and finite; rows missing a selected value are
  dropped (missing tokens: "", na, nan, null — case-insensitive);
- a constant target is rejected (nothing to fit).

No third-party dependency beyond numpy. Errors are actionable
(:class:`CsvError`) so the desk can show the row/column at fault.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

DEFAULT_MISSING = ("", "na", "nan", "null")


class CsvError(ValueError):
    """Actionable CSV-import failure (message names the row/column at fault)."""


def infer_target(header: Sequence[str]) -> str:
    names = list(header)
    for cand in ("y", "target"):
        if cand in names:
            return cand
    lowered = [n.lower() for n in names]
    for cand in ("y", "target"):
        if cand in lowered:
            return names[lowered.index(cand)]
    return names[-1]


def _is_missing(cell: str, tokens: Sequence[str]) -> bool:
    return cell.strip().lower() in tokens


def _numeric(cell: str, row: int, col: str) -> float:
    try:
        v = float(cell)
    except ValueError as exc:
        raise CsvError(f"Non-numeric value {cell.strip()!r} at row {row}, "
                       f"column {col!r}.") from exc
    if not np.isfinite(v):
        raise CsvError(f"Non-finite value {cell.strip()!r} at row {row}, "
                       f"column {col!r}; infinity is not missing.")
    return v


def parse_numeric_csv(data: Any, name: str = "upload.csv",
                      target: Optional[str] = None,
                      features: Optional[Iterable[str]] = None,
                      missing_tokens: Sequence[str] = DEFAULT_MISSING
                      ) -> Dict[str, Any]:
    """Parse a numeric CSV into arrays. ``data`` is ``str`` or ``bytes``.

    Returns ``{X, y, feature_names, target, header, name, rows_read,
    rows_dropped, warnings}``. Raises :class:`CsvError` on any convention
    violation.
    """
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise CsvError(f"{name}: CSV must be UTF-8 (BOM ok); invalid byte "
                           f"near position {exc.start}.") from exc
    else:
        text = str(data)
    tokens = tuple(t.strip().lower() for t in missing_tokens)

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        raw_header = next(reader)
    except StopIteration as exc:
        raise CsvError("Empty file: expected one header row.") from exc
    header = tuple(c.strip() for c in raw_header)
    if not header or any(not h for h in header):
        raise CsvError("Header has an empty column name after trimming.")
    dups = sorted({h for h in header if header.count(h) > 1})
    if dups:
        raise CsvError(f"Duplicate column name(s): {dups}. Names must be unique.")

    rows: List[Sequence[str]] = []
    row_numbers: List[int] = []
    for row in reader:
        rn = reader.line_num
        if not row or not any(c.strip() for c in row):
            continue
        if len(row) != len(header):
            raise CsvError(f"Row {rn} has {len(row)} cells; expected "
                           f"{len(header)} from the header.")
        rows.append(row)
        row_numbers.append(rn)
    if not rows:
        raise CsvError("No non-empty data rows after the header.")

    tgt = target or infer_target(header)
    if tgt not in header:
        raise CsvError(f"Target column {tgt!r} is not in the header.")
    feats = (list(features) if features is not None
             else [h for h in header if h != tgt])
    if not feats:
        raise CsvError("Select at least one feature column.")
    unknown = [f for f in feats if f not in header]
    if unknown:
        raise CsvError(f"Feature column(s) not in the header: {unknown}.")
    if tgt in feats:
        raise CsvError(f"Target column {tgt!r} cannot also be a feature.")

    col = {h: i for i, h in enumerate(header)}
    selected = feats + [tgt]
    keep_rows: List[List[float]] = []
    dropped = 0
    for r, rn in zip(rows, row_numbers):
        if any(_is_missing(r[col[c]], tokens) for c in selected):
            dropped += 1
            continue
        keep_rows.append([_numeric(r[col[c]], rn, c) for c in selected])
    if not keep_rows:
        raise CsvError("All rows dropped as missing — nothing to import.")
    arr = np.asarray(keep_rows, dtype=np.float64)
    X = np.ascontiguousarray(arr[:, :-1])
    y = np.ascontiguousarray(arr[:, -1])
    if np.ptp(y) == 0.0:
        raise CsvError(f"Target {tgt!r} is constant; nothing to fit.")

    warnings: List[str] = []
    for j, f in enumerate(feats):
        if np.ptp(X[:, j]) == 0.0:
            warnings.append(f"Feature {f!r} is constant.")
    return {
        "X": X, "y": y, "feature_names": list(feats), "target": tgt,
        "header": list(header), "name": Path(name).stem,
        "rows_read": len(rows), "rows_dropped": dropped, "warnings": warnings,
    }
