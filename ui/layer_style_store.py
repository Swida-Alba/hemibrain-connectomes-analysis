"""Auto-saving layer-style draft store for the Skeleton advanced layer editor.

Drafts live in ``local_data/layer_style_drafts/`` (gitignored) as one CSV per
draft plus a JSON metadata sidecar:

    my_layers.csv         header: layer,neuron,color,synapse_color,pre_synaptic_color,post_synaptic_color
    my_layers.meta.json   {name, slug, created_at, updated_at, dirty, row_count}

Every edit is flushed atomically (temp file + ``os.replace``), so a draft
survives even if the UI process or its port dies mid-session. The ``dirty``
flag stays True until the user explicitly exports the CSV.

The CSV layout matches the Skeleton backend's ``layer_map_csv`` contract
(``layer`` + ``neuron`` + optional color columns), so a draft file can be
passed to ``VisualizeSkeleton(layer_map_csv=...)`` without conversion.
"""
import csv
import json
import math
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import PROJECT_ROOT

# Columns written to the draft CSV, in order. All columns are always written
# so a draft round-trips losslessly through the Skeleton backend parser.
LAYER_STYLE_COLUMNS = (
    "layer",
    "neuron",
    "color",
    "synapse_color",
    "pre_synaptic_color",
    "post_synaptic_color",
)
REQUIRED_COLUMNS = ("layer", "neuron")

# The colour columns written/exported depend on the synapse mode: 'synapse'
# carries a single per-connection synapse colour, while 'pre-post sites' carries
# separate pre- and post-synaptic site colours.
MODE_COLUMNS = {
    "synapse": ("layer", "neuron", "color", "synapse_color"),
    "pre-post sites": ("layer", "neuron", "color", "pre_synaptic_color", "post_synaptic_color"),
}
COLOR_FIELDS = {"color", "synapse_color", "pre_synaptic_color", "post_synaptic_color"}


# ---------------------------------------------------------------------------
# Layer options (selection box) & mode-aware column helpers
# ---------------------------------------------------------------------------

def mode_columns(mode: str) -> tuple:
    """Ordered columns for a synapse mode (fall back to all columns)."""
    return MODE_COLUMNS.get(mode, MODE_COLUMNS["synapse"])


def available_layers(rows: List[dict]) -> List[int]:
    """Contiguous layer numbers offered by the layer selection box.

    Layers always start at 1 and grow one at a time: the returned options are
    ``1 .. k+1`` where ``k`` is the length of the contiguous run of used layers
    starting at 1. If no row uses layer 1, only ``[1]`` is returned (never a
    higher number like 2), so layer assignment stays gapless.
    """
    used = set()
    for row in rows or []:
        try:
            used.add(int(float(str(row.get("layer", "") or "").strip())))
        except (TypeError, ValueError):
            continue
    run = 0
    while (run + 1) in used:
        run += 1
    return list(range(1, run + 2))


_store_dir = PROJECT_ROOT / "local_data" / "layer_style_drafts"
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Paths & naming
# ---------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Reduce a draft name to a safe file slug (empty string if unusable)."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return slug[:80]


def _csv_path(slug: str) -> Path:
    return _store_dir / f"{slug}.csv"


def _meta_path(slug: str) -> Path:
    return _store_dir / f"{slug}.meta.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text crash-safely: temp file in the same dir, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Row validation / normalization
# ---------------------------------------------------------------------------

def normalize_rows(rows: List[dict]) -> List[dict]:
    """Coerce rows to the canonical layer-style shape.

    Keeps the known columns as strings, drops nothing; missing keys become
    empty strings so the editor can hold half-filled rows.
    """
    normalized = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {col: str(row.get(col, "") or "").strip() for col in LAYER_STYLE_COLUMNS}
        )
    return normalized


def validate_rows(rows: List[dict]) -> List[str]:
    """Return human-readable errors (empty list when the table is usable).

    Half-filled rows are reported by 1-based index. A row with no neuron and
    no color is ignored as an empty scaffolding row.
    """
    errors = []
    numeric_layers = []
    for i, row in enumerate(rows or [], start=1):
        if not isinstance(row, dict):
            errors.append(f"Row {i}: not a mapping")
            continue
        layer = str(row.get("layer", "") or "").strip()
        neuron = str(row.get("neuron", "") or "").strip()
        if not layer and not neuron:
            continue  # completely empty rows are ignored, not invalid
        if not layer:
            errors.append(f"Row {i}: missing layer")
        else:
            try:
                parsed = float(layer)
                if not math.isfinite(parsed) or not parsed.is_integer():
                    raise ValueError
                numeric_layers.append(int(parsed))
            except (TypeError, ValueError, OverflowError):
                errors.append(f"Row {i}: layer '{layer}' is not a number")
        if not neuron:
            errors.append(f"Row {i}: missing neuron")

    if numeric_layers:
        used = set(numeric_layers)
        minimum = min(used)
        maximum = max(used)
        if minimum not in (0, 1):
            errors.append(
                "Layers must start at 0 or 1 and remain continuous "
                f"(found {minimum})"
            )
        missing = [layer for layer in range(minimum, maximum + 1) if layer not in used]
        if missing:
            shown = ", ".join(str(layer) for layer in missing[:8])
            if len(missing) > 8:
                shown += ", …"
            errors.append(f"Layers are not continuous: missing layer(s) {shown}")
    return errors


def complete_rows(rows: List[dict]) -> List[dict]:
    """Rows usable as a layer-map input (layer + neuron filled)."""
    return [
        row for row in normalize_rows(rows)
        if all(row[col] for col in REQUIRED_COLUMNS)
    ]


def next_layer_number(rows: List[dict]) -> int:
    """Return the next layer number for a newly applied selection batch.

    An empty table starts at layer 1. Existing 0-based CSVs remain supported;
    their next layer is still the current maximum plus one. Callers that need
    to accept the result as runnable input should validate the table first.
    """
    layers = []
    for row in complete_rows(rows):
        try:
            value = float(str(row.get("layer", "")).strip())
            if math.isfinite(value) and value.is_integer():
                layers.append(int(value))
        except (TypeError, ValueError, OverflowError):
            continue
    return max(layers) + 1 if layers else 1


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _read_meta(slug: str) -> Optional[dict]:
    try:
        meta = json.loads(_meta_path(slug).read_text(encoding="utf-8"))
        if isinstance(meta, dict) and meta.get("name"):
            return meta
    except (OSError, ValueError):
        pass
    return None


def _write_meta(slug: str, meta: dict) -> None:
    _atomic_write_text(_meta_path(slug), json.dumps(meta, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_draft(name: str, rows: List[dict], dirty: bool = True) -> Optional[str]:
    """Auto-save a layer style table under *name*; returns the slug or None.

    The CSV and its metadata are written atomically. When a draft was loaded
    under a different name, call :func:`delete_draft` on the old name to get
    rename semantics (the editor does this).
    """
    slug = sanitize_name(name)
    if not slug:
        return None
    rows = normalize_rows(rows)
    with _lock:
        existing = _read_meta(slug)
        now = _now_iso()
        meta = {
            "name": (name or "").strip(),
            "slug": slug,
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
            "dirty": bool(dirty),
            "row_count": len(rows),
        }
        try:
            _atomic_write_text(_csv_path(slug), rows_to_csv(rows))
            _write_meta(slug, meta)
        except OSError:
            return None
    return slug


def _csv_quote(value: str) -> str:
    if any(ch in value for ch in ',"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def rows_to_csv(rows: List[dict]) -> str:
    """Serialize rows in the layer-map CSV format (all columns always present)."""
    return rows_to_csv_for_mode(rows, mode=None, columns=LAYER_STYLE_COLUMNS)


def rows_to_csv_for_mode(rows: List[dict], mode: str, columns=None) -> str:
    """Serialize rows in the layer-map CSV format for a synapse mode.

    For 'synapse' the output keeps ``layer, neuron, color, synapse_color``; for
    'pre-post sites' it keeps ``layer, neuron, color, pre_synaptic_color,
    post_synaptic_color``, so exported/uploaded CSVs match the mode.
    """
    normalized = normalize_rows(rows)
    if columns is None:
        columns = mode_columns(mode)
    lines = [",".join(columns)]
    for row in normalized:
        lines.append(",".join(_csv_quote(row[col]) for col in columns))
    return "\n".join(lines) + "\n"


def load_draft(name: str) -> Optional[List[dict]]:
    """Load a draft's rows (normalized), or None when the draft is absent."""
    slug = sanitize_name(name)
    if not slug:
        return None
    path = _csv_path(slug)
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {col: (record.get(col) or "").strip() for col in LAYER_STYLE_COLUMNS}
                for record in reader
            ]
        return rows
    except (OSError, csv.Error, UnicodeDecodeError, ValueError):
        return None


def _quote_css_colors(text: str) -> str:
    """Quote unquoted CSS color functions so their commas do not split fields.

    ``rgba/rgb/hsla/hsl(...)`` values contain commas; a hand-written CSV that
    leaves them unquoted would otherwise be split across columns by the CSV
    reader. Already-quoted values are left untouched.
    """
    return re.sub(
        r'(?<![\"\w])(rgba|rgb|hsla|hsl)\s*\([^)]*\)',
        lambda m: chr(34) + m.group(0) + chr(34),
        text,
    )


def load_rows_from_csv_text(text: str) -> List[dict]:
    """Parse CSV *text* into normalized layer-style rows (used by CSV upload)."""
    import io
    reader = csv.DictReader(io.StringIO(_quote_css_colors(text)))
    rows = [
        {col: (record.get(col) or "").strip() for col in LAYER_STYLE_COLUMNS}
        for record in reader
    ]
    return normalize_rows(rows)


def get_meta(name: str) -> Optional[dict]:
    """Return a copy of the draft metadata, or None."""
    slug = sanitize_name(name)
    if not slug:
        return None
    with _lock:
        meta = _read_meta(slug)
    return dict(meta) if meta else None


def list_drafts() -> List[dict]:
    """Metadata of every draft, newest update first."""
    drafts = []
    with _lock:
        if _store_dir.exists():
            for meta_file in _store_dir.glob("*.meta.json"):
                slug = meta_file.name[: -len(".meta.json")]
                meta = _read_meta(slug)
                # Skip orphan metadata whose CSV is gone.
                if meta and _csv_path(slug).exists():
                    drafts.append(meta)
    drafts.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
    return drafts


def pending_drafts() -> List[dict]:
    """Drafts with unsaved (dirty) edits - the recovery reminder list."""
    return [meta for meta in list_drafts() if meta.get("dirty")]


def set_dirty(name: str, dirty: bool) -> bool:
    """Flip the dirty flag (e.g. False after a successful export)."""
    slug = sanitize_name(name)
    if not slug:
        return False
    with _lock:
        meta = _read_meta(slug)
        if meta is None:
            return False
        meta["dirty"] = bool(dirty)
        meta["updated_at"] = _now_iso()
        try:
            _write_meta(slug, meta)
        except OSError:
            return False
    return True


def mark_exported(name: str) -> bool:
    """Mark a draft as exported: no pending unsaved changes."""
    return set_dirty(name, False)


def mark_dirty(name: str) -> bool:
    """Mark a draft as having pending unsaved changes."""
    return set_dirty(name, True)


def draft_csv_path(name: str) -> Optional[str]:
    """Absolute path of the draft CSV (usable as the Skeleton layer_map_csv)."""
    slug = sanitize_name(name)
    if not slug:
        return None
    path = _csv_path(slug)
    return str(path) if path.exists() else None


def delete_draft(name: str) -> bool:
    """Remove a draft's CSV and metadata. Returns success."""
    slug = sanitize_name(name)
    if not slug:
        return False
    with _lock:
        existed = _csv_path(slug).exists() or _meta_path(slug).exists()
        try:
            _csv_path(slug).unlink(missing_ok=True)
            _meta_path(slug).unlink(missing_ok=True)
        except OSError:
            return False
    return existed
