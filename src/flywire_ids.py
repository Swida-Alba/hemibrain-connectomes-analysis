"""Canonical FlyWire body-ID handling.

FlyWire root/body IDs are identifiers, not measurements.  They are commonly
larger than JavaScript's safe-integer range, so DROCAT keeps them as exact
decimal strings everywhere inside the application.  The CAVE/CloudVolume
adapters are the only consumers that receive integers, and they use
``body_id_to_api_int`` at that narrow boundary.
"""

from __future__ import annotations

import math
import numbers
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional


JS_SAFE_INTEGER = 2**53 - 1
SIGNED_INT64_MAX = 2**63 - 1

_DIGITS = re.compile(r"^[0-9]+$")
_INTEGRAL_DECIMAL = re.compile(r"^([0-9]+)\.0+$")


class FlyWireBodyIdError(ValueError):
    """Raised when a FlyWire body ID cannot be preserved exactly."""


def is_banc_dataset(dataset: object) -> bool:
    """Return whether *dataset* identifies a BANC release."""

    return "banc" in str(dataset or "").strip().lower()


def is_flywire_dataset(dataset: object) -> bool:
    """Return whether *dataset* belongs to the FlyWire family.

    FAFB and BANC are both FlyWire datasets.  Keeping this predicate here
    prevents individual callers from accidentally handling only FAFB.
    """

    normalized = str(dataset or "").strip().lower()
    return any(token in normalized for token in ("flywire", "fafb", "banc"))


def dataset_folder(dataset: object) -> str:
    """Map a dataset identifier to the repository folder convention."""

    return str(dataset or "").replace(":", "_").replace(".", "_")


def resolve_flywire_dataset_dir(
    project_root: str | Path, dataset: object
) -> Optional[Path]:
    """Resolve an existing FlyWire dataset directory without FAFB fallback.

    The exact name is tried first, followed by the repository-safe spelling
    used for identifiers containing ``:`` or ``.``.  In particular, a missing
    BANC directory never silently resolves to the FAFB directory.
    """

    root = Path(project_root) / "datasets"
    raw_name = str(dataset or "").strip()
    candidates = []
    for name in (raw_name, dataset_folder(raw_name)):
        if name and name not in candidates:
            candidates.append(name)
    for name in candidates:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _canonical_digits(digits: str) -> str:
    """Canonicalize decimal digits while preserving their exact integer."""

    canonical = digits.lstrip("0")
    return canonical or "0"


def normalize_flywire_body_id(value: Any, *, field: str = "bodyId") -> str:
    """Return an exact canonical decimal string for one FlyWire body ID.

    Integral Python/numpy integers are safe.  Small integral floats and
    legacy strings such as ``"123.0"`` are accepted.  A float-like value
    above JavaScript's safe-integer limit is rejected because its original
    digits may already have been rounded and cannot be recovered.
    """

    if value is None or isinstance(value, bool):
        raise FlyWireBodyIdError(f"{field} is missing or boolean: {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise FlyWireBodyIdError(f"{field} is empty")
        if _DIGITS.fullmatch(text):
            return _canonical_digits(text)
        match = _INTEGRAL_DECIMAL.fullmatch(text)
        if match:
            digits = _canonical_digits(match.group(1))
            if int(digits) > JS_SAFE_INTEGER:
                raise FlyWireBodyIdError(
                    f"{field} looks like a rounded float ({value!r}); "
                    "the exact FlyWire ID cannot be recovered"
                )
            return digits
        raise FlyWireBodyIdError(
            f"{field} must be an exact decimal integer string: {value!r}"
        )

    if isinstance(value, numbers.Integral):
        integer = int(value)
    elif isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise FlyWireBodyIdError(f"{field} is not integral: {value!r}")
        integer = int(value)
    elif isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise FlyWireBodyIdError(f"{field} is not a finite integer: {value!r}")
        if abs(numeric) > JS_SAFE_INTEGER:
            raise FlyWireBodyIdError(
                f"{field} arrived as an unsafe float ({value!r}); "
                "read FlyWire IDs as strings before parsing"
            )
        integer = int(numeric)
    else:
        # Decimal-like third-party values occasionally expose an exact string
        # representation.  Do not accept scientific notation or arbitrary
        # coercions here; identifiers must remain lossless.
        try:
            text = str(value).strip()
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise FlyWireBodyIdError(f"invalid {field}: {value!r}") from exc
        if _DIGITS.fullmatch(text):
            return _canonical_digits(text)
        raise FlyWireBodyIdError(f"invalid {field}: {value!r}")

    if integer < 0:
        raise FlyWireBodyIdError(f"{field} must be non-negative: {value!r}")
    return str(integer)


def normalize_flywire_body_ids(values: Iterable[Any], *, field: str = "bodyId") -> list[str]:
    """Normalize an iterable of FlyWire body IDs to canonical strings."""

    return [normalize_flywire_body_id(value, field=field) for value in values]


def body_id_to_api_int(value: Any, *, field: str = "bodyId") -> int:
    """Convert one canonical FlyWire ID to the checked API integer form."""

    canonical = normalize_flywire_body_id(value, field=field)
    integer = int(canonical)
    if integer > SIGNED_INT64_MAX:
        raise FlyWireBodyIdError(
            f"{field} exceeds the signed 64-bit API range: {canonical}"
        )
    return integer


def normalize_flywire_id_columns(frame, columns: Iterable[str]):
    """Normalize present identifier columns in a pandas-like frame in place."""

    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value, _column=column: normalize_flywire_body_id(
                    value, field=_column
                )
            ).astype("string")
    return frame
