from __future__ import annotations

import csv
import math
import numbers
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Blade:
    blade_name: str
    weight: float


def validate_blades(blades: Iterable[Blade]) -> tuple[Blade, ...]:
    try:
        items = tuple(blades)
    except TypeError as exc:
        raise ValueError("blades must be an iterable of Blade records") from exc
    if not 2 <= len(items) <= 120:
        raise ValueError("blade count must be between 2 and 120")
    names = set()
    result = []
    for i, blade in enumerate(items, 1):
        if not isinstance(blade, Blade):
            raise ValueError(f"blade {i} must be a Blade")
        if not isinstance(blade.blade_name, str) or not blade.blade_name.strip():
            raise ValueError(f"blade {i} name must be nonblank")
        if blade.blade_name in names:
            raise ValueError(f"duplicate blade name {blade.blade_name!r}")
        names.add(blade.blade_name)
        if isinstance(blade.weight, bool):
            raise ValueError(f"blade {i} weight must be a finite positive number")
        try:
            weight = float(blade.weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"blade {i} weight must be a finite positive number") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"blade {i} weight must be a finite positive number")
        result.append(Blade(blade.blade_name, weight))
    return tuple(result)


def read_blades_csv(path) -> tuple[Blade, ...]:
    if path is None:
        raise ValueError("CSV path is required")
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;")
            except csv.Error:
                dialect = csv.excel
            # Sniff only the delimiter. Sniffer may infer doublequote=False
            # from malformed records or skipinitialspace=True and lose IDs.
            reader = csv.reader(
                f,
                delimiter=dialect.delimiter,
                quotechar='"',
                doublequote=True,
                strict=True,
                skipinitialspace=False,
            )
            try:
                headers = next(reader)
            except StopIteration as exc:
                raise ValueError("CSV is empty; expected blade_name and weight headers") from exc
            if len(set(headers)) != len(headers):
                raise ValueError("duplicate CSV headers")
            if "blade_name" not in headers or "weight" not in headers:
                raise ValueError("CSV requires blade_name and weight columns")
            ni, wi = headers.index("blade_name"), headers.index("weight")
            blades = []
            for rownum, row in enumerate(reader, 2):
                if not row or all(not cell.strip() for cell in row):
                    raise ValueError(f"row {rownum}: blank row")
                if len(row) != len(headers):
                    raise ValueError(f"row {rownum}: malformed row (expected {len(headers)} columns)")
                name, raw = row[ni], row[wi]
                try:
                    weight = float(raw)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"row {rownum}: invalid weight {raw!r}; expected finite positive number"
                    ) from exc
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(f"row {rownum}: invalid weight {raw!r}; expected finite positive number")
                blades.append(Blade(name, weight))
                if len(blades) > 120:
                    raise ValueError(f"row {rownum}: blade count exceeds 120")
    except csv.Error as exc:
        raise ValueError(f"malformed CSV near line {reader.line_num}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8 encoded") from exc
    if not blades:
        raise ValueError("CSV contains no blade data")
    return validate_blades(blades)


def sample_blades(n: int = 32, seed: int = 42) -> tuple[Blade, ...]:
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 120:
        raise ValueError("blade count must be an integer between 2 and 120")
    rng = random.Random(seed)
    return tuple(Blade(f"Blade-{i + 1:03d}", round(100.0 + rng.uniform(-3.5, 3.5), 3)) for i in range(n))


def closure_demo_blades(seed: int = 42) -> tuple[Blade, ...]:
    """Synthetic example only: 99 ordinary masses and one 50 g closing blade."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    return tuple(
        [Blade(f"Blade-{i + 1:03d}", round(rng.uniform(30, 40), 3)) for i in range(99)]
        + [Blade("Closing-100", 50.0)]
    )


def _safe_excel_name(name: str) -> str:
    stripped = name.lstrip(" \t\r\n\x00\x0b\x0c")
    if stripped.startswith(("=", "+", "-", "@")):
        return "'" + name
    return name


def export_schedule_csv(
    path,
    blades: Iterable[Blade],
    permutation: Sequence[int],
    angles_rad: Sequence[float],
    *,
    metadata=None,
    overwrite: bool = False,
) -> None:
    """Export the four mounting columns; metadata is reserved, not serialized.

    Publish a complete sibling temporary file, refusing any existing target
    unless overwrite is the explicit boolean True. Formula-like IDs receive
    a leading apostrophe; safe identifiers are preserved exactly.
    """
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be the explicit boolean True or False")
    if path is None:
        raise ValueError("output path is required")
    items = validate_blades(blades)
    n = len(items)
    try:
        perm = tuple(permutation)
        angles = tuple(angles_rad)
    except TypeError as exc:
        raise ValueError("permutation and angles must be sequences") from exc
    if (
        len(perm) != n
        or any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in perm)
        or set(perm) != set(range(n))
    ):
        raise ValueError("permutation must contain every blade index exactly once")
    if len(angles) != n or any(
        isinstance(a, bool) or not isinstance(a, numbers.Real) or not math.isfinite(a) for a in angles
    ):
        raise ValueError("angles_rad must contain one finite angle per blade")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary when supplied")
    target = Path(path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(str(target))
    rows = [
        [slot + 1, f"{math.degrees(angles[slot]):.9f}".rstrip("0").rstrip("."),
         _safe_excel_name(items[idx].blade_name), items[idx].weight]
        for slot, idx in enumerate(perm)
    ]
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=target.parent, delete=False
        ) as f:
            temp_name = f.name
            writer = csv.writer(f)
            writer.writerow(["slot_number", "angle_deg", "blade_name", "weight_g"])
            writer.writerows(rows)
        if overwrite:
            os.replace(temp_name, target)
            temp_name = None
        else:
            # Atomic exclusive publication of a complete sibling file. Hard links
            # are supported by the verified local NTFS filesystem; failure leaves
            # the target untouched instead of exposing a partial schedule.
            os.link(temp_name, target)
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
