"""Rotor geometry and requested eight-point first-harmonic model.

Mode B is a radial-projection estimate, not a measured mass eccentricity.
"""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class RotorInputs:
    mode: str = "blade_only"
    diameter_mm: float = 500.0
    thickness_mm: float = 50.0
    density_kg_m3: float = 7850.0
    cg_radius_mm: float | None = None
    cg_offset_mm: float = 0.0
    runout_mm: tuple[float, ...] = (0.0,) * 8
    start_angle_deg: float = 0.0  # global slot-1 and runout-zero direction


@dataclass(frozen=True)
class HarmonicFit:
    ex_mm: float
    ey_mm: float
    mean_mm: float
    residual_rms_mm: float
    fitted_mm: tuple[float, ...]


@dataclass(frozen=True)
class RotorModel:
    mode: str
    ex_mm: float
    ey_mm: float
    disk_mass_g: float
    disk_ux_gmm: float
    disk_uy_gmm: float
    radius_mm: float
    runout_mean_mm: float
    runout_residual_rms_mm: float
    runout_fit_mm: tuple[float, ...]
    warnings: tuple[str, ...]
    runout_samples_mm: tuple[float, ...] = (0.0,) * 8
    start_angle_deg: float = 0.0


@dataclass(frozen=True)
class SlotGeometry:
    angles_rad: np.ndarray
    radii_mm: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray


def _finite(value, name, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or not math.isfinite(float(value))
        or (positive and value <= 0)
    ):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return float(value)


def first_harmonic(runout_mm) -> HarmonicFit:
    raw = tuple(runout_mm)
    if any(isinstance(v, (bool, str, bytes)) or not isinstance(v, (int, float, np.number)) for v in raw):
        raise ValueError("runout_mm must contain numeric readings")
    a = np.asarray(raw, dtype=np.float64)
    if a.shape != (8,) or not np.all(np.isfinite(a)):
        raise ValueError("runout_mm must contain exactly eight finite readings")
    mean = math.fsum(map(float, a)) / 8
    centered = a - mean
    t = 2 * np.pi * np.arange(8) / 8
    ex = float(0.25 * np.dot(centered, np.cos(t)))
    ey = float(0.25 * np.dot(centered, np.sin(t)))
    fitted = mean + ex * np.cos(t) + ey * np.sin(t)
    rms = float(np.sqrt(np.mean((a - fitted) ** 2)))
    return HarmonicFit(ex, ey, mean, rms, tuple(map(float, fitted)))


def build_rotor(inputs: RotorInputs) -> RotorModel:
    if not isinstance(inputs, RotorInputs):
        raise TypeError("inputs must be RotorInputs")
    if inputs.mode not in ("blade_only", "combined"):
        raise ValueError("mode must be 'blade_only' or 'combined'")
    start = _finite(inputs.start_angle_deg, "start_angle_deg")
    if not 0 <= start < 360:
        raise ValueError("start_angle_deg must be from 0 inclusive to 360 exclusive")
    if inputs.mode == "blade_only":
        radius = (
            _finite(inputs.cg_radius_mm, "cg_radius_mm", True)
            if inputs.cg_radius_mm is not None
            else _finite(inputs.diameter_mm, "diameter_mm", True) / 2
            + _finite(inputs.cg_offset_mm, "cg_offset_mm")
        )
        if radius <= 0:
            raise ValueError("radius must be positive")
        return RotorModel(inputs.mode, 0.0, 0.0, 0.0, 0.0, 0.0, radius, 0.0, 0.0, (0.0,) * 8, (), start_angle_deg=start)
    d = _finite(inputs.diameter_mm, "diameter_mm", True)
    th = _finite(inputs.thickness_mm, "thickness_mm", True)
    rho = _finite(inputs.density_kg_m3, "density_kg_m3", True)
    r = (
        _finite(inputs.cg_radius_mm, "cg_radius_mm", True)
        if inputs.cg_radius_mm is not None
        else d / 2 + _finite(inputs.cg_offset_mm, "cg_offset_mm")
    )
    if r <= 0:
        raise ValueError("radius must be positive")
    fit = first_harmonic(inputs.runout_mm)
    e = math.hypot(fit.ex_mm, fit.ey_mm)
    if e >= r:
        raise ValueError("runout eccentricity must be smaller than nominal CG radius")
    mass = rho * math.pi * (d / 2000) ** 2 * (th / 1000) * 1000
    phase = math.radians(start)
    ex = fit.ex_mm * math.cos(phase) - fit.ey_mm * math.sin(phase)
    ey = fit.ex_mm * math.sin(phase) + fit.ey_mm * math.cos(phase)
    ux, uy = mass * ex, mass * ey
    if not all(map(math.isfinite, (mass, ux, uy))):
        raise ValueError("input dynamic range overflows model")
    return RotorModel(
        inputs.mode,
        ex,
        ey,
        mass,
        ux,
        uy,
        r,
        fit.mean_mm,
        fit.residual_rms_mm,
        fit.fitted_mm,
        ("Runout-derived eccentricity is a geometry estimate, not measured mass eccentricity.",),
        tuple(map(float, inputs.runout_mm)),
        start,
    )


def slot_geometry(n: int, rotor: RotorModel) -> SlotGeometry:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 2 <= n <= 120:
        raise ValueError("n must be an integer from 2 through 120")
    angles = math.radians(rotor.start_angle_deg) + 2 * np.pi * np.arange(n, dtype=np.float64) / n
    radii = rotor.radius_mm + rotor.ex_mm * np.cos(angles) + rotor.ey_mm * np.sin(angles)
    return SlotGeometry(angles, radii, radii * np.cos(angles), radii * np.sin(angles))
