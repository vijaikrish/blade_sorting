import math
import pytest
from rotor_math import RotorInputs, first_harmonic, build_rotor, slot_geometry


def test_harmonic_recovers_quadrature_and_mean():
    readings = [3 + 2 * math.cos(2 * math.pi * i / 8) - 4 * math.sin(2 * math.pi * i / 8) for i in range(8)]
    fit = first_harmonic(readings)
    assert fit.ex_mm == pytest.approx(2)
    assert fit.ey_mm == pytest.approx(-4)
    assert fit.mean_mm == pytest.approx(3)
    assert fit.residual_rms_mm == pytest.approx(0, abs=1e-14)


def test_blade_only_ignores_inactive_invalid_fields():
    model = build_rotor(
        RotorInputs(mode="blade_only", thickness_mm=-1, density_kg_m3=float("nan"), runout_mm=(float("nan"),))
    )
    assert model.disk_mass_g == model.disk_ux_gmm == model.ex_mm == 0
    assert model.radius_mm == 250


def test_combined_dft_and_slots():
    readings = tuple(
        1 + 0.1 * math.cos(2 * math.pi * i / 8) + 0.2 * math.sin(2 * math.pi * i / 8) for i in range(8)
    )
    model = build_rotor(RotorInputs(mode="combined", runout_mm=readings, cg_radius_mm=100))
    geometry = slot_geometry(4, model)
    assert geometry.angles_rad == pytest.approx([0, math.pi / 2, math.pi, 3 * math.pi / 2])
    assert geometry.radii_mm == pytest.approx([100.1, 100.2, 99.9, 99.8])


def test_slot_count_validation():
    for n in (True, 1, 121, 2.5):
        with pytest.raises((TypeError, ValueError)):
            slot_geometry(n, build_rotor(RotorInputs()))


def test_harmonic_rejects_boolean_and_string_readings():
    for values in ([True] + [0.0] * 7, ["0"] * 8):
        with pytest.raises(ValueError):
            first_harmonic(values)
