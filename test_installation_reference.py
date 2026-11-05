"""Installation angle and rotor-fixed runout reference regressions."""
import math
import pytest
from rotor_math import RotorInputs, build_rotor, slot_geometry


def test_start_angle_rotates_slot_and_runout_axes_together():
    readings = tuple(.04 * math.cos(j * math.pi / 4) for j in range(8))
    model = build_rotor(RotorInputs(mode="combined", start_angle_deg=90, runout_mm=readings,
                                    cg_radius_mm=250))
    slots = slot_geometry(4, model)
    assert slots.angles_rad == pytest.approx([math.pi / 2, math.pi, 3*math.pi/2, 2*math.pi])
    assert (model.ex_mm, model.ey_mm) == pytest.approx((0, .04), abs=1e-12)
    assert slots.radii_mm == pytest.approx([250.04,250,249.96,250])
    assert model.disk_ux_gmm == pytest.approx(0, abs=1e-9)
    assert model.disk_uy_gmm == pytest.approx(model.disk_mass_g*.04)


def test_blade_only_start_angle_changes_mounting_schedule_not_disk():
    model = build_rotor(RotorInputs(mode="blade_only", start_angle_deg=27, cg_radius_mm=123))
    geometry = slot_geometry(5, model)
    assert geometry.angles_rad[0] == pytest.approx(math.radians(27))
    assert (model.disk_ux_gmm,model.disk_uy_gmm) == (0,0)


@pytest.mark.parametrize("angle",[-1,360,float("nan"),True,"90"])
def test_invalid_start_angle_rejected(angle):
    with pytest.raises(ValueError):
        build_rotor(RotorInputs(start_angle_deg=angle))
