"""Synthetic CSV demonstration is reproducible and not a measurement record."""
from data_io import closure_demo_blades, read_blades_csv


def test_100_blade_closure_csv_has_99_blades_in_30_40_g_and_fixed_50_g():
    blades = read_blades_csv("examples/100_blades_with_closure.csv")
    assert len(blades) == 100
    assert all(30 <= b.weight <= 40 for b in blades[:99])
    assert blades[-1].weight == 50
    assert blades[-1].blade_name == "Closing-100"
    assert blades == closure_demo_blades(seed=42)


def test_demo_blades_repeat_seed_and_vary_when_seed_changes():
    assert closure_demo_blades(seed=8) == closure_demo_blades(seed=8)
    assert closure_demo_blades(seed=8) != closure_demo_blades(seed=9)
