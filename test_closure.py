"""Fixed closing-blade slot and separately measured bare-rotor vector."""
import itertools
import math
import pytest
from solver import SolverConfig, solve_fixed_closure, vector_sum


def test_fixed_closure_is_constrained_and_vectors_match_full_schedule():
    masses = [33., 34., 50., 36., 37.]
    theta = [2*math.pi*i/5 for i in range(5)]
    x, y = [200*math.cos(a) for a in theta], [200*math.sin(a) for a in theta]
    result = solve_fixed_closure(masses, x, y, fixed_blade_index=2, fixed_slot=4,
                                 disk=(12.,-7.), mode='demonstration',
                                 config=SolverConfig(iterations=1000, restarts=2))
    assert result.permutation[4] == 2
    assert set(result.permutation) == set(range(5))
    assert (result.net_ux_gmm,result.net_uy_gmm) == pytest.approx(vector_sum(masses,result.permutation,x,y,(12.,-7.)))
    truth = min(math.hypot(*vector_sum(masses,p,x,y,(12.,-7.)))
                for p in itertools.permutations(range(5)) if p[4]==2)
    assert result.residual_gmm >= truth-1e-8
    assert result.residual_gmm <= truth+1e-8
    assert any('demonstration' in w.lower() for w in result.warnings)


def test_measured_rotor_vector_replaces_disk_estimate_and_closure_radius():
    m = [30.,40.,50.]
    x,y=[100.,-50.,-50.],[0.,86.6025403784,-86.6025403784]
    r=solve_fixed_closure(m,x,y,fixed_blade_index=2,fixed_slot=0,
                          mode='measured',closure_radius_mm=80.,
                          measured_disk=(25.,-10.),disk=(999.,999.),
                          config=SolverConfig(iterations=10,restarts=1))
    assert r.permutation[0]==2
    assert (r.net_ux_gmm,r.net_uy_gmm)==pytest.approx(vector_sum(m,r.permutation,[80.,-50.,-50.],y,(25.,-10.)))


@pytest.mark.parametrize('kwargs',[{'mode':'pending'},{'mode':'measured','closure_radius_mm':100.},
                                   {'mode':'measured','measured_disk':(1.,2.)},
                                   {'mode':'unknown'}, {'fixed_slot':3}, {'fixed_blade_index':-1}])
def test_incomplete_or_invalid_closure_inputs_rejected(kwargs):
    args=dict(fixed_blade_index=2,fixed_slot=0,mode='demonstration',config=SolverConfig(iterations=2,restarts=1))
    args.update(kwargs)
    with pytest.raises(ValueError):
        solve_fixed_closure([30.,35.,50.],[100.,-50.,-50.],[0.,86.,-86.],**args)
