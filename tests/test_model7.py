import numpy as np

from fog import fog_matrix, observe
from gate import GatedCSC, footprint_pool
from paper import shift, estimate_motion, phase_transport, trust
from worlds import janus_patterns, random_dictionary

rng = np.random.default_rng(1)


def smooth_world(n=64):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(rng.random((n, n)), 2.0) * 4


def test_gate_off_is_model6_csc():
    A = fog_matrix(4)
    g = GatedCSC(random_dictionary(12), A)
    y = observe(rng.random((32, 32)), 4, .01, rng)
    assert np.allclose(g.solve_gated(y, 1e-3, .1, iters=80), g.solve(y, 1e-3, .1, iters=80))


def test_footprint_pool_sees_whole_atom():
    e = np.zeros((32, 32)); e[10, 12] = 1.
    m = footprint_pool(e)
    assert m.shape == (25, 25)
    assert m[3:11, 5:13].min() == 1.0 and m.sum() == 64      # every atom covering (10, 12), no other


def test_shift_and_motion_estimate():
    world = smooth_world()
    a, b = world[10:42, 10:42], world[11:43, 8:40]          # window moved down 1, left 2
    assert np.allclose(shift(a, (-1, 2))[:-1, 2:], b[:-1, 2:])   # content moves up 1, right 2
    A = fog_matrix(8)
    y = (A @ b.ravel()).reshape(32, 32)
    assert estimate_motion(a, y, A) == (-1, 2)              # content moved up 1, right 2


def test_phase_transport_approximates_shift():
    world = smooth_world()
    a = world[10:42, 10:42]
    exact = shift(a, (0, 2), np.nan)
    ph = phase_transport(a, (0, 2))
    m = ~np.isnan(exact)
    m[:, :6] = False; m[:, -4:] = False                      # away from the wrap-around border
    assert np.mean((ph[m] - exact[m]) ** 2) < 0.1 * np.var(a)


def test_trust_accepts_own_history_rejects_foreign():
    A = fog_matrix(8)
    w1, w2 = smooth_world(), smooth_world()
    x, other = w1[5:37, 5:37], w2[5:37, 5:37]
    y = observe(x, 8, .01, rng)
    assert trust(y, x, A, .01, 1e-5) > 0.99
    assert trust(y, other, A, .01, 1e-5) < 0.01


def test_janus_patterns_are_attenuation_py():
    a, b = janus_patterns()
    assert a.shape == b.shape == (128, 128)
    assert 0 <= a.min() and a.max() <= 1 and 0 <= b.min() and b.max() <= 1
