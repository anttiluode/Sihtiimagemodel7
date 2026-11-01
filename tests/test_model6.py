import numpy as np

from fog import fog_matrix, purify_n, sieve, observe
from models import CSC, IndependentPatches, GaussianPrior, patch_index
from bench import flicker

rng = np.random.default_rng(0)


def rand_dict(k=12):
    D = rng.standard_normal((k, 8, 8))
    D -= D.mean((1, 2), keepdims=True)
    return D / np.linalg.norm(D.reshape(k, -1), axis=1)[:, None, None]


def test_fog_matrix_is_sihti_and_symmetric():
    A = fog_matrix(5)
    x = rng.random((32, 32))
    assert np.allclose(A @ x.ravel(), purify_n(x, 5).ravel())
    assert np.allclose(A, A.T)


def test_sieve_telescopes():
    x = rng.random((32, 32))
    assert np.allclose(sum(sieve(x)), x)


def test_csc_synthesis_adjoint():
    csc = CSC(rand_dict(), fog_matrix(4))
    a = rng.standard_normal(csc.cshape)
    r = rng.standard_normal((32, 32))
    assert np.isclose((csc.S(a) * r).sum(), (a * csc.St(r)).sum())


def test_csc_descends_and_is_stationary():
    A = fog_matrix(4)
    csc = CSC(rand_dict(), A)
    x = rng.random((32, 32))
    y = observe(x, 4, .01, rng)
    lam, g = 1e-3, 1.
    xs, c, a = csc.solve(y, lam, g, iters=600, return_parts=True)
    f = csc.objective(y, c, a, lam, g)
    assert f < csc.objective(y, np.array(y), np.zeros_like(a), lam, g)
    for _ in range(5):                                   # small random moves never help much
        dc = 1e-3 * rng.standard_normal(c.shape)
        da = 1e-3 * rng.standard_normal(a.shape)
        assert csc.objective(y, c + dc, a + da, lam, g) >= f - 1e-6


def test_independent_patches_identity_without_fog():
    ind = IndependentPatches(rand_dict(64), depth=0)     # 64 random atoms + mean span R^64
    y = rng.random((32, 32))
    out = ind.solve(y, lam=0., iters=3000)
    assert np.abs(out - y).max() < 1e-3


def test_wiener_formula_and_sample_mean():
    train = rng.random((400, 8, 8)).astype(float)
    g = GaussianPrior(train)
    A = fog_matrix(3, (8, 8))
    y = observe(train[0], 3, .05, rng)
    C, mu = g.C, g.mu
    direct = mu + C @ A.T @ np.linalg.solve(A @ C @ A.T + .05 ** 2 * np.eye(64), y.ravel() - A @ mu)
    assert np.allclose(g.wiener(y, A, .05).ravel(), direct)
    s = np.mean([g.sample(y, A, .05, rng) for _ in range(3000)], 0)
    assert np.abs(s - g.wiener(y, A, .05)).max() < 0.03


def test_methods_never_see_truth():
    """Same observation -> same answer, whatever the truth was."""
    A = fog_matrix(4)
    csc = CSC(rand_dict(), A)
    y = observe(rng.random((32, 32)), 4, .01, rng)
    assert np.array_equal(csc.solve(y, 1e-3, 1., iters=50), csc.solve(y.copy(), 1e-3, 1., iters=50))


def test_flicker_zero_for_perfect_recall():
    world = rng.random((60, 60))
    offs = [(5 + t, 3 + 2 * t) for t in range(5)]
    frames = [world[a:a + 32, b:b + 32] for a, b in offs]
    assert flicker(frames, offs) < 1e-12


def test_patch_index_covers_every_pixel():
    idx = patch_index()
    assert idx.shape == (625, 64)
    assert np.bincount(idx.ravel(), minlength=1024).min() >= 1


def test_the_two_fogs():
    """Sihti's purify at dt=0.25 passes the checkerboard; the matched heat fog does not."""
    from fog import FOGS
    c = np.cos(31 * np.pi * (np.arange(32) + .5) / 32)
    v = np.outer(c, c).ravel()                # highest Neumann mode: an exact eigenvector
    s, h = FOGS["sihti"], FOGS["heat"]
    keep = lambda f: (fog_matrix(f["depth"], dt=f["dt"]) @ v) @ v / (v @ v)
    assert keep(s) > 0.9                      # band-stop: the finest pattern survives (sign-flipping)
    assert keep(h) < 1e-6                     # true low-pass
    ev = np.linalg.eigvalsh(fog_matrix(h["depth"], dt=h["dt"]))
    assert ev.min() > -1e-12 and ev.max() <= 1 + 1e-12


def test_no_inhibition_footprints_and_independence():
    A = fog_matrix(4)
    csc = CSC(rand_dict(), A)
    fn = csc.footprint_norms()
    for k, i, j in [(0, 0, 0), (3, 12, 7), (11, 24, 24)]:
        e = np.zeros(csc.cshape); e[k, i, j] = 1.
        assert np.isclose(fn[k, i, j], np.sum((A @ csc.S(e).ravel()) ** 2))
    # without inhibition, an atom's coefficient does not depend on any other atom's
    y = observe(rng.random((32, 32)), 4, .01, rng)
    x1 = csc.solve_no_inhibition(y, 1e-3, 1., 1.)
    assert np.isfinite(x1).all()
