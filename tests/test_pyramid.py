import numpy as np

from pyramid import Pyramid

rng = np.random.default_rng(0)


def test_tight_frame_exact_reconstruction():
    P = Pyramid((32, 32), 3, 4)
    x = rng.random((32, 32))
    assert np.allclose(P.synthesise(P.analyse(x)), x, atol=1e-10)
    tot = P.high ** 2 + P.low ** 2 + sum(b ** 2 for b in P.bands) * sum(g ** 2 for g in P.G)
    assert np.allclose(tot, 1.0)


def test_energy_is_preserved():
    """Tight frame: sum of squared real coefficients equals the image energy."""
    P = Pyramid((32, 32), 3, 4)
    x = rng.standard_normal((32, 32))
    p = P.analyse(x)
    e = (p["high"] ** 2).sum() + (p["low"] ** 2).sum() + (p["c"].real ** 2).sum()
    assert np.isclose(e, (x ** 2).sum())


def test_phase_is_position():
    """A grating shifted by d pixels: band phase moves by exactly f*d."""
    P = Pyramid((32, 32), 3, 4)
    f = 2 * np.pi * 4 / 32                       # 4 cycles per 32 px, inside band 2
    xx = np.arange(32)[None, :] * np.ones((32, 1))
    a = P.analyse(np.cos(f * xx))["c"][1, 0]
    b = P.analyse(np.cos(f * (xx - 3)))["c"][1, 0]  # shifted right by 3 px
    dphi = np.angle(b * np.conj(a))
    assert np.allclose(np.abs(dphi), (f * 3) % (2 * np.pi), atol=1e-6)


def test_congruence_high_on_edges_low_on_noise():
    P = Pyramid((32, 32), 3, 4)
    edge = np.zeros((32, 32)); edge[:, 16:] = 1.
    noise = 0.05 * rng.standard_normal((32, 32))
    # without a threshold the contrast is modest: adjacent undecimated scales overlap
    pc_edge = P.congruence(edge)[:, 15:17].mean()
    pc_noise = np.median(P.congruence(noise))
    assert pc_edge > 0.9 and pc_edge - pc_noise > 0.15
    # Kovesi's noise threshold (here: the median scale-summed energy of the noise) removes noise
    T = np.median(np.abs(P.analyse(noise)["c"].sum(0))) * 2
    assert np.median(P.congruence(noise, T)) < 0.1
    assert P.congruence(edge + noise, T)[:, 15:17].mean() > 0.8
