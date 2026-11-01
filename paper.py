"""The living paper (Model 5) carried by the competing atoms (Model 6) through the fog.

Frame t (observation y_t through the fog A):
 1. motion: integer translation d, estimated from the data only, by choosing the shift of the
    previous estimate whose fogged version best matches y_t (search +-3 px, interior pixels);
 2. transport: mu = shift(x_{t-1}, d), pi_minus = shift(pi_{t-1}, d); pixels entering
    the frame get precision 0. "phase" transport (report only) instead advances every pyramid
    coefficient's phase by omega_s * (d . direction_k) -- position as phase, which also works
    for sub-pixel motion but is exact only for narrow bands;
 3. trust: is this memory of this scene? r = y_t - A mu on interior pixels, pooled log-likelihood
    ratio of N(0, v_same) vs N(0, 2 var(y_t) + s^2); trust = sigmoid(LLR). v_same = s^2 + e_fog,
    e_fog = the typical fogged reconstruction error, measured on validation;
 4. inference: Model 6's objective + beta/2 sum_p trust * pi_minus_p (x_p - mu_p)^2;
 5. update: pi_t = pi_minus + 1 (every pixel was observed again, through the fog).
"""
import numpy as np

from pyramid import Pyramid


def shift(img, d, fill=0.0):
    """Integer translation, content moves by d=(dy,dx); uncovered pixels get `fill`."""
    out = np.full_like(img, fill)
    dy, dx = d
    h, w = img.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def estimate_motion(prev, y, A, search=3, margin=6):
    best, bd = np.inf, (0, 0)
    sl = (slice(margin, -margin), slice(margin, -margin))
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            m = shift(prev, (dy, dx), np.nan)
            mm = np.where(np.isnan(m), prev.mean(), m)
            r = (y - (A @ mm.ravel()).reshape(y.shape))[sl]
            valid = ~np.isnan(m)[sl]
            e = np.mean(r[valid] ** 2)
            if e < best:
                best, bd = e, (dy, dx)
    return bd


_PYR = {}


def phase_transport(prev, d):
    """Move `prev` by d by advancing each band's phase by omega_s * (d . dir_k)."""
    P = _PYR.setdefault(prev.shape, Pyramid(prev.shape, 3, 4))
    p = P.analyse(prev)
    dy, dx = d
    for s in range(P.S):
        for k in range(P.K):
            proj = dx * P.dirs[k, 0] + dy * P.dirs[k, 1]
            p["c"][s, k] = p["c"][s, k] * np.exp(-1j * P.omega[s] * proj)
    # residual bands are moved exactly (they carry no orientation phase)
    p["high"] = shift(p["high"], d, 0.0)
    p["low"] = shift(p["low"], d, float(p["low"].mean()))
    return P.synthesise(p)


def trust(y, mu, A, sigma, e_fog, margin=4):
    r = (y - (A @ mu.ravel()).reshape(y.shape))[margin:-margin, margin:-margin].ravel()
    v1 = sigma ** 2 + e_fog
    v2 = 2 * np.var(y) + sigma ** 2
    llr = np.sum(-.5 * r ** 2 / v1 - .5 * np.log(v1) + .5 * r ** 2 / v2 + .5 * np.log(v2))
    return float(1 / (1 + np.exp(-np.clip(llr, -50, 50))))


def run_paper(frames_y, solver, A, sigma, beta, e_fog, use_trust=True, transport="pixel",
              history=None):
    """Reconstruct a sequence with memory. `solver(y, mem_w, mem_mu)` reconstructs one frame.
    `history`: optional (x_prev, pi_prev) to start from (the foreign-history control).
    Returns reconstructions, trusts, motions."""
    xs, trusts, motions = [], [], []
    x_prev, pi_prev = (None, None) if history is None else history
    for y in frames_y:
        if x_prev is None:
            x = solver(y, None, None)
            pi = np.ones_like(y)
            trusts.append(np.nan); motions.append(None)
        else:
            d = estimate_motion(x_prev, y, A)
            if transport == "phase":
                mu = phase_transport(x_prev, d)
                valid = shift(np.ones_like(y), d, 0.0)
                mu = np.where(valid > 0, mu, float(x_prev.mean()))
            else:
                mu = shift(x_prev, d, float(x_prev.mean()))
            pim = shift(pi_prev, d, 0.0)
            tr = trust(y, mu, A, sigma, e_fog) if use_trust else 1.0
            x = solver(y, beta * tr * pim, mu)
            pi = pim + 1
            trusts.append(tr); motions.append(d)
        xs.append(x)
        x_prev, pi_prev = x, pi
    return xs, trusts, motions
