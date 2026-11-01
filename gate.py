"""Model 6's competing atoms, given a reason to stay silent, and a memory.

GatedCSC solves Model 6's objective with two additions:

  1/2||y - A x||^2 + gamma/2 ||H c||^2 + lam * sum_j  w_j |a_j|      x = c + S a
                                       + 1/2 sum_p  m_p (x_p - mu_p)^2     (memory, optional)

w_j  the silence weight of atom j's position. It comes from EVIDENCE, computed
     once from the Wiener estimate of the observation (never from CSC's own
     output, so the gate cannot confirm its own hallucinations):
        PC gate      e = phase congruency of the Wiener estimate (Kovesi, threshold T)
        energy gate  e = local oriented energy / (energy + tau)
     e is max-pooled over each atom's 8x8 footprint and w = 1 + kappa (1 - e):
     an atom where the evidence shows no feature costs up to (1 + kappa) times more.
     Same pyramid, same pooling, same knobs for both gates: only phase vs energy differs.
m_p  per-pixel memory weight (the living paper: beta * trust * transported precision).
"""
import numpy as np
from scipy.ndimage import maximum_filter

from models import CSC, soft, P as PATCH
from pyramid import Pyramid


def footprint_pool(evidence):
    """(32,32) evidence -> (25,25) max over each 8x8 atom footprint (top-left indexed)."""
    m = maximum_filter(evidence, size=PATCH, origin=(-(PATCH // 2), -(PATCH // 2)), mode="nearest")
    n = evidence.shape[0] - PATCH + 1
    return m[:n, :n]


class Evidence:
    """Noise-calibrated PC and energy maps of a Wiener estimate."""

    def __init__(self, gauss, A, sigma, shape=(32, 32), seed=0):
        self.P = Pyramid(shape, 3, 4)
        self.gauss, self.A, self.sigma = gauss, A, sigma
        rng = np.random.default_rng(seed)
        # what the Wiener estimate of PURE NOISE looks like through the pyramid:
        # it sets the units of both thresholds (so neither gate gets a hand-picked scale)
        E, En = [], []
        for _ in range(16):
            xn = gauss.wiener(sigma * rng.standard_normal(shape), A, sigma) - gauss.wiener(
                np.zeros(shape), A, sigma)
            c = self.P.analyse(xn)["c"]
            E.append(np.abs(c.sum(0)).ravel())
            En.append((np.abs(c) ** 2).sum((0, 1)).ravel())
        self.T0 = float(np.median(np.concatenate(E)))
        self.tau0 = float(np.median(np.concatenate(En)))

    def maps(self, xw):
        return dict(pc=self.P, xw=xw)

    def pc(self, xw, t):
        return self.P.congruence(xw, T=t * self.T0)

    def energy(self, xw, t):
        e = self.P.energy(xw)
        return e / (e + t * self.tau0)


class GatedCSC(CSC):
    def solve_gated(self, y, lam, gamma, w=None, mem_w=None, mem_mu=None, iters=400):
        A = self.A
        yf = np.ravel(y)
        W = np.ones(self.cshape) if w is None else np.broadcast_to(w, self.cshape)
        mw = None if mem_w is None else np.ravel(mem_w)
        mu = None if mem_mu is None else np.ravel(mem_mu)
        L = self.LA * (1 + self.LS) + gamma * self.LH
        if mw is not None:
            L = L + mw.max() * (1 + self.LS)
        c = np.array(y, float)
        a = np.zeros(self.cshape)
        cz, az, t = c.copy(), a.copy(), 1.
        for _ in range(iters):
            x = (cz + self.S(az)).ravel()
            g = A.T @ (A @ x - yf)
            if mw is not None:
                g = g + mw * (x - mu)
            gc = (g + gamma * (self.HtH @ cz.ravel())).reshape(self.shape)
            ga = self.St(g.reshape(self.shape))
            c_new = cz - gc / L
            a_new = soft(az - ga / L, lam * W / L)
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            cz = c_new + (t - 1) / t_new * (c_new - c)
            az = a_new + (t - 1) / t_new * (a_new - a)
            c, a, t = c_new, a_new, t_new
        return c + self.S(a)
