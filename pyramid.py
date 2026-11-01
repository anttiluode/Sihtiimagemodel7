"""Undecimated complex steerable pyramid (frequency domain), and local phase.

Sihti's sieve, made oriented and complex. Every coefficient c = a e^{i phi}:
the amplitude a says something is there, the phase phi says where (sub-pixel)
and what kind (step edge: phi = +-pi/2, line: phi = 0 or pi).

Construction (all masks real and even, so analysis and synthesis are each
other's adjoint):
    H(r) = cos(pi/2 * clip(-log2(r/pi), 0, 1))       high-pass, 1 above pi
    L(r) = sin(pi/2 * clip(-log2(r/pi), 0, 1))       low-pass,  H^2 + L^2 = 1
    band s (s = 1..S):  B_s(r) = L(r) L(2r) ... L(2^{s-1} r) * H(2^s r)
    residuals:          high = H(r),  low = L(r) ... L(2^S r)
    orientation k of K: G_k(t) = alpha |cos(t - t_k)|^{K-1},  sum_k G_k^2 = 1
So high^2 + sum_s B_s^2 + low^2 = 1 exactly: a tight frame (tested).

The complex coefficient uses the one-sided mask 2 G_k 1[cos(t - t_k) > 0]:
its real part is the real steerable band, its imaginary part the quadrature
(Hilbert) partner. Undecimated: every band keeps the full image grid, so a
shift of the image by d changes the phase at band (s, k) by about w_s * d_k
(tested), where w_s is the band's centre frequency.
"""
import numpy as np
from math import comb


class Pyramid:
    def __init__(self, shape=(32, 32), n_scales=3, n_orient=4):
        self.shape, self.S, self.K = shape, n_scales, n_orient
        h, w = shape
        fy = np.fft.fftfreq(h)[:, None] * 2 * np.pi
        fx = np.fft.fftfreq(w)[None, :] * 2 * np.pi
        r = np.sqrt(fx ** 2 + fy ** 2)
        t = np.arctan2(fy, fx)
        lr = np.log2(np.maximum(r, 1e-12) / np.pi)

        def H(scale):
            return np.cos(np.pi / 2 * np.clip(-(lr + scale), 0, 1))

        def L(scale):
            return np.sin(np.pi / 2 * np.clip(-(lr + scale), 0, 1))

        self.high = H(0)
        lowprod = np.ones(shape)
        self.bands = []
        for s in range(1, n_scales + 1):
            lowprod = lowprod * L(s - 1)
            self.bands.append(lowprod * H(s))
        self.low = lowprod * L(n_scales)
        self.low[0, 0] = 1.0                                   # DC belongs to the low residual
        self.high[0, 0] = 0.0
        for b in self.bands:
            b[0, 0] = 0.0
        K = n_orient
        alpha = np.sqrt(2 ** (2 * (K - 1)) / (K * comb(2 * K - 2, K - 1)))
        self.tk = np.arange(K) * np.pi / K
        self.G = [alpha * np.abs(np.cos(t - tk)) ** (K - 1) for tk in self.tk]
        self.Gplus = [2 * alpha * np.abs(np.cos(t - tk)) ** (K - 1) * (np.cos(t - tk) > 0)
                      for tk in self.tk]
        # band centre frequencies (energy-weighted mean radius), for phase <-> shift
        self.omega = np.array([(r * b ** 2).sum() / (b ** 2).sum() for b in self.bands])
        self.dirs = np.stack([np.cos(self.tk), np.sin(self.tk)], 1)   # (x, y) unit vectors

    # ------------------------------------------------------------------ analysis
    def analyse(self, x):
        """x (H, W) real -> dict(high, low, c) with c complex (S, K, H, W)."""
        X = np.fft.fft2(np.asarray(x, float))
        c = np.empty((self.S, self.K) + self.shape, complex)
        for s, b in enumerate(self.bands):
            for k, gp in enumerate(self.Gplus):
                c[s, k] = np.fft.ifft2(X * b * gp)
        return dict(high=np.fft.ifft2(X * self.high).real,
                    low=np.fft.ifft2(X * self.low).real, c=c)

    def synthesise(self, p):
        """Exact inverse of analyse (uses only the real parts of c)."""
        Y = np.fft.fft2(p["high"]) * self.high + np.fft.fft2(p["low"]) * self.low
        for s, b in enumerate(self.bands):
            for k, g in enumerate(self.G):
                Y = Y + np.fft.fft2(p["c"][s, k].real) * b * g
        return np.fft.ifft2(Y).real

    # ------------------------------------------------------------------ local phase
    def congruence(self, x, T=0.0, eps=1e-4):
        """Kovesi-style phase congruency per pixel, with noise threshold T.

        For orientation k:  E_k = |sum_s c_{s,k}|  (the scales' complex sum),
                            A_k = sum_s |c_{s,k}|.
        E_k = A_k exactly when every scale has the same phase (a feature sits
        here); random phases (noise) give E_k ~ sqrt(S) * typical |c|.
        PC = sum_k max(E_k - T, 0) / (sum_k A_k + eps)   in [0, 1].
        """
        c = self.analyse(x)["c"]
        E = np.abs(c.sum(0))
        A = np.abs(c).sum(0)
        return np.maximum(E - T, 0).sum(0) / (A.sum(0) + eps)

    def energy(self, x):
        """Local oriented energy: sum over scales and orientations of |c|^2."""
        return (np.abs(self.analyse(x)["c"]) ** 2).sum((0, 1))
