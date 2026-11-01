"""Every way of seeing back through the fog that this repository compares.

All methods receive only the observation y (and the fog operator A, which is
known); none ever sees the clean image. Everything learned here is learned
from *training* images only.

Arms
----
wiener        linear focus: exact LMMSE under a full-image Gaussian learned from
              training crops (Ledoit-Wolf covariance). The optimal *linear*
              unblur; mostly a set of per-mode attenuations.
wiener_sample the same posterior, but a sample instead of the mean
              (SihtiImageModel2's "sampled residue" failure mode: right amount
              of detail, wrong places).
independent   the same learned dictionary of 8x8 atoms, but every overlapping
              patch decides alone (sparse code of its own fogged patch under a
              local copy of the fog), then the decisions are averaged.
csc           the same atoms placed at every position of the image, competing
              for ONE shared reconstruction under the true fog
              (convolutional sparse coding, FISTA; Rozell's LCA solves the same
              objective with neural dynamics). Overlapping atoms inhibit each
              other through the shared residual: this is the coordination.
epll          Model 4's engine (a 20-state patch Gaussian mixture, hard state
              choice per patch) applied to the fog with EPLL's half-quadratic
              schedule (Zoran & Weiss 2011). Coordinated by its global x-step.
"""
import numpy as np
from scipy.signal import fftconvolve

from fog import fog_matrix

P = 8   # patch / atom size


# ------------------------------------------------------------------ patches
def patch_index(shape=(32, 32), p=P):
    """(n_patches, p*p) flat pixel indices of every overlapping p x p patch."""
    h, w = shape
    base = (np.arange(p)[:, None] * w + np.arange(p)[None, :]).ravel()
    tl = (np.arange(h - p + 1)[:, None] * w + np.arange(w - p + 1)[None, :]).ravel()
    return tl[:, None] + base[None, :]


def sample_patches(images, n, rng, p=P):
    imgs = np.asarray(images, float)
    k = rng.integers(0, len(imgs), n)
    yy = rng.integers(0, imgs.shape[1] - p + 1, n)
    xx = rng.integers(0, imgs.shape[2] - p + 1, n)
    return np.stack([imgs[a, b:b + p, c:c + p].ravel() for a, b, c in zip(k, yy, xx)])


# ------------------------------------------------------------------ Gaussian
class GaussianPrior:
    """Full-image Gaussian (mean + Ledoit-Wolf covariance) and its exact posterior."""

    def __init__(self, train):
        from sklearn.covariance import LedoitWolf
        X = np.asarray(train, float).reshape(len(train), -1)
        self.shape = np.shape(train)[1:]
        self.mu = X.mean(0)
        self.C = LedoitWolf().fit(X).covariance_
        self._cache = {}

    def _gain(self, A, sigma):
        import hashlib
        key = (hashlib.sha1(np.ascontiguousarray(A).tobytes()).hexdigest(), float(sigma))
        if key not in self._cache:
            C = self.C
            S = A @ C @ A.T + sigma ** 2 * np.eye(len(C))
            K = np.linalg.solve(S, A @ C).T                 # C A^T S^-1
            Pc = C - K @ A @ C
            Pc = (Pc + Pc.T) / 2
            ev, U = np.linalg.eigh(Pc)
            L = U * np.sqrt(np.clip(ev, 0, None))
            self._cache[key] = (K, L)
        return self._cache[key]

    def wiener(self, y, A, sigma):
        K, _ = self._gain(A, sigma)
        yf = np.ravel(y)
        return (self.mu + K @ (yf - A @ self.mu)).reshape(self.shape)

    def sample(self, y, A, sigma, rng):
        K, L = self._gain(A, sigma)
        z = rng.standard_normal(L.shape[1])
        return self.wiener(y, A, sigma) + (L @ z).reshape(self.shape)


# ------------------------------------------------------------------ dictionary
def fit_dictionary(train, n_atoms=64, n_patches=40000, alpha=0.05, seed=0):
    """Zero-mean, unit-norm 8x8 atoms learned from clean training patches."""
    from sklearn.decomposition import MiniBatchDictionaryLearning
    rng = np.random.default_rng(seed)
    X = sample_patches(train, n_patches, rng)
    X = X - X.mean(1, keepdims=True)
    X = X[np.linalg.norm(X, axis=1) > 1e-3]
    dl = MiniBatchDictionaryLearning(n_components=n_atoms, alpha=alpha, batch_size=256,
                                     max_iter=30, random_state=seed)
    D = dl.fit(X).components_
    D = D - D.mean(1, keepdims=True)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    return D.reshape(n_atoms, P, P)


def _power(op, x0, iters=60):
    x = x0 / np.linalg.norm(x0)
    lam = 0.
    for _ in range(iters):
        x = op(x)
        lam = np.linalg.norm(x)
        x = x / lam
    return lam


def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.)


# ------------------------------------------------------------------ CSC
class CSC:
    """min_{c,a}  1/2 ||y - A(c + S a)||^2 + gamma/2 ||H c||^2 + lam * sum|a|

    S a  = sum_k  d_k (*) a_k   (every atom at every position, 'full' conv)
    c    = the smooth core; H = I - A penalises any detail in it, so every
           piece of detail the fog removed has to be paid for in atoms.
    Solved by FISTA. Rozell's LCA is the neural dynamics of the same objective:
    its lateral inhibition between two atoms is exactly the overlap of their
    fogged footprints, <A S e_i, A S e_j>.
    """

    def __init__(self, D, A, shape=(32, 32)):
        self.D = np.asarray(D, float)
        self.Df = self.D[:, ::-1, ::-1].copy()
        self.A = A
        self.shape = shape
        n = shape[0] * shape[1]
        self.H = np.eye(n) - A
        self.HtH = self.H.T @ self.H
        self.cshape = (len(D), shape[0] - P + 1, shape[1] - P + 1)
        rng = np.random.default_rng(0)
        self.LA = _power(lambda v: A.T @ (A @ v), rng.standard_normal(n))
        self.LH = _power(lambda v: self.HtH @ v, rng.standard_normal(n))
        self.LS = _power(lambda a: self.St(self.S(a)), rng.standard_normal(self.cshape))

    def S(self, a):
        return fftconvolve(a, self.D, mode="full", axes=(1, 2)).sum(0)

    def St(self, r):
        return fftconvolve(np.broadcast_to(r, (len(self.D),) + r.shape), self.Df,
                           mode="valid", axes=(1, 2))

    def objective(self, y, c, a, lam, gamma):
        x = c + self.S(a)
        r = self.A @ x.ravel() - np.ravel(y)
        return .5 * r @ r + gamma / 2 * np.sum((self.H @ c.ravel()) ** 2) + lam * np.abs(a).sum()

    def solve(self, y, lam, gamma, iters=400, c0=None, return_parts=False):
        A = self.A
        yf = np.ravel(y)
        L = self.LA * (1 + self.LS) + gamma * self.LH          # bound on the Lipschitz constant
        c = np.array(y, float) if c0 is None else np.array(c0, float)
        a = np.zeros(self.cshape)
        cz, az, t = c.copy(), a.copy(), 1.
        for _ in range(iters):
            x = cz + self.S(az)
            g = A.T @ (A @ x.ravel() - yf)
            gc = (g + gamma * (self.HtH @ cz.ravel())).reshape(self.shape)
            ga = self.St(g.reshape(self.shape))
            c_new = cz - gc / L
            a_new = soft(az - ga / L, lam / L)
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            cz = c_new + (t - 1) / t_new * (c_new - c)
            az = a_new + (t - 1) / t_new * (a_new - a)
            c, a, t = c_new, a_new, t_new
        x = c + self.S(a)
        return (x, c, a) if return_parts else x

    # ---- the tight control: same atoms, operator, core and sparsity; competition OFF
    def _core(self, y, gamma):
        key = float(gamma)
        if getattr(self, "_core_key", None) != key:
            self._core_M = np.linalg.inv(self.A.T @ self.A + gamma * self.HtH) @ self.A.T
            self._core_key = key
        return (self._core_M @ np.ravel(y)).reshape(self.shape)

    def footprint_norms(self):
        """||A S e_j||^2 for every atom j at every position (exact, cached)."""
        if not hasattr(self, "_fn"):
            K, h, w = self.cshape
            out = np.empty(self.cshape)
            for k in range(K):
                cols = np.zeros((h * w,) + self.shape)
                for i in range(h):
                    for j in range(w):
                        cols[i * w + j, i:i + P, j:j + P] = self.D[k]
                F = cols.reshape(h * w, -1) @ self.A.T
                out[k] = (F ** 2).sum(1).reshape(h, w)
            self._fn = out
        return self._fn

    def solve_no_inhibition(self, y, lam, gamma, scale=1.0):
        """Every atom fits the SAME shared residual of the smooth core ALONE
        (LCA with the lateral-inhibition term deleted: u = b, a = soft(u)).
        c0 = argmin 1/2||y - A c||^2 + gamma/2 ||H c||^2,   r = y - A c0,
        a_j = scale * soft(<A S e_j, r>, lam) / ||A S e_j||^2."""
        c0 = self._core(y, gamma)
        r = np.ravel(y) - self.A @ c0.ravel()
        b = self.St((self.A.T @ r).reshape(self.shape))
        a = scale * soft(b, lam) / self.footprint_norms()
        return c0 + self.S(a)


# ------------------------------------------------------------------ independent patches
class IndependentPatches:
    """Same atoms, same sparsity; every overlapping patch decides ALONE.

    Patch p sees only its own fogged pixels y_p and models them with a local
    copy of the fog (purify^depth on an isolated 8x8 patch):
        min_{m,alpha} 1/2 ||y_p - A8 (m 1 + D alpha)||^2 + lam |alpha|_1
    Decisions are then averaged where patches overlap. No patch knows what its
    neighbours chose.
    """

    def __init__(self, D, depth, shape=(32, 32), dt=0.25):
        self.D = np.asarray(D, float).reshape(len(D), -1)      # (K, 64)
        self.A8 = fog_matrix(depth, (P, P), dt)
        self.idx = patch_index(shape)
        self.shape = shape
        self.count = np.bincount(self.idx.ravel(), minlength=shape[0] * shape[1])
        M = np.vstack([np.ones(P * P) / P, self.D])             # mean atom + dictionary
        G = self.A8 @ M.T
        self.M = M
        self.L = np.linalg.eigvalsh(G.T @ G).max()

    def solve(self, y, lam, iters=300):
        Y = np.ravel(y)[self.idx]                               # (n, 64)
        M, A8, L = self.M, self.A8, self.L
        w = np.zeros((len(Y), len(M)))
        z, t = w.copy(), 1.
        mask = np.ones(len(M)); mask[0] = 0                    # the mean is not sparse
        for _ in range(iters):
            R = (z @ M) @ A8.T - Y
            g = (R @ A8) @ M.T
            w_new = z - g / L
            w_new = np.where(mask > 0, soft(w_new, lam / L), w_new)
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            z = w_new + (t - 1) / t_new * (w_new - w)
            w, t = w_new, t_new
        Xp = w @ M
        out = np.bincount(self.idx.ravel(), weights=Xp.ravel(),
                          minlength=self.count.size) / self.count
        return out.reshape(self.shape)


# ------------------------------------------------------------------ EPLL (Model 4's states)
class EPLL:
    """Zoran & Weiss EPLL with a K-state patch Gaussian mixture, on the fog.

    z-step: every patch picks its most likely state under its current value
            (hard choice, as Model 4's local step) and is Wiener-filtered by it;
    x-step: exact solve of  1/s^2 ||A x - y||^2 + beta sum_p ||P_p x - z_p||^2.
    """

    def __init__(self, train, A, n_states=20, n_patches=30000, seed=0, shape=(32, 32)):
        from sklearn.mixture import GaussianMixture
        rng = np.random.default_rng(seed)
        X = sample_patches(train, n_patches, rng)
        X = X - X.mean(1, keepdims=True)
        gm = GaussianMixture(n_states, covariance_type="full", reg_covar=1e-6,
                             random_state=seed, max_iter=200).fit(X)
        self.logw = np.log(gm.weights_)
        ev, U = np.linalg.eigh(gm.covariances_)
        self.ev, self.U = np.clip(ev, 1e-8, None), U                # (K,64), (K,64,64)
        self.A = A
        self.shape = shape
        self.idx = patch_index(shape)
        self.count = np.bincount(self.idx.ravel(), minlength=shape[0] * shape[1])
        self.AtA = A.T @ A

    def zstep(self, x, beta):
        Xp = np.ravel(x)[self.idx]
        m = Xp.mean(1, keepdims=True)
        V = Xp - m
        proj = np.einsum("kij,ni->nkj", self.U, V)                 # (n,K,64)
        var = self.ev[None] + 1 / beta
        ll = -.5 * (proj ** 2 / var).sum(-1) - .5 * np.log(var).sum(-1) + self.logw[None]
        k = ll.argmax(1)
        shrink = (self.ev / (self.ev + 1 / beta))[k]               # (n,64)
        Z = np.einsum("nij,nj->ni", self.U[k], proj[np.arange(len(k)), k] * shrink)
        return Z + m, k

    def solve(self, y, sigma, x0, betas=(1, 4, 8, 16, 32, 64), inner=1):
        x = np.array(x0, float)
        yf = np.ravel(y)
        for b in betas:
            beta = b / sigma ** 2
            for _ in range(inner):
                Z, k = self.zstep(x, beta)
                rhs = self.A.T @ yf / sigma ** 2 + beta * np.bincount(
                    self.idx.ravel(), weights=Z.ravel(), minlength=self.count.size)
                Mx = self.AtA / sigma ** 2 + beta * np.diag(self.count)
                x = np.linalg.solve(Mx, rhs).reshape(self.shape)
        self.states = k
        return x
