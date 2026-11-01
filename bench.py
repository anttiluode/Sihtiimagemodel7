"""Shared machinery: fit the learned parts once, run every arm on the same observations."""
import json
import pickle
from pathlib import Path

import numpy as np

from fog import fog_matrix, observe
from models import GaussianPrior, fit_dictionary, CSC, IndependentPatches, EPLL
from metrics import score

ROOT = Path(__file__).resolve().parent
ARMS = ("fogged", "wiener", "wiener_sample", "independent", "no_inhibition", "epll", "csc")


def load_params(path=ROOT / "params.json", fog="sihti"):
    """Parameters for one fog: the top level is the Sihti fog; other fogs override."""
    p = json.loads(Path(path).read_text())
    q = {k: v for k, v in p.items() if k not in ("heat",)}
    if fog != "sihti" and fog in p:
        q.update(p[fog])
    return q


class Bench:
    def __init__(self, train, depth, sigma, cache=None, dt=0.25):
        self.depth, self.sigma, self.dt = depth, sigma, dt
        self.A = fog_matrix(depth, dt=dt)
        blob = None
        if cache and Path(cache).exists():
            blob = pickle.loads(Path(cache).read_bytes())
        if blob is None:
            blob = dict(gauss=GaussianPrior(train), D=fit_dictionary(train))
            blob["epll"] = EPLL(train, self.A)
            if cache:
                Path(cache).parent.mkdir(parents=True, exist_ok=True)
                Path(cache).write_bytes(pickle.dumps(blob))
        self.gauss, self.D, self.ep = blob["gauss"], blob["D"], blob["epll"]
        self.ep.A, self.ep.AtA = self.A, self.A.T @ self.A           # same states, this fog
        self.csc = CSC(self.D, self.A)
        self.ind = IndependentPatches(self.D, depth, dt=dt)

    def observe_set(self, truths, seed):
        rng = np.random.default_rng(seed)
        return [observe(t, self.depth, self.sigma, rng, self.dt) for t in truths]

    def run(self, Y, p, arms=ARMS, seed=0):
        rng = np.random.default_rng(seed + 7)
        s, A = self.sigma, self.A
        out = {}
        W = [self.gauss.wiener(y, A, s) for y in Y]
        for arm in arms:
            if arm == "fogged":
                out[arm] = [np.asarray(y) for y in Y]
            elif arm == "wiener":
                out[arm] = W
            elif arm == "wiener_sample":
                out[arm] = [self.gauss.sample(y, A, s, rng) for y in Y]
            elif arm == "independent":
                out[arm] = [self.ind.solve(y, p["ind_lam"]) for y in Y]
            elif arm == "no_inhibition":
                out[arm] = [self.csc.solve_no_inhibition(y, p["noinh_lam"], p["csc_gamma"], p["noinh_scale"])
                            for y in Y]
            elif arm == "epll":
                b = tuple(p["epll_beta_scale"] * v for v in (1, 4, 8, 16, 32, 64))
                out[arm] = [self.ep.solve(y, s, w, betas=b) for y, w in zip(Y, W)]
            elif arm == "csc_converged":
                out[arm] = [self.csc.solve(y, p["csc_lam"], p["csc_gamma"], iters=1600) for y in Y]
            elif arm == "csc":
                out[arm] = [self.csc.solve(y, p["csc_lam"], p["csc_gamma"], iters=p["csc_iters"])
                            for y in Y]
        return out


def score_all(out, truths):
    return {arm: score(v, truths) for arm, v in out.items()}


# ------------------------------------------------------------------ camera pans
def pans(photos, n_per_photo, frames, seed, scale=.35):
    """Integer camera pans over whole photographs: list of (frames, offsets)."""
    from skimage.transform import rescale
    from data import _photo
    rng = np.random.default_rng(seed)
    seqs = []
    for name in photos:
        img = rescale(_photo(name), scale, anti_aliasing=True)
        h, w = img.shape
        made = 0
        while made < n_per_photo:
            vx = int(rng.choice([-2, -1, 1, 2]))
            vy = int(rng.integers(-1, 2))
            span_x, span_y = abs(vx) * (frames - 1), abs(vy) * (frames - 1)
            x0 = int(rng.integers(span_x if vx < 0 else 0, w - 32 - (span_x if vx > 0 else 0)))
            y0 = int(rng.integers(span_y if vy < 0 else 0, h - 32 - (span_y if vy > 0 else 0)))
            offs = [(y0 + t * vy, x0 + t * vx) for t in range(frames)]
            fr = np.array([img[a:a + 32, b:b + 32] for a, b in offs], np.float32)
            if fr.std() < .05:
                continue
            seqs.append((fr, offs))
            made += 1
    return seqs


def flicker(recons, offs, margin=4):
    """Mean temporal std of the reconstruction over world pixels that every frame
    sees at least `margin` px inside its border (the truth's std there is 0)."""
    ys = [o[0] for o in offs]
    xs = [o[1] for o in offs]
    Y0, Y1 = max(ys) + margin, min(ys) + 32 - margin
    X0, X1 = max(xs) + margin, min(xs) + 32 - margin
    if Y1 <= Y0 or X1 <= X0:
        return np.nan
    stack = np.array([r[Y0 - a:Y1 - a, X0 - b:X1 - b] for r, (a, b) in zip(recons, offs)])
    return float(stack.std(0).mean())
