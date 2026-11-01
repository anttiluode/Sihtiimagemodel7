"""What 'in focus' is allowed to mean here.

Astra's correction, adopted before any test run: fine-band energy near 1.0x
does NOT establish correct detail (wrong edges and noise can have the right
energy). So every arm is scored on five things, and only the first four can
pass a gate:

  mse          pixel error (all pixels)
  edge_f       edge POSITION: Canny edges of the result vs Canny edges of the
               truth, F-score with a 1.5 px tolerance (pooled over the set)
  orient_deg   edge ORIENTATION: structure-tensor angle error at the truth's
               edge pixels, weighted by the truth's gradient energy
  flicker      TEMPORAL consistency on camera pans (see bench.py)
  band_ratio   fine-band energy (Sihti r0 + r1) of the result / of the truth:
               reported, never a success criterion
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import canny, structure_tensor

from fog import sieve

EDGE_SIGMA = 1.0
EDGE_LOW, EDGE_HIGH = 0.04, 0.08      # absolute gradient thresholds, fixed for all arms
TOL = 1.5


def edges(x):
    return canny(np.asarray(x, float), sigma=EDGE_SIGMA,
                 low_threshold=EDGE_LOW, high_threshold=EDGE_HIGH)


def edge_counts(x, t):
    """(matched_pred, n_pred, matched_true, n_true) within TOL px."""
    ex, et = edges(x), edges(t)
    dt_true = distance_transform_edt(~et) if et.any() else np.full(et.shape, np.inf)
    dt_pred = distance_transform_edt(~ex) if ex.any() else np.full(ex.shape, np.inf)
    return (int((dt_true[ex] <= TOL).sum()), int(ex.sum()),
            int((dt_pred[et] <= TOL).sum()), int(et.sum()))


def f_score(counts):
    c = np.asarray(counts, float).sum(0)
    p = c[0] / max(c[1], 1)
    r = c[2] / max(c[3], 1)
    return 0. if p + r == 0 else 2 * p * r / (p + r)


def orientation(x):
    Arr, Arc, Acc = structure_tensor(np.asarray(x, float), sigma=1.0, order="rc")
    return .5 * np.arctan2(2 * Arc, Arr - Acc), Arr + Acc


def orient_err(x, t):
    """(sum weighted |angle error| in degrees, sum weights) at truth edges."""
    et = edges(t)
    if not et.any():
        return 0., 0.
    th_x, _ = orientation(x)
    th_t, energy = orientation(t)
    d = np.abs(th_x - th_t) % np.pi
    d = np.minimum(d, np.pi - d)
    w = energy[et]
    return float((np.degrees(d[et]) * w).sum()), float(w.sum())


def band_energy(x):
    b = sieve(x)
    return float((b[0] ** 2).sum() + (b[1] ** 2).sum())


def score(recons, truths):
    """Set-level scores for one arm."""
    recons, truths = np.asarray(recons, float), np.asarray(truths, float)
    mse = ((recons - truths) ** 2).reshape(len(truths), -1).mean(1)
    counts = [edge_counts(x, t) for x, t in zip(recons, truths)]
    oe = np.array([orient_err(x, t) for x, t in zip(recons, truths)])
    be = sum(band_energy(x) for x in recons) / sum(band_energy(t) for t in truths)
    return dict(mse=float(mse.mean()), mse_per_image=mse.tolist(),
                edge_f=float(f_score(counts)),
                orient_deg=float(oe[:, 0].sum() / max(oe[:, 1].sum(), 1e-12)),
                band_ratio=float(be))
