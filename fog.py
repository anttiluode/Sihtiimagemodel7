"""The fog: Sihti's own purify step, applied `depth` times, plus sensor noise.

purify(x) = one heat-diffusion step on the pixel lattice (5-point Laplacian,
edge-replicated borders = Neumann, dt = 0.25). Copied unchanged from
anttiluode/SihtiImageModel2 sihti2/sieve.py.

The observation is the Sihti *core* at `depth`:  y = purify^depth(x) + noise.
Inverting it means putting back the residue bands r_k that the sieve removed.
NOTE: at dt = 0.25 this "fog" is a band-stop, not a blur (see FOGS below).

Because the Neumann 5-point Laplacian is a symmetric matrix, A = purify^depth
is symmetric too (A^T = A, tested). On 32x32 images A is a dense 1024x1024
matrix, so every linear solve in this repository is exact.
"""
import numpy as np

DEPTHS = (0, 1, 4, 16, 64)          # Sihti's octave depths (for the band metrics)


def purify(x, dt=0.25):
    """One diffusion step. x: (..., H, W). Unchanged from SihtiImageModel2."""
    dt = x.dtype.type(dt)
    p = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    lap = (p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2]
           + p[..., 1:-1, 2:] - 4.0 * x)
    return x + dt * lap


def purify_n(x, n, dt=0.25):
    for _ in range(n):
        x = purify(x, dt)
    return x


def sieve(x, depths=DEPTHS):
    """Fine -> coarse residue bands, last = core; sum(bands) == x exactly."""
    x = np.asarray(x, float)
    states, cur, step = [], x, 0
    for d in depths:
        while step < d:
            cur = purify(cur)
            step += 1
        states.append(cur)
    bands = [states[k] - states[k + 1] for k in range(len(states) - 1)]
    bands.append(states[-1])
    return bands


def fog_matrix(depth, shape=(32, 32), dt=0.25):
    """Dense matrix of purify^depth acting on flattened images."""
    h, w = shape
    eye = np.eye(h * w).reshape(h * w, h, w)
    return purify_n(eye, depth, dt).reshape(h * w, h * w).T.copy()


def observe(x, depth, sigma, rng, dt=0.25):
    """y = purify^depth(x) + N(0, sigma^2). Only y ever reaches a method."""
    return purify_n(np.asarray(x, float), depth, dt) + sigma * rng.standard_normal(np.shape(x))


# Two fogs. Away from borders purify(dt=0.25) replaces a pixel by the mean of its
# four neighbours, so its response per step is (cos kx + cos ky)/2: it kills the
# middle frequencies but PASSES the checkerboard corner (|-0.995|^8 = 0.96 for the
# highest 32x32 Neumann mode). It is a band-stop, not a blur (found by Astra).
# "heat" is the same Laplacian integrated over the same time (8 x 0.25 = 2) with
# small steps (32 x 1/16): a true low-pass, the checkerboard falls to ~1e-10.
FOGS = {"sihti": dict(depth=8, dt=0.25), "heat": dict(depth=32, dt=1 / 16),
        "sihti16": dict(depth=16, dt=0.25)}
