import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fog import sieve
from metrics import edges

LABELS = dict(truth="truth", fogged="fogged", wiener="Wiener (linear)",
              wiener_sample="Wiener sample", independent="independent patches",
              no_inhibition="atoms, no inhibition", epll="EPLL (Model 4 states)", csc="CSC (competing atoms)")


def _fine(x):
    b = sieve(x)
    return b[0] + b[1]


def comparison_figure(truths, out, path, picks=None):
    arms = ["truth"] + [a for a in ("fogged", "wiener", "wiener_sample", "independent", "no_inhibition", "epll", "csc") if a in out]
    picks = picks if picks is not None else np.linspace(0, len(truths) - 1, 4).astype(int)
    fig, ax = plt.subplots(3 * len(picks), len(arms), figsize=(1.9 * len(arms), 1.9 * 3 * len(picks)))
    for r, i in enumerate(picks):
        t = truths[i]
        lim = np.abs(_fine(t)).max() + 1e-9
        for c, arm in enumerate(arms):
            x = t if arm == "truth" else out[arm][i]
            a0, a1, a2 = ax[3 * r, c], ax[3 * r + 1, c], ax[3 * r + 2, c]
            a0.imshow(np.clip(x, 0, 1), cmap="gray", vmin=0, vmax=1)
            a1.imshow(_fine(x), cmap="RdBu_r", vmin=-lim, vmax=lim)
            ov = np.dstack([edges(t) * 1., edges(x) * 1., np.zeros_like(t)])
            a2.imshow(ov)
            if arm != "truth":
                a0.set_title(f"{LABELS[arm]}\nmse {((x - t) ** 2).mean():.5f}", fontsize=6)
            else:
                a0.set_title("truth", fontsize=6)
            for a in (a0, a1, a2):
                a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a1.set_ylabel("r0+r1", fontsize=6)
                a2.set_ylabel("edges\nred=truth\ngreen=result\nyellow=both", fontsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def temporal_figure(frames, Y, out, offs, path, margin=4):
    ys, xs = [o[0] for o in offs], [o[1] for o in offs]
    Y0, Y1 = max(ys) + margin, min(ys) + 32 - margin
    X0, X1 = max(xs) + margin, min(xs) + 32 - margin
    rows = [("truth", list(frames))] + [(a, out[a]) for a in ("wiener", "wiener_sample", "independent", "csc")]
    fig, ax = plt.subplots(len(rows), len(frames) + 1, figsize=(1.6 * (len(frames) + 1), 1.7 * len(rows)))
    for r, (name, seq) in enumerate(rows):
        crops = np.array([s[Y0 - a:Y1 - a, X0 - b:X1 - b] for s, (a, b) in zip(seq, offs)])
        for t in range(len(frames)):
            ax[r, t].imshow(np.clip(crops[t], 0, 1), cmap="gray", vmin=0, vmax=1)
            ax[r, t].set_xticks([]); ax[r, t].set_yticks([])
        ax[r, -1].imshow(crops.std(0), cmap="magma", vmin=0, vmax=.08)
        ax[r, -1].set_xticks([]); ax[r, -1].set_yticks([])
        ax[r, 0].set_ylabel(LABELS.get(name, name), fontsize=6)
        if r == 0:
            for t in range(len(frames)):
                ax[r, t].set_title(f"frame {t}", fontsize=6)
            ax[r, -1].set_title("temporal std", fontsize=6)
        else:
            ax[r, -1].set_title(f"flicker {crops.std(0).mean():.4f}", fontsize=6)
    fig.suptitle("same world region, every frame reconstructed on its own", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def atoms_figure(D, path):
    k = len(D)
    n = int(np.ceil(np.sqrt(k)))
    fig, ax = plt.subplots(n, n, figsize=(n * .6, n * .6))
    for i, a in enumerate(ax.ravel()):
        a.set_xticks([]); a.set_yticks([])
        if i < k:
            m = np.abs(D[i]).max()
            a.imshow(D[i], cmap="RdBu_r", vmin=-m, vmax=m)
    fig.suptitle("the 64 atoms (learned on clean training patches)", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
