import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import edges


def _off(a):
    a.set_xticks([]); a.set_yticks([])


def gate_figure(truths, Y, W, rec, ev, q, path, picks=(5, 21, 40, 58)):
    cols = ["truth", "fogged", "Wiener", "PC evidence", "energy evidence", "csc", "csc+pc", "csc+energy", "epll"]
    fig, ax = plt.subplots(2 * len(picks), len(cols), figsize=(1.8 * len(cols), 3.6 * len(picks)))
    for r, i in enumerate(picks):
        t = truths[i]
        imgs = {"truth": t, "fogged": Y[i], "Wiener": W[i],
                "PC evidence": ev.pc(W[i], q["pc_t"]), "energy evidence": ev.energy(W[i], q["energy_t"]),
                "csc": rec["csc"][i], "csc+pc": rec["csc+pc"][i], "csc+energy": rec["csc+energy"][i],
                "epll": rec["epll"][i]}
        for c, name in enumerate(cols):
            x = imgs[name]
            a0, a1 = ax[2 * r, c], ax[2 * r + 1, c]
            if "evidence" in name:
                a0.imshow(x, cmap="inferno", vmin=0, vmax=1)
                a1.axis("off")
            else:
                a0.imshow(np.clip(x, 0, 1), cmap="gray", vmin=0, vmax=1)
                a1.imshow(np.dstack([edges(t) * 1., edges(x) * 1., np.zeros_like(t)]))
                if name not in ("truth",):
                    a0.set_title(f"{name}\nmse {np.mean((x - t) ** 2):.5f}", fontsize=6)
            if r == 0 and "evidence" in name:
                a0.set_title(name, fontsize=6)
            if name == "truth":
                a0.set_title("truth", fontsize=6)
            _off(a0); _off(a1)
    fig.suptitle("edge overlays: red = truth only, green = result only (spurious), yellow = both", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def frontier_figure(frontier, sc, path, fog):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for kind, col in (("pc", "#c2410c"), ("energy", "#2563eb")):
        f = frontier[kind]
        ax[0].plot([p["flat_rate"] for p in f], [p["mse"] for p in f], "o-", color=col,
                   label=f"{kind} gate, kappa 1/4/16/32")
        ax[1].plot([p["flat_rate"] for p in f], [p["recall"] for p in f], "o-", color=col)
    for name, mk in (("csc", "s"), ("epll", "^"), ("independent", "v"), ("csc+pc", "*"), ("csc+energy", "P")):
        s = sc[name]
        ax[0].plot(s["flat_rate"], s["mse"], mk, color="k", label=name)
        ax[1].plot(s["flat_rate"], s["recall"], mk, color="k")
    ax[0].set_xlabel("spurious edges in flat regions (rate)"); ax[0].set_ylabel("MSE")
    ax[1].set_xlabel("spurious edges in flat regions (rate)"); ax[1].set_ylabel("edge recall")
    ax[0].legend(fontsize=6)
    fig.suptitle(f"silence vs detail, {fog} fog (report only)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def paper_figure(frames, Y, runs, offs, path, margin=4):
    ys, xs = [o[0] for o in offs], [o[1] for o in offs]
    Y0, Y1 = max(ys) + margin, min(ys) + 32 - margin
    X0, X1 = max(xs) + margin, min(xs) + 32 - margin
    rows = [("truth", list(frames)), ("fogged", list(Y))] + [(k, runs[k]) for k in ("per_frame", "paper", "paper_phase")]
    fig, ax = plt.subplots(len(rows), len(frames) + 1, figsize=(1.6 * (len(frames) + 1), 1.7 * len(rows)))
    for r, (name, seq) in enumerate(rows):
        crops = np.array([s[Y0 - a:Y1 - a, X0 - b:X1 - b] for s, (a, b) in zip(seq, offs)])
        for t in range(len(frames)):
            ax[r, t].imshow(np.clip(crops[t], 0, 1), cmap="gray", vmin=0, vmax=1); _off(ax[r, t])
        ax[r, -1].imshow(crops.std(0), cmap="magma", vmin=0, vmax=.08); _off(ax[r, -1])
        ax[r, 0].set_ylabel(name, fontsize=6)
        ax[r, -1].set_title(f"temporal std {crops.std(0).mean():.4f}", fontsize=6)
    fig.suptitle("one world region through a pan: per-frame vs the living paper", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def worlds_figure(truths, Y, worlds, path, picks=(5, 21, 40, 58)):
    cols = ["truth", "fogged"] + list(worlds)
    fig, ax = plt.subplots(len(picks), len(cols), figsize=(1.9 * len(cols), 1.9 * len(picks)))
    for r, i in enumerate(picks):
        for c, name in enumerate(cols):
            x = truths[i] if name == "truth" else Y[i] if name == "fogged" else worlds[name][i]
            ax[r, c].imshow(np.clip(x, 0, 1), cmap="gray", vmin=0, vmax=1); _off(ax[r, c])
            ax[r, c].set_title(name if name in ("truth", "fogged") else
                               f"{name} world\nmse {np.mean((x - truths[i]) ** 2):.5f}", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
