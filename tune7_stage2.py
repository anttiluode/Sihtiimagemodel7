"""Stage 2 of Model 7 validation tuning (written after stage 1, before any test run).

Stage 1 chose every gate by validation MSE, and both gates went to the weakest corner of their
grids: MSE barely rewards silence, so MSE-selection turns S1/S2 into tests of almost-no-gate.
The gates' claim is "more silence at no cost", so the selection rule now matches the claim,
identically for both gates:
    choose the setting with the LOWEST flat-region spurious-edge rate among those whose
    validation MSE <= 1.01 x CSC's and whose recall >= CSC's - 0.02.
(If none qualifies, the gate is off: w = 1.) Grids widened on both sides.
The memory weight beta stays selected by validation MSE; its grid extended below its edge.
"""
import json
import time

import numpy as np

from bench import Bench, ROOT, load_params, pans
from data import natural
from gate import GatedCSC, Evidence, footprint_pool
from metrics import score
from paper import run_paper
from run7 import silence
from tune7 import N_VAL, VAL_PAN_PHOTOS

GRIDS = dict(pc=dict(t=(0.5, 1, 2), kappa=(1, 2, 4, 8)),
             energy=dict(t=(1, 2, 4, 8), kappa=(1, 2, 4, 8)))
BETA2 = (0.003, 0.01, 0.03)


def main():
    t0 = time.time()
    params = json.loads((ROOT / "params.json").read_text())
    m7 = params["model7"]
    log = m7["log"] + ["--- stage 2 (tune7_stage2.py): claim-matched gate selection, beta extended ---"]
    train, val, _ = natural()
    V = val[:N_VAL]
    cache = ROOT / "runs" / "fit_natural.pkl"
    for fog in ("heat", "sihti"):
        p = load_params(fog=fog)
        b = Bench(train, p["depth"], p["sigma"], cache=cache, dt=p.get("dt", 0.25))
        g, ev = GatedCSC(b.D, b.A), Evidence(b.gauss, b.A, p["sigma"])
        Y = b.observe_set(V, seed=11)
        W = [b.gauss.wiener(y, b.A, p["sigma"]) for y in Y]
        base = [g.solve_gated(y, p["csc_lam"], p["csc_gamma"], iters=p["csc_iters"]) for y in Y]
        s0 = score(base, V); s0.update(silence(base, V))
        log.append(f"[{fog}] csc: val mse {s0['mse']:.6f} flat {s0['flat_rate']:.4f} recall {s0['recall']:.3f}")
        for kind, grid in GRIDS.items():
            ok = []
            for t in grid["t"]:
                for k in grid["kappa"]:
                    rec = []
                    for y, xw in zip(Y, W):
                        e = ev.pc(xw, t) if kind == "pc" else ev.energy(xw, t)
                        rec.append(g.solve_gated(y, p["csc_lam"], p["csc_gamma"], iters=p["csc_iters"],
                                                 w=1 + k * (1 - footprint_pool(e))))
                    s = score(rec, V); s.update(silence(rec, V))
                    qual = s["mse"] <= 1.01 * s0["mse"] and s["recall"] >= s0["recall"] - .02
                    log.append(f"[{fog}] {kind} t={t} kappa={k}: val mse {s['mse']:.6f} flat {s['flat_rate']:.4f} "
                               f"recall {s['recall']:.3f}{'  (qualifies)' if qual else ''}")
                    if qual:
                        ok.append((s["flat_rate"], t, k))
            if ok:
                _, t_b, k_b = min(ok)
            else:
                t_b, k_b = grid["t"][0], 0
                log.append(f"[{fog}] {kind}: no setting qualifies -> gate off")
            m7[fog][f"{kind}_t"], m7[fog][f"{kind}_kappa"] = t_b, k_b
            log.append(f"[{fog}] {kind} chosen t={t_b} kappa={k_b}")
            if k_b == max(grid["kappa"]) or t_b in (min(grid["t"]), max(grid["t"])):
                log.append(f"[{fog}] NOTE: {kind} choice on the edge of its grid")
        if fog == "heat":
            seqs = pans(VAL_PAN_PHOTOS, 2, 6, seed=5)
            solver = lambda y, mw, mu: g.solve_gated(y, p["csc_lam"], p["csc_gamma"], mem_w=mw,
                                                     mem_mu=mu, iters=p["csc_iters"])
            r, pf = {}, []
            for beta in BETA2:
                errs = []
                for i, (fr, offs) in enumerate(seqs):
                    Yf = b.observe_set(fr, seed=100 + i)
                    xs, _, _ = run_paper(Yf, solver, b.A, p["sigma"], beta, m7["heat"]["e_fog"])
                    errs += [np.mean((x - t_) ** 2) for x, t_ in zip(xs[3:], fr[3:])]
                    if beta == BETA2[0]:
                        pf += [np.mean((solver(y, None, None) - t_) ** 2) for y, t_ in zip(Yf[3:], fr[3:])]
                r[beta] = float(np.mean(errs))
                log.append(f"[heat] paper beta={beta}: val-pan mse (frames 3-5) {r[beta]:.6f}")
            log.append(f"[heat] per-frame csc on the same val pans: {np.mean(pf):.6f}")
            prev = min((v, k) for k, v in [(0.03, 0.001611), (0.1, 0.001701), (0.3, 0.001792)])
            best = min(r, key=r.get)
            m7["heat"]["beta"] = best if r[best] <= prev[0] else prev[1]
            log.append(f"[heat] beta chosen {m7['heat']['beta']} (stages 1+2 combined)")
            if m7["heat"]["beta"] == min(BETA2):
                log.append("[heat] NOTE: beta on the low edge of its grid")
    m7["log"] = log
    m7["seconds"] = m7.get("seconds", 0) + round(time.time() - t0)
    params["model7"] = m7
    (ROOT / "params.json").write_text(json.dumps(params, indent=2))
    print("\n".join(log[-60:]))


if __name__ == "__main__":
    main()
