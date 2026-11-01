"""Validation-only tuning for Model 7. Every knob chosen by validation MSE (the same rule as
Model 6). Model 6's CSC parameters (lam, gamma, iterations) are inherited frozen, per fog.
Writes params.json (Model 6's parameters plus a "model7" block with the full log).
The test photographs (camera, coffee, clock) are never opened.
"""
import json
import time

import numpy as np

from bench import Bench, ROOT, load_params, pans
from data import natural
from gate import GatedCSC, Evidence, footprint_pool
from metrics import score
from paper import run_paper
from worlds import janus_dictionary, random_dictionary

GATE_T = (1, 2, 4)
GATE_KAPPA = (2, 8)
WORLD_LAM = (0.5, 1.0, 2.0)
BETA = (0.03, 0.1, 0.3)
N_VAL = 30
VAL_PAN_PHOTOS = ["astronaut", "chelsea", "coins", "rocket"]


def gated_recons(g, ev, Y, W, p, kind, t, kappa):
    out = []
    for y, xw in zip(Y, W):
        e = ev.pc(xw, t) if kind == "pc" else ev.energy(xw, t)
        out.append(g.solve_gated(y, p["csc_lam"], p["csc_gamma"], w=1 + kappa * (1 - footprint_pool(e)),
                                 iters=p["csc_iters"]))
    return out


def main():
    t0 = time.time()
    base = json.loads((ROOT / "params.json").read_text())
    train, val, _ = natural()
    V = val[:N_VAL]
    cache = ROOT / "runs" / "fit_natural.pkl"
    log, m7 = [], {}
    for fog in ("heat", "sihti"):
        p = load_params(fog=fog)
        b = Bench(train, p["depth"], p["sigma"], cache=cache, dt=p.get("dt", 0.25))
        g = GatedCSC(b.D, b.A)
        ev = Evidence(b.gauss, b.A, p["sigma"])
        Y = b.observe_set(V, seed=11)
        W = [b.gauss.wiener(y, b.A, p["sigma"]) for y in Y]
        base_mse = score([g.solve_gated(y, p["csc_lam"], p["csc_gamma"], iters=p["csc_iters"]) for y in Y], V)["mse"]
        log.append(f"[{fog}] csc (Model 6 parameters): val mse {base_mse:.6f}")
        m7[fog] = {}
        for kind in ("pc", "energy"):
            r = {}
            for t in GATE_T:
                for k in GATE_KAPPA:
                    r[(t, k)] = score(gated_recons(g, ev, Y, W, p, kind, t, k), V)["mse"]
                    log.append(f"[{fog}] {kind} gate t={t} kappa={k}: val mse {r[(t, k)]:.6f}")
            t_best, k_best = min(r, key=r.get)
            m7[fog][f"{kind}_t"], m7[fog][f"{kind}_kappa"] = t_best, k_best
            for name, v, grid in (("t", t_best, GATE_T), ("kappa", k_best, GATE_KAPPA)):
                if v in (min(grid), max(grid)):
                    log.append(f"[{fog}] NOTE: {kind} {name}={v} on the edge of its grid")
        if fog == "heat":
            # worlds: each non-natural dictionary gets its own lam (x Model 6's), gamma fixed
            for wname, D in (("janus", janus_dictionary()), ("random", random_dictionary())):
                gw = GatedCSC(D, b.A)
                r = {}
                for f in WORLD_LAM:
                    r[f] = score([gw.solve_gated(y, f * p["csc_lam"], p["csc_gamma"], iters=p["csc_iters"])
                                  for y in Y], V)["mse"]
                    log.append(f"[heat] world={wname} lam x{f}: val mse {r[f]:.6f}")
                m7["heat"][f"{wname}_lam_factor"] = min(r, key=r.get)
            # the living paper: e_fog from validation single frames, beta on validation pans
            R = [g.solve_gated(y, p["csc_lam"], p["csc_gamma"], iters=p["csc_iters"]) for y in Y]
            e_fog = float(np.median([np.mean((b.A @ (r_ - t_).ravel()) ** 2) for r_, t_ in zip(R, V)]))
            m7["heat"]["e_fog"] = e_fog
            log.append(f"[heat] e_fog (median fogged reconstruction error) = {e_fog:.3g}")
            seqs = pans(VAL_PAN_PHOTOS, 2, 6, seed=5)
            solver = lambda y, mw, mu: g.solve_gated(y, p["csc_lam"], p["csc_gamma"], mem_w=mw,
                                                     mem_mu=mu, iters=p["csc_iters"])
            r = {}
            for beta in BETA:
                errs = []
                for i, (fr, offs) in enumerate(seqs):
                    xs, _, _ = run_paper(b.observe_set(fr, seed=100 + i), solver, b.A, p["sigma"], beta, e_fog)
                    errs += [np.mean((x - t_) ** 2) for x, t_ in zip(xs[3:], fr[3:])]
                r[beta] = float(np.mean(errs))
                log.append(f"[heat] paper beta={beta}: val-pan mse (frames 3-5) {r[beta]:.6f}")
            m7["heat"]["beta"] = min(r, key=r.get)
            if m7["heat"]["beta"] in (min(BETA), max(BETA)):
                log.append(f"[heat] NOTE: beta={m7['heat']['beta']} on the edge of its grid")
    m7["log"] = log
    m7["seconds"] = round(time.time() - t0)
    base["model7"] = m7
    (ROOT / "params.json").write_text(json.dumps(base, indent=2))
    print("\n".join(log))
    print(json.dumps({f: {k: v for k, v in m7[f].items()} for f in ("heat", "sihti")}, indent=1))


if __name__ == "__main__":
    main()
