"""The preregistered test of Model 7 (docs/PREREG.md). Writes results/.

    python run7.py                # everything (about 40 min on 2 cores)
"""
import json
import time

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude

from bench import Bench, ROOT, load_params, pans, flicker
from data import natural, TEST_PHOTOS
from gate import GatedCSC, Evidence, footprint_pool
from metrics import score, edges, edge_counts
from paper import run_paper
from worlds import janus_dictionary, random_dictionary
from figures7 import gate_figure, frontier_figure, paper_figure, worlds_figure


def boot_ratio(a, b, n=2000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def silence(recons, truths):
    """Edge precision, recall, and the spurious-edge rate in the flattest third of each truth."""
    c = np.array([edge_counts(x, t) for x, t in zip(recons, truths)], float).sum(0)
    fp = px = 0
    for x, t in zip(recons, truths):
        g = gaussian_gradient_magnitude(t, 1.0)
        flat = g <= np.quantile(g, 1 / 3)
        fp += (edges(x) & flat).sum()
        px += flat.sum()
    return dict(precision=c[0] / max(c[1], 1), recall=c[2] / max(c[3], 1), flat_rate=fp / px)


def full_score(recons, truths):
    s = score(recons, truths)
    s.update(silence(recons, truths))
    return s


def gate_maps(ev, xw, kind, t):
    return ev.pc(xw, t) if kind == "pc" else ev.energy(xw, t)


def run_gates(fog, train, test, m7, out):
    p = load_params(fog=fog)
    b = Bench(train, p["depth"], p["sigma"], cache=ROOT / "runs" / "fit_natural.pkl", dt=p.get("dt", 0.25))
    g, ev = GatedCSC(b.D, b.A), Evidence(b.gauss, b.A, p["sigma"])
    Y = b.observe_set(test, seed=23)
    W = [b.gauss.wiener(y, b.A, p["sigma"]) for y in Y]
    q = m7[fog]
    rec = {"csc": [g.solve_gated(y, p["csc_lam"], p["csc_gamma"], iters=p["csc_iters"]) for y in Y]}
    for kind in ("pc", "energy"):
        rec[f"csc+{kind}"] = [g.solve_gated(y, p["csc_lam"], p["csc_gamma"], iters=p["csc_iters"],
                                            w=1 + q[f"{kind}_kappa"] * (1 - footprint_pool(gate_maps(ev, xw, kind, q[f"{kind}_t"]))))
                              for y, xw in zip(Y, W)]
    ctx = b.run(Y, p, arms=("wiener", "independent", "epll"), seed=23)     # Model 6 context arms
    rec.update(ctx)
    sc = {a: full_score(v, test) for a, v in rec.items()}
    gate_figure(test, Y, W, rec, ev, q, out / f"gates_{fog}.png")
    # report-only frontier: silence vs error as kappa grows, both gates at their chosen t
    frontier = {}
    for kind in ("pc", "energy"):
        frontier[kind] = []
        for kappa in (1, 4, 16, 32):
            r = [g.solve_gated(y, p["csc_lam"], p["csc_gamma"], iters=p["csc_iters"],
                               w=1 + kappa * (1 - footprint_pool(gate_maps(ev, xw, kind, q[f"{kind}_t"]))))
                 for y, xw in zip(Y, W)]
            s = full_score(r, test)
            frontier[kind].append(dict(kappa=kappa, mse=s["mse"], flat_rate=s["flat_rate"], recall=s["recall"]))
    frontier_figure(frontier, sc, out / f"frontier_{fog}.png", fog)
    return b, g, sc, frontier, rec


def gates_S(sc, tag):
    c, pc, en = sc["csc"], sc["csc+pc"], sc["csc+energy"]
    return {
        f"S1 phase congruency gives silence ({tag})": dict(
            flat_rate=[pc["flat_rate"], c["flat_rate"]], mse=[pc["mse"], c["mse"]], recall=[pc["recall"], c["recall"]],
            passed=bool(pc["flat_rate"] <= .8 * c["flat_rate"] and pc["mse"] <= 1.01 * c["mse"]
                        and pc["recall"] >= c["recall"] - .02)),
        f"S2 phase beats energy ({tag})": dict(
            flat_rate=[pc["flat_rate"], en["flat_rate"]], mse=[pc["mse"], en["mse"]],
            passed=bool(pc["flat_rate"] <= .9 * en["flat_rate"] and pc["mse"] <= 1.01 * en["mse"])),
    }


def main():
    t0 = time.time()
    params = json.loads((ROOT / "params.json").read_text())
    m7 = params["model7"]
    train, _, test = natural()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    R = dict(n_test=len(test))

    # ---- 1. silence: gates on both fogs
    b, g, sc_h, fr_h, rec_h = run_gates("heat", train, test, m7, out)
    _, _, sc_s, fr_s, _ = run_gates("sihti", train, test, m7, out)
    R["heat"] = dict(scores=sc_h, frontier=fr_h)
    R["sihti"] = dict(scores=sc_s, frontier=fr_s)
    G = {}
    G.update(gates_S(sc_h, "heat fog"))
    G.update(gates_S(sc_s, "Sihti fog"))

    # ---- 2. which world? (heat fog)
    p = load_params(fog="heat")
    Y = b.observe_set(test, seed=23)
    worlds = {"natural": rec_h["csc"]}
    for name, D in (("janus", janus_dictionary()), ("random", random_dictionary())):
        gw = GatedCSC(D, b.A)
        lam = m7["heat"][f"{name}_lam_factor"] * p["csc_lam"]
        worlds[name] = [gw.solve_gated(y, lam, p["csc_gamma"], iters=p["csc_iters"]) for y in Y]
    wsc = {k: full_score(v, test) for k, v in worlds.items()}
    R["worlds"] = wsc
    r1 = boot_ratio(wsc["natural"]["mse_per_image"], wsc["janus"]["mse_per_image"])
    r2 = boot_ratio(wsc["natural"]["mse_per_image"], wsc["random"]["mse_per_image"])
    G["W1 the learned world matters"] = dict(
        natural_over_janus=r1, natural_over_random=r2,
        passed=bool(r1[0] <= .9 and r1[2] < 1 and r2[0] <= .9 and r2[2] < 1))
    worlds_figure(test, Y, worlds, out / "worlds.png")

    # ---- 3. the living paper over time (heat fog, 24 test pans, same seeds as Model 6)
    seqs = pans(TEST_PHOTOS, 8, 6, seed=31)
    e_fog, beta = m7["heat"]["e_fog"], m7["heat"]["beta"]
    solver = lambda y, mw, mu: g.solve_gated(y, p["csc_lam"], p["csc_gamma"], mem_w=mw, mem_mu=mu,
                                             iters=p["csc_iters"])
    arms = {k: dict(flicker=[], late=[]) for k in ("per_frame", "independent", "paper", "paper_no_trust", "paper_phase")}
    finals, ys_all, trusts = [], [], []
    for i, (fr, offs) in enumerate(seqs):
        Yf = b.observe_set(fr, seed=31 * 1000 + i)
        ys_all.append(Yf)
        runs = {"per_frame": [solver(y, None, None) for y in Yf],
                "independent": [b.ind.solve(y, p["ind_lam"]) for y in Yf]}
        xs, tr, _ = run_paper(Yf, solver, b.A, p["sigma"], beta, e_fog)
        runs["paper"] = xs
        trusts += [t for t in tr if t == t]
        runs["paper_no_trust"] = run_paper(Yf, solver, b.A, p["sigma"], beta, e_fog, use_trust=False)[0]
        runs["paper_phase"] = run_paper(Yf, solver, b.A, p["sigma"], beta, e_fog, transport="phase")[0]
        for k, v in runs.items():
            arms[k]["flicker"].append(flicker(v, offs))
            arms[k]["late"].append(float(np.mean([np.mean((x - t) ** 2) for x, t in zip(v[3:], fr[3:])])))
        finals.append((xs[-1], np.full_like(xs[-1], 6.0)))
        if i == 0:
            paper_figure(fr, Yf, runs, offs, out / "paper.png")
    T = {k: dict(flicker=float(np.nanmean(v["flicker"])), late_mse=float(np.mean(v["late"])),
                 late_per_seq=v["late"]) for k, v in arms.items()}
    T["min_trust_on_own_history"] = float(min(trusts))
    # foreign history: sequence i's first frame, with the memory sequence (i+1) ended in
    cut = dict(per_frame=[], trust=[], no_trust=[], trust_values=[])
    for i, (fr, offs) in enumerate(seqs):
        y0 = ys_all[i][0]
        hist = finals[(i + 1) % len(seqs)]
        cut["per_frame"].append(float(np.mean((solver(y0, None, None) - fr[0]) ** 2)))
        for key, ut in (("trust", True), ("no_trust", False)):
            xs, tr, _ = run_paper([y0], solver, b.A, p["sigma"], beta, e_fog, use_trust=ut, history=hist)
            cut[key].append(float(np.mean((xs[0] - fr[0]) ** 2)))
            if ut:
                cut["trust_values"].append(tr[0])
    T["foreign_history"] = cut
    R["temporal"] = T
    pf, pp = T["per_frame"], T["paper"]
    rr = boot_ratio(arms["paper"]["late"], arms["per_frame"]["late"])
    G["T1 memory: calmer and better"] = dict(
        flicker=[pp["flicker"], pf["flicker"]], late_mse_ratio=rr,
        passed=bool(pp["flicker"] <= .5 * pf["flicker"] and rr[0] <= .95 and rr[2] < 1))
    ct = np.mean(cut["trust"]) / np.mean(cut["per_frame"])
    cn = np.mean(cut["no_trust"]) / np.mean(cut["per_frame"])
    G["T2 a foreign memory is switched off"] = dict(
        with_trust_ratio=float(ct), without_trust_ratio=float(cn),
        passed=bool(ct <= 1.05 and cn >= 1.5))
    R["gates"] = G
    R["seconds"] = round(time.time() - t0)
    for blk in (R["heat"]["scores"], R["sihti"]["scores"], R["worlds"]):
        for v in blk.values():
            v["mse_per_image"] = [round(x, 7) for x in v["mse_per_image"]]
    (out / "metrics.json").write_text(json.dumps(R, indent=1, default=float))
    for k, v in G.items():
        print(f"{k}: {'PASS' if v['passed'] else 'FAIL'}  {json.dumps({a: b_ for a, b_ in v.items() if a != 'passed'}, default=float)}")
    print(f"done in {R['seconds']} s")


if __name__ == "__main__":
    main()
