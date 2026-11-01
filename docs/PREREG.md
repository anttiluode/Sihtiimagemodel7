# Sihtiimagemodel7 — preregistration

Written before any Model 7 test run. Parameters are chosen by `tune7.py` on validation only
(every knob by validation MSE, the same rule as Model 6) and committed with this file.
Model 6's CSC parameters (λ, γ, 400 iterations, per fog) are inherited frozen.

## Where this comes from

Model 6 passed "competing atoms beat linear focus, independent patches and Model 4's states" on
both fogs, and failed two gates for one reason: CSC writes faint atoms into **flat** regions where
the fog left only noise (spurious-edge rate 10–13 % vs EPLL's 4 %), and the same noise-driven atoms
flicker between frames. The Nov 1 conversation proposed two reasons to stay silent:

* **in space — cross-scale phase congruency.** Real features have the same local phase at every
  scale (Morrone & Owens 1987; Kovesi 1999); noise does not.
* **in time — memory.** Model 5's living paper (transport, precision, trust) carrying the estimate.

And one control from the review of `attenuation.py`: fitting a stream into a world only means
something if the world matters — a world learned from the wrong images must fit worse.

## Instruments

* `pyramid.py` — undecimated complex steerable pyramid, 3 scales × 4 orientations on 32×32: tight
  frame (exact reconstruction, tested), phase = position (a grating shifted by d moves phase by f·d,
  tested to 1e-6). Known limit, measured before the test: adjacent undecimated scales overlap, so
  pure noise has congruency ≈ 0.74 without a threshold (an ideal edge ≈ 0.97); Kovesi's noise
  threshold T is required (tested).
* Evidence for the gates is computed from the **Wiener estimate** of the observation, never from
  CSC's own output. Thresholds are in units of what the Wiener estimate of pure noise produces
  (T0 for congruency, τ0 for energy), so neither gate gets a hand-picked scale.
* Gate: w = 1 + κ(1 − e) per atom position, e = evidence max-pooled over the atom's 8×8 footprint.
  The PC and energy gates share the pyramid, the pooling, the grids (t ∈ {1,2,4}, κ ∈ {2,8}) and
  the selection rule; only phase-congruency vs energy differs.

## Data and fogs

Model 6's: 64 test crops of `camera`, `coffee`, `clock` (seed 23), validation crops from the
training photographs. **Primary fog: heat** (true low-pass, matched time); **replication: Sihti fog**
(`purify⁸`, the band-stop). Temporal: Model 6's 24 test pans (seed 31), heat fog.

## Metrics

MSE; edge precision and recall (Canny, 1.5 px); **flat-region spurious-edge rate** (Canny edges of
the result inside the flattest third of the truth's gradient magnitude); orientation error;
flicker. Paired bootstrap CIs (2000, seed 0) for MSE ratios.

## Gates

| Gate | Claim | Pass condition |
|---|---|---|
| **S1** | phase congruency gives the atoms silence | flat-rate(csc+pc) ≤ 0.8 × flat-rate(csc) **and** MSE(csc+pc) ≤ 1.01 × MSE(csc) **and** recall(csc+pc) ≥ recall(csc) − 0.02 — on the heat fog; repeated on the Sihti fog |
| **S2** | it is the *phase*, not the energy | flat-rate(csc+pc) ≤ 0.9 × flat-rate(csc+energy) **and** MSE(csc+pc) ≤ 1.01 × MSE(csc+energy) — heat; repeated on Sihti |
| **W1** | the learned world matters (the attenuation.py control) | MSE(natural world) ≤ 0.9 × MSE(Janus A/B world) and ≤ 0.9 × MSE(random atoms), both CIs < 1 (heat fog; each world's λ tuned on validation) |
| **T1** | memory makes the atoms calmer *and* better | flicker(paper) ≤ 0.5 × flicker(per-frame CSC) **and** late-frame MSE (frames 3–5) ≤ 0.95 × per-frame, CI < 1 |
| **T2** | a foreign memory is switched off | on a first frame given another sequence's final memory: MSE with trust ≤ 1.05 × per-frame **and** without trust ≥ 1.5 × per-frame |

Report-only: Model 6's Wiener / independent / EPLL on the same metrics; κ frontier (1, 4, 16, 32)
for both gates; the "phase transport" paper (memory moved by advancing each band's phase by
ω_s·(d·direction) instead of shifting pixels — position as phase, exact only for narrow bands).

## Pilot disclosure (validation only, before this file)

* 20 validation crops, heat fog: CSC flat-rate 12.5 %, MSE 0.001622; PC gate flat-rate 8.2–9.8 % at
  MSE 0.001614–0.001771; energy gate 9.4–11.5 % at 0.001611–0.001632. This shaped the grids.
* 4 validation pans (training photographs): per-frame flicker 0.0166, MSE 0.00176; paper β = 0.1:
  flicker 0.0030, MSE 0.00160; β = 1: flicker 0.0007, MSE 0.00176. Motion estimated correctly in
  every step. This shaped the β grid.
* No test photograph has been opened for Model 7.

## Amendment 1 — after validation stage 1, before any test run

**Gate selection rule changed.** Stage 1 (`tune7.py`) selected each gate by validation MSE, and
both gates went to the weakest corner of their grids (smallest t, smallest κ): MSE barely rewards
silence, so that rule would have turned S1/S2 into tests of an almost-absent gate. The gates' claim
is *more silence at no cost*, so the rule now matches the claim and is identical for both gates:
**among settings whose validation MSE ≤ 1.01 × CSC's and recall ≥ CSC's − 0.02, choose the lowest
validation flat-region spurious-edge rate** (if none qualifies, the gate is off). Grids widened:
PC t ∈ {0.5, 1, 2}, energy t ∈ {1, 2, 4, 8}, κ ∈ {1, 2, 4, 8} (`tune7_stage2.py`). β (memory) stays
selected by validation MSE; its grid is extended below its stage-1 edge (0.003, 0.01).

**Seen on validation, disclosed.** The world learned from Janus's two synthetic images already
matched the natural world on validation (MSE 0.001787 vs 0.001792; random atoms 0.002059). W1 is
kept exactly as preregistered; this disclosure is so that its likely outcome is not mistaken for a
post-hoc surprise.
