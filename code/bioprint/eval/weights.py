"""Is a WEIGHTED scaled-Manhattan better than the plain mean? OWNER: Agent H (eval).

Answers the lead's question - "the score is a simple mean of scaled deviations,
why not a weighted average?" - with evidence on CMU, and sweeps the threshold rule
while it is at it. Same data, same protocols and the same metric functions as
eval/cmu.py (imported, not copied); the scorer's own centre / spread / cap code is
reused so that unit weights reproduce eval/cmu.py to the last digit (checked at
start-up).

    cd code/bioprint
    ~/miniconda3/envs/bigidea/bin/python -m eval.weights          # report
    ~/miniconda3/envs/bigidea/bin/python -m eval.weights --save   # + eval/results/weights-<time>.json

What is compared
----------------
  baseline   every feature weight 1 (the shipped scorer).
  family     one weight per family H / DD / UD, grid {0.5, 1, 1.5, 2}^3.
  disc       per-feature Fisher-style discriminability, w_j = between-subject
             variance / mean within-subject variance of feature j (also a tempered
             sqrt variant, because the raw ratio is very peaky).
All weight vectors are normalised to mean 1, so the score keeps its unit ("typical
spreads") and stays comparable to the threshold rule.

How it stays honest
-------------------
51 subjects is not many. Subjects are split into two halves (alternating in id
order); the weights are CHOSEN on one half (grid winner by protocol-2 EER, or the
variance ratio estimated on those subjects only) and every number reported is
measured on the OTHER half, then the halves swap. The "cross-fit" line is the
mean over all 51 subjects, each scored with weights it played no part in choosing,
and is the number to quote. The oracle line (best grid point on the eval half
itself) is printed only to show how much the grid can overfit.

Threshold rule
--------------
With the chosen weights: THRESHOLD_CEILING swept over the live per-user rule,
and a population-calibrated constant (the threshold at which impostors on the fit
half get FAR 5% / 1% / EER) applied unchanged to the eval half.

Result (2026-09-19, 28 features, N=10; engine/scorer.py left unchanged)
-----------------------------------------------------------------------
  | weights (cross-fit)          | P2 EER | P2 FRR@FAR5% | P2 thr FRR/FAR | P1 EER |
  |------------------------------|--------|--------------|----------------|--------|
  | baseline, all 1              | 20.2%  | 51.6%        | 21.6% / 26.2%  | 7.5%   |
  | family H/DD/UD ~ 1.5/1.5/0.5 | 19.6%  | 48.7%        | 23.9% / 21.5%  | 7.3%   |
  | disc^1 (variance ratio)      | 21.4%  | 50.1%        | 30.9% / 17.7%  | 9.4%   |
  | disc^0.5                     | 19.8%  | 48.6%        | 26.2% / 19.8%  | 7.6%   |
The family gain is +0.6..0.8 points on every split tried (alternating, seeds 1-3,
with/without Return): real but small, under the 1-point bar, and at the shipped
ceiling it only trades FRR for FAR. Per-feature weights overfit at 25 subjects.
Threshold: the population-EER constant lands at 1.8-1.9, i.e. the ceiling already
IS the population operating point; a constant calibrated to FAR 5% (t ~ 1.3)
costs FRR 72% cross-session at N=10, so a low-FAR point is not reachable from
this channel alone.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # code/bioprint
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import scorer  # noqa: E402
from eval import cmu  # noqa: E402

FAMILIES = ("H", "DD", "UD")
GRID = (0.5, 1.0, 1.5, 2.0)
CEILINGS = (1.5, 1.7, 1.9, 2.1, 2.3)
FARS = (0.01, 0.05)


# ---------------------------------------------------------------- precomputation
#
# Centre, spread and the leave-one-out centres/spreads do not depend on the
# weights; only the final reduction over features does. So every per-feature
# capped deviation is computed once and a weight vector is then a dot product.


@dataclass
class Prepared:
    """One subject under one protocol: capped per-feature deviations (rows x F)."""
    subject: str
    genuine: np.ndarray   # (n_genuine, F)
    impostor: np.ndarray  # (n_impostor, F)
    loo: np.ndarray       # (n_loo, F)  each enrolment sample vs the model without it


def _prepare(train: np.ndarray, genuine: np.ndarray, impostor: np.ndarray,
             families: np.ndarray, sid: str) -> Prepared:
    cap = scorer.DEVIATION_CAP
    c, s = scorer._fit_arrays(train, families)
    dev = lambda X: scorer._deviations(c, s, X, cap)  # noqa: E731  (broadcasts over rows)
    held_out = range(len(train)) if len(train) <= scorer.LOO_MAX_SAMPLES else np.linspace(
        0, len(train) - 1, scorer.LOO_MAX_SAMPLES).astype(int)
    loo = []
    for i in held_out:
        rest = np.delete(train, i, axis=0)
        if len(rest) < 2:
            continue
        ci, si = scorer._fit_arrays(rest, families)
        loo.append(scorer._deviations(ci, si, train[i], cap))
    return Prepared(sid, dev(genuine), dev(impostor), np.asarray(loo))


def prepare(ds: cmu.Dataset, splitter) -> dict[str, Prepared]:
    families = np.array([scorer._family(n) for n in ds.names])
    out = {}
    for sid, s in ds.subjects.items():
        sp = splitter(s)
        out[sid] = _prepare(sp.train, sp.genuine, cmu.impostors_for(ds, sid), families, sid)
    return out


def _check_matches_scorer(ds: cmu.Dataset, prep: dict[str, Prepared], splitter) -> None:
    """Unit weights must reproduce engine.scorer exactly, or nothing below means anything."""
    sid, s = next(iter(ds.subjects.items()))
    sp = splitter(s)
    model = scorer.fit(sp.train.tolist(), ds.names)
    p = prep[sid]
    for row, dev in zip(sp.genuine[:20].tolist(), p.genuine[:20]):
        assert abs(scorer.distance(model, row)[0] - dev.mean()) < 1e-9
    assert abs(model.threshold - threshold_for(p, np.ones(len(ds.names)), scorer.THRESHOLD_CEILING)) < 1e-9


# ---------------------------------------------------------------- scoring with weights


def normalise(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w / w.mean()


def family_vector(names: list[str], fam: dict[str, float]) -> np.ndarray:
    """Per-feature weights from per-family ones; unknown prefixes get 1. Mean-normalised."""
    return normalise([fam.get(scorer._family(n), 1.0) for n in names])


def threshold_for(p: Prepared, w: np.ndarray, ceiling: float) -> float:
    """The live per-user rule (scorer.fit) applied to weighted leave-one-out distances."""
    if len(p.loo) == 0:
        return ceiling
    loo = p.loo @ w / len(w)
    raw = float(np.quantile(loo, scorer.LOO_QUANTILE) * scorer.THRESHOLD_MARGIN)
    return float(np.clip(raw, scorer.THRESHOLD_FLOOR, ceiling))


def evaluate(prep: dict[str, Prepared], w: np.ndarray, subjects: list[str],
             ceiling: float = scorer.THRESHOLD_CEILING, constant: float | None = None) -> dict:
    """Mean per-subject EER, FRR@FAR, and FRR/FAR at the model threshold (per-user LOO
    rule with `ceiling`, or the `constant` if given) over `subjects`."""
    F = len(w)
    rows = []
    for sid in subjects:
        p = prep[sid]
        g, i = p.genuine @ w / F, p.impostor @ w / F
        t = constant if constant is not None else threshold_for(p, w, ceiling)
        frr, far = cmu.rates_at(g, i, t)
        row = {"eer": cmu.eer(g, i), "frr_model": frr, "far_model": far, "threshold": t}
        for f in FARS:
            row[f"frr_at_far_{f:g}"] = cmu.frr_at_far(g, i, f)
        rows.append(row)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {"n_subjects": len(rows)}


def pooled_scores(prep: dict[str, Prepared], w: np.ndarray, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    F = len(w)
    g = np.concatenate([prep[s].genuine @ w / F for s in subjects])
    i = np.concatenate([prep[s].impostor @ w / F for s in subjects])
    return g, i


def calibrated_thresholds(prep: dict[str, Prepared], w: np.ndarray, subjects: list[str]) -> dict[str, float]:
    """Population constants from the FIT half: the score at which pooled impostors
    reach FAR 1% / 5%, and the pooled-EER point. Every subject contributes the same
    number of impostor rows, so the pooled quantile is the mean per-subject FAR."""
    g, i = pooled_scores(prep, w, subjects)
    out = {f"far_{f:g}": float(np.quantile(i, f)) for f in FARS}
    ts = cmu._thresholds(g, i)
    frr, far = cmu._rates(g, i, ts)
    out["eer"] = float(ts[int(np.argmin(np.abs(frr - far)))])
    return out


# ---------------------------------------------------------------- weight families


def discriminability(ds: cmu.Dataset, subjects: list[str], power: float = 1.0) -> np.ndarray:
    """w_j = var over subjects of the subject median / mean within-subject variance,
    estimated on `subjects` only (all 400 reps each), raised to `power`, mean-normalised."""
    med = np.array([np.median(ds.subjects[s].X, axis=0) for s in subjects])
    within = np.array([ds.subjects[s].X.var(axis=0, ddof=1) for s in subjects]).mean(axis=0)
    between = med.var(axis=0, ddof=1)
    return normalise((between / np.maximum(within, 1e-12)) ** power)


def family_grid(grid=GRID) -> list[dict[str, float]]:
    """Every H/DD/UD combination, deduplicated up to scale (weights are mean-normalised)."""
    seen, out = set(), []
    for combo in itertools.product(grid, repeat=len(FAMILIES)):
        key = tuple(np.round(np.asarray(combo) / max(combo), 6))
        if key not in seen:
            seen.add(key)
            out.append(dict(zip(FAMILIES, combo)))
    return out


def halves(ds: cmu.Dataset, seed: int | None = None) -> list[list[str]]:
    """Alternating subjects in id order by default; a seed shuffles first (robustness check)."""
    ids = sorted(ds.subjects)
    if seed is not None:
        ids = list(np.random.default_rng(seed).permutation(ids))
    return [sorted(ids[0::2]), sorted(ids[1::2])]


# ---------------------------------------------------------------- report


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _row(label: str, p1: dict, p2: dict) -> str:
    return (f"  {label:<34} P2: EER {_pct(p2['eer'])} FRR@1% {_pct(p2['frr_at_far_0.01'])}"
            f" FRR@5% {_pct(p2['frr_at_far_0.05'])} thr FRR {_pct(p2['frr_model'])}/FAR {_pct(p2['far_model'])}"
            f" | P1: EER {_pct(p1['eer'])} FRR@5% {_pct(p1['frr_at_far_0.05'])}")


def _fam_label(fam: dict[str, float]) -> str:
    return "H/DD/UD = " + "/".join(f"{fam[f]:g}" for f in FAMILIES)


def crossfit(prep1, prep2, folds: list[tuple[list[str], np.ndarray]], ceiling=scorer.THRESHOLD_CEILING) -> dict:
    """Evaluate each fold's weights on its eval half; average over all subjects."""
    per_fold, n = [], 0
    for eval_ids, w in folds:
        per_fold.append({"p1": evaluate(prep1, w, eval_ids, ceiling), "p2": evaluate(prep2, w, eval_ids, ceiling),
                         "eval_subjects": eval_ids, "weights": w.tolist()})
        n += len(eval_ids)
    agg = {}
    for proto in ("p1", "p2"):
        keys = [k for k in per_fold[0][proto] if k != "n_subjects"]
        agg[proto] = {k: float(sum(f[proto][k] * len(f["eval_subjects"]) for f in per_fold) / n) for k in keys}
    return {"crossfit": agg, "folds": per_fold}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Weighted scaled-Manhattan vs the plain mean, on CMU, cross-fitted.")
    ap.add_argument("--data", type=Path, default=cmu.DATA)
    ap.add_argument("--n", type=int, default=cmu.ENROLL_TARGET, help="protocol-2 enrolment size")
    ap.add_argument("--with-return", action="store_true", help="keep the Return columns (31 features)")
    ap.add_argument("--grid", type=float, nargs="+", default=list(GRID), help="family weight values")
    ap.add_argument("--ceilings", type=float, nargs="+", default=list(CEILINGS))
    ap.add_argument("--seed", type=int, help="shuffle subjects before halving (default: alternate in id order)")
    ap.add_argument("--save", action="store_true", help="write JSON to eval/results/weights-<timestamp>.json")
    ap.add_argument("--json", type=Path, help="write JSON to this path instead")
    args = ap.parse_args(argv)

    ds = cmu.load(cmu.ensure_data(args.data), include_return=args.with_return)
    names = ds.names
    F = len(names)
    split1 = cmu.split_km
    split2 = lambda s: cmu.split_bioprint(s, args.n, "first")  # noqa: E731
    t0 = time.perf_counter()
    prep1, prep2 = prepare(ds, split1), prepare(ds, split2)
    _check_matches_scorer(ds, prep2, split2)
    _check_matches_scorer(ds, prep1, split1)
    A, B = halves(ds, args.seed)
    all_ids = sorted(ds.subjects)
    ones = np.ones(F)
    out: dict = {"n_subjects": len(all_ids), "features": F, "enroll_n": args.n, "halves": [A, B],
                 "seed": args.seed, "grid": args.grid, "ceilings": args.ceilings}

    print(f"CMU weights study — {len(all_ids)} subjects, {F} features, P2 N={args.n},"
          f" halves {len(A)}/{len(B)} ({'seed ' + str(args.seed) if args.seed is not None else 'alternating'};"
          f" prep {time.perf_counter() - t0:.1f}s)")
    print("P1 = K&M replication (train 200, test 200); P2 = enrol N of session 1, genuine = sessions 2-8."
          " thr = live per-user LOO rule, ceiling", scorer.THRESHOLD_CEILING)

    # a. baseline -------------------------------------------------------------
    base = crossfit(prep1, prep2, [(A, ones), (B, ones)])
    out["baseline"] = base
    print("\n[a] baseline (all weights 1), all 51 subjects")
    print(_row("baseline", base["crossfit"]["p1"], base["crossfit"]["p2"]))

    # b. family weights ---------------------------------------------------------
    print("\n[b] family weights H/DD/UD, chosen on the fit half by P2 EER, measured on the other half")
    grid = family_grid(tuple(args.grid))
    table = []
    for fam in grid:
        w = family_vector(names, fam)
        r = {"family": fam, "w": w.tolist()}
        for tag, ids in (("A", A), ("B", B)):
            r[f"p2_{tag}"] = evaluate(prep2, w, ids)
            r[f"p1_{tag}"] = evaluate(prep1, w, ids)
        table.append(r)
    out["family_grid"] = table
    folds = []
    for fit_tag, eval_tag, eval_ids in (("A", "B", B), ("B", "A", A)):
        best = min(table, key=lambda r: r[f"p2_{fit_tag}"]["eer"])
        folds.append((eval_ids, family_vector(names, best["family"])))
        print(f"  fit on {fit_tag} -> {_fam_label(best['family'])}  (fit-half P2 EER {_pct(best[f'p2_{fit_tag}']['eer'])})")
        print(_row(f"   eval on {eval_tag}", best[f"p1_{eval_tag}"], best[f"p2_{eval_tag}"]))
    fam_cf = crossfit(prep1, prep2, folds)
    out["family_best"] = fam_cf | {"chosen": [min(table, key=lambda r: r["p2_A"]["eer"])["family"],
                                              min(table, key=lambda r: r["p2_B"]["eer"])["family"]]}
    print(_row("family weights, cross-fit", fam_cf["crossfit"]["p1"], fam_cf["crossfit"]["p2"]))
    # How much is on the table at all? Best grid point on the eval halves themselves.
    oracle = []
    for tag, ids in (("A", A), ("B", B)):
        best = min(table, key=lambda r: r[f"p2_{tag}"]["eer"])
        oracle.append((ids, family_vector(names, best["family"])))
        print(f"  oracle on {tag}: {_fam_label(best['family'])} P2 EER {_pct(best[f'p2_{tag}']['eer'])} (optimistic)")
    out["family_oracle"] = crossfit(prep1, prep2, oracle)
    # Top of the grid on the whole population, so the shape of the landscape is visible.
    print("  grid, ranked by P2 EER over all subjects (in-sample; for the landscape only):")
    ranked = sorted(table, key=lambda r: (r["p2_A"]["eer"] * len(A) + r["p2_B"]["eer"] * len(B)))
    for r in ranked[:5] + [r for r in table if all(v == 1.0 for v in r["family"].values())]:
        e2 = (r["p2_A"]["eer"] * len(A) + r["p2_B"]["eer"] * len(B)) / len(all_ids)
        e1 = (r["p1_A"]["eer"] * len(A) + r["p1_B"]["eer"] * len(B)) / len(all_ids)
        print(f"    {_fam_label(r['family']):<22} P2 EER {_pct(e2)}  P1 EER {_pct(e1)}")

    # c. discriminability weights ------------------------------------------------
    print("\n[c] per-feature discriminability weights, estimated on the fit half only")
    out["disc"] = {}
    disc_folds = {}
    for power in (1.0, 0.5):
        folds = [(B, discriminability(ds, A, power)), (A, discriminability(ds, B, power))]
        disc_folds[power] = folds
        cf = crossfit(prep1, prep2, folds)
        out["disc"][f"power_{power:g}"] = cf | {"names": names}
        wA = folds[0][1]
        top = np.argsort(wA)[::-1][:5]
        print(_row(f"disc^{power:g}, cross-fit", cf["crossfit"]["p1"], cf["crossfit"]["p2"]))
        print(f"    weights fit on A: min {wA.min():.2f} max {wA.max():.2f}; top: "
              + ", ".join(f"{names[j]}={wA[j]:.2f}" for j in top))

    # d. threshold rule ----------------------------------------------------------
    # Which weights? The best cross-fit P2 EER among the candidates; ties go to the baseline.
    cands = {"baseline": [(A, ones), (B, ones)], "family": folds_of(fam_cf),
             "disc^1": disc_folds[1.0], "disc^0.5": disc_folds[0.5]}
    results = {"baseline": base, "family": fam_cf,
               "disc^1": out["disc"]["power_1"], "disc^0.5": out["disc"]["power_0.5"]}
    best_name = min(results, key=lambda k: (round(results[k]["crossfit"]["p2"]["eer"], 4), k != "baseline"))
    gain = base["crossfit"]["p2"]["eer"] - results[best_name]["crossfit"]["p2"]["eer"]
    out["best"] = {"weights": best_name, "p2_eer_gain_pts": 100 * gain}
    out["threshold"] = {}
    print(f"\n[d] threshold rule (best weights = {best_name}, cross-fit P2 EER gain over baseline {100 * gain:+.1f} pts)")
    for wname in dict.fromkeys(["baseline", best_name]):
        chosen = cands[wname]
        rec = out["threshold"][wname] = {"ceiling_sweep": {}, "constant": {}}
        print(f"  weights = {wname}")
        print("   per-user LOO rule, THRESHOLD_CEILING swept (all subjects, cross-fit weights):")
        for c in args.ceilings:
            cf = crossfit(prep1, prep2, chosen, ceiling=c)
            rec["ceiling_sweep"][c] = cf["crossfit"]
            p1, p2 = cf["crossfit"]["p1"], cf["crossfit"]["p2"]
            print(f"     ceiling {c:<4} P2: FRR {_pct(p2['frr_model'])} FAR {_pct(p2['far_model'])}"
                  f" (mean thr {p2['threshold']:.2f}) | P1: FRR {_pct(p1['frr_model'])} FAR {_pct(p1['far_model'])}")
        print("   population constant, calibrated on the fit half's P2 scores, applied unchanged to the eval half:")
        for target in ("far_0.01", "far_0.05", "eer"):
            per = []
            for (eval_ids, w), fit_ids in zip(chosen, (A, B)):
                t = calibrated_thresholds(prep2, w, fit_ids)[target]
                per.append({"t": t, "p2": evaluate(prep2, w, eval_ids, constant=t),
                            "p1": evaluate(prep1, w, eval_ids, constant=t), "n": len(eval_ids)})
            n = sum(x["n"] for x in per)
            agg = {proto: {k: sum(x[proto][k] * x["n"] for x in per) / n for k in per[0][proto] if k != "n_subjects"}
                   for proto in ("p1", "p2")}
            rec["constant"][target] = {"folds": per, "crossfit": agg}
            ts = "/".join(f"{x['t']:.2f}" for x in per)
            print(f"     calibrated to {target:<9} t = {ts}  P2: FRR {_pct(agg['p2']['frr_model'])}"
                  f" FAR {_pct(agg['p2']['far_model'])} | P1: FRR {_pct(agg['p1']['frr_model'])} FAR {_pct(agg['p1']['far_model'])}")

    dest = args.json or (cmu.RESULTS / f"weights-{time.strftime('%Y%m%d-%H%M%S')}.json" if args.save else None)
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1, default=str))
        print(f"\nwrote {dest}")
    return out


def folds_of(cf: dict) -> list[tuple[list[str], np.ndarray]]:
    return [(f["eval_subjects"], np.asarray(f["weights"])) for f in cf["folds"]]


if __name__ == "__main__":
    main()
