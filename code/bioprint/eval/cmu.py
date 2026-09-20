"""CMU Keystroke Dynamics benchmark for OUR scorer. OWNER: Agent E (eval).

Runs `engine.scorer` (fit / score, interface only) on Killourhy & Maxion's
DSL-StrongPasswordData.csv: 51 subjects x 8 sessions x 50 reps of `.tie5Roanl`,
31 timing columns in seconds. Re-runs unchanged whenever the scorer changes.

    cd code/bioprint
    ~/miniconda3/envs/bigidea/bin/python -m eval.cmu            # report
    ~/miniconda3/envs/bigidea/bin/python -m eval.cmu --save     # + eval/results/cmu-<time>.json

Protocols
---------
1. **K&M replication** (DSN 2009): per subject, train on reps 1-200 (sessions 1-4),
   genuine test = reps 201-400 (sessions 5-8), impostors = first 5 reps of each of
   the other 50 subjects (250). Published scaled-Manhattan EER 0.096 (sd 0.069).
   This is a pipeline sanity check, not the product number.
2. **BioPrint-realistic**: enroll on N reps of session 1 (N = ENROLL_TARGET = 10),
   genuine test = every rep of sessions 2-8 (cross-session, 350), impostors as above.
   Reports EER, FRR at FAR 1% / 5% (per-subject oracle threshold), and the FRR / FAR
   at `model.threshold`, i.e. what the live system would actually do. Sweeps N.

Return / Enter
--------------
The live vector excludes Enter (engine/features.py EXCLUDED), so the product-shaped
numbers drop H.Return, DD.l.Return, UD.l.Return (28 features). K&M used all 31, so
protocol 1 is reported both ways; the with-Return figure is the one comparable to
the paper.

Metrics convention: score larger = less like the owner; a sample is ACCEPTED when
score <= threshold (the scorer flags `score > threshold`). FRR = genuine rejected,
FAR = impostors accepted. EER and FRR@FAR are computed per subject, then averaged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # code/bioprint
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import scorer  # noqa: E402  (after sys.path fix)

CMU_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
DATA = Path(__file__).resolve().parent / "data" / "DSL-StrongPasswordData.csv"
RESULTS = Path(__file__).resolve().parent / "results"
ENROLL_TARGET = 10  # mirrors server.ENROLL_TARGET (not imported: server pulls in fastapi + db)
PUBLISHED_EER = 0.096  # Killourhy & Maxion 2009, scaled Manhattan, sd 0.069

# CMU key token -> browser event.code. "Shift.r" is the capital R (CMU times the R
# key only; the live capture would ALSO see ShiftLeft/ShiftRight as its own keystroke).
KEYS = [
    ("period", "Period"), ("t", "KeyT"), ("i", "KeyI"), ("e", "KeyE"), ("five", "Digit5"),
    ("Shift.r", "KeyR"), ("o", "KeyO"), ("a", "KeyA"), ("n", "KeyN"), ("l", "KeyL"),
    ("Return", "Enter"),
]


def cmu_and_live_names() -> tuple[list[str], list[str]]:
    """The 31 CMU column names in file order, and the same columns named the way
    engine/features.keystroke_vector names a live vector (H.KeyT#1, DD.KeyT#1.KeyI#2...)."""
    cmu, live = [], []
    for i, (tok, code) in enumerate(KEYS):
        lab = f"{code}#{i}"
        cmu.append(f"H.{tok}")
        live.append(f"H.{lab}")
        if i + 1 < len(KEYS):
            ntok, ncode = KEYS[i + 1]
            nlab = f"{ncode}#{i + 1}"
            cmu += [f"DD.{tok}.{ntok}", f"UD.{tok}.{ntok}"]
            live += [f"DD.{lab}.{nlab}", f"UD.{lab}.{nlab}"]
    return cmu, live


# ---------------------------------------------------------------- loading


@dataclass
class Subject:
    id: str
    session: np.ndarray  # (400,) int, 1..8
    rep: np.ndarray  # (400,) int, 1..50 within session
    X: np.ndarray  # (400, F) float, MILLISECONDS, file row order (session, rep)


@dataclass
class Dataset:
    names: list[str]  # live-style feature names, len F
    subjects: dict[str, Subject] = field(default_factory=dict)

    def without_return(self) -> "Dataset":
        keep = [j for j, n in enumerate(self.names) if "Enter#" not in n]
        return Dataset(
            [self.names[j] for j in keep],
            {k: Subject(s.id, s.session, s.rep, s.X[:, keep]) for k, s in self.subjects.items()},
        )


def ensure_data(path: Path = DATA) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {CMU_URL} -> {path}", file=sys.stderr)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(CMU_URL, tmp)
        tmp.rename(path)
    return path


def load(path: Path = DATA, include_return: bool = True, *, synthetic_only: bool = False) -> Dataset:
    """CSV -> per-subject arrays in ms. Rows are kept in file order (session, rep)."""
    if not synthetic_only or Path(path).resolve() == DATA.resolve():
        raise RuntimeError('Legacy all-session loading is disabled: use eval.strict_cmu.load_partition(train/dev); test is sealed')
    cmu_names, live_names = cmu_and_live_names()
    rows: dict[str, list[tuple[int, int, list[float]]]] = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if header[:3] != ["subject", "sessionIndex", "rep"] or header[3:] != cmu_names:
            raise ValueError(f"unexpected CMU header: {header}")
        for row in r:
            if not row:
                continue
            rows.setdefault(row[0], []).append(
                (int(row[1]), int(row[2]), [float(v) * 1000.0 for v in row[3:]])
            )
    ds = Dataset(live_names)
    for sid, rs in rows.items():
        rs.sort(key=lambda t: (t[0], t[1]))
        ds.subjects[sid] = Subject(
            sid,
            np.array([t[0] for t in rs]),
            np.array([t[1] for t in rs]),
            np.array([t[2] for t in rs], dtype=float),
        )
    return ds if include_return else ds.without_return()


# ---------------------------------------------------------------- metrics


def _rates(genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FRR (genuine score > t) and FAR (impostor score <= t) at each threshold."""
    g = np.sort(genuine)
    i = np.sort(impostor)
    frr = 1.0 - np.searchsorted(g, thresholds, side="right") / len(g)
    far = np.searchsorted(i, thresholds, side="right") / len(i)
    return frr, far


def _thresholds(genuine, impostor) -> np.ndarray:
    s = np.unique(np.concatenate([genuine, impostor]))
    return np.concatenate([[-np.inf], s])


def eer(genuine, impostor) -> float:
    """Equal error rate. Scans every threshold; at the point where FRR and FAR are
    closest, returns their mean (the usual discrete approximation)."""
    genuine, impostor = np.asarray(genuine, float), np.asarray(impostor, float)
    frr, far = _rates(genuine, impostor, _thresholds(genuine, impostor))
    k = int(np.argmin(np.abs(frr - far)))
    return float((frr[k] + far[k]) / 2.0)


def frr_at_far(genuine, impostor, target: float) -> float:
    """Lowest FRR achievable with FAR <= target (oracle per-subject threshold)."""
    genuine, impostor = np.asarray(genuine, float), np.asarray(impostor, float)
    frr, far = _rates(genuine, impostor, _thresholds(genuine, impostor))
    ok = far <= target + 1e-12
    return float(frr[ok].min())  # t = -inf always satisfies FAR = 0


def rates_at(genuine, impostor, threshold: float) -> tuple[float, float]:
    """(FRR, FAR) at one fixed threshold."""
    frr, far = _rates(np.asarray(genuine, float), np.asarray(impostor, float), np.array([threshold]))
    return float(frr[0]), float(far[0])


# ---------------------------------------------------------------- protocols


@dataclass
class Split:
    train: np.ndarray
    genuine: np.ndarray


def split_km(s: Subject) -> Split:
    """K&M: first 200 reps train, remaining 200 test."""
    return Split(s.X[:200], s.X[200:400])


def split_bioprint(s: Subject, n: int, window: str = "first") -> Split:
    """N reps of session 1 train ('first' = reps 1..N, 'last' = reps 51-N..50);
    every rep of sessions 2-8 is a genuine test."""
    s1 = s.X[s.session == 1]
    train = s1[:n] if window == "first" else s1[len(s1) - n:]
    return Split(train, s.X[s.session >= 2])


def impostors_for(ds: Dataset, sid: str, per_subject: int = 5) -> np.ndarray:
    """First `per_subject` reps (session 1) of every other subject."""
    return np.concatenate([o.X[:per_subject] for k, o in ds.subjects.items() if k != sid])


@dataclass
class Latency:
    fit_ms: list[float] = field(default_factory=list)
    score_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        def s(v):
            a = np.asarray(v)
            return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "n": int(a.size)}
        return {"fit_ms": s(self.fit_ms), "score_ms": s(self.score_ms)}


def evaluate(ds: Dataset, splitter, lat: Latency | None = None, fars=(0.01, 0.05)) -> dict:
    """Run fit/score for every subject. Returns per-subject and aggregate metrics."""
    per = []
    for sid, s in ds.subjects.items():
        sp = splitter(s)
        t0 = time.perf_counter()
        model = scorer.fit(sp.train.tolist(), ds.names)
        t1 = time.perf_counter()
        if lat is not None:
            lat.fit_ms.append((t1 - t0) * 1000)

        def run(X):
            out = np.empty(len(X))
            flags = np.empty(len(X), dtype=bool)
            for j, row in enumerate(X.tolist()):
                a = time.perf_counter()
                res = scorer.score(model, row)
                if lat is not None:
                    lat.score_ms.append((time.perf_counter() - a) * 1000)
                out[j], flags[j] = res.score, res.flagged
            return out, flags

        g, gflag = run(sp.genuine)
        i, iflag = run(impostors_for(ds, sid))
        row = {
            "subject": sid,
            "n_train": len(sp.train),
            "n_genuine": len(g),
            "n_impostor": len(i),
            "eer": eer(g, i),
            "model_threshold": float(model.threshold),
            # Rates the LIVE decision would give: uses `flagged`, so any hard rule
            # the scorer adds on top of `score > threshold` is honoured.
            "frr_model": float(gflag.mean()),
            "far_model": float((~iflag).mean()),
        }
        for f in fars:
            row[f"frr_at_far_{f:g}"] = frr_at_far(g, i, f)
        per.append(row)

    keys = [k for k in per[0] if k.startswith(("eer", "frr", "far"))]
    agg = {k: {"mean": float(np.mean([r[k] for r in per])), "sd": float(np.std([r[k] for r in per], ddof=1)),
               "median": float(np.median([r[k] for r in per]))} for k in keys}
    agg["subjects_with_frr_model_over_20pct"] = int(sum(r["frr_model"] > 0.20 for r in per))
    agg["subjects_with_far_model_over_5pct"] = int(sum(r["far_model"] > 0.05 for r in per))
    return {"aggregate": agg, "per_subject": per}


def gate_a(e: float) -> str:
    if e < 0.10:
        return "PASS (<10%): the password vector is viable; proceed as designed"
    if e <= 0.20:
        return "MARGINAL (10-20%): more enrolment samples or chunk features before committing"
    return "FAIL (>20%): password vector cannot carry identity alone; shift weight to challenge + device axes"


# ---------------------------------------------------------------- report


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _line(label: str, a: dict) -> str:
    return (f"  {label:<26} EER {_pct(a['eer']['mean'])} ± {_pct(a['eer']['sd'])}"
            f" | FRR@FAR1% {_pct(a['frr_at_far_0.01']['mean'])}"
            f" | FRR@FAR5% {_pct(a['frr_at_far_0.05']['mean'])}"
            f" | model thr: FRR {_pct(a['frr_model']['mean'])} FAR {_pct(a['far_model']['mean'])}")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Evaluate engine.scorer on the CMU keystroke benchmark.")
    ap.add_argument("--data", type=Path, default=DATA, help="CSV path (downloaded from CMU if missing)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 50], help="enrollment sizes to sweep")
    ap.add_argument("--subjects", type=int, default=0, help="use only the first K subjects (0 = all; for quick runs)")
    ap.add_argument("--save", action="store_true", help="write JSON to eval/results/cmu-<timestamp>.json")
    ap.add_argument("--json", type=Path, help="write JSON to this path instead")
    args = ap.parse_args(argv)

    full = load(ensure_data(args.data), include_return=True)
    if args.subjects:
        full.subjects = dict(list(full.subjects.items())[: args.subjects])
    variants = {"with_return": full, "no_return": full.without_return()}
    lat = Latency()
    out: dict = {
        "scorer_doc": (scorer.__doc__ or "").strip().splitlines()[0:1],
        "n_subjects": len(full.subjects),
        "enroll_target": ENROLL_TARGET,
        "protocol1": {}, "protocol2": {}, "protocol2_window_last": {},
    }

    print(f"CMU keystroke benchmark — {len(full.subjects)} subjects, scorer: {out['scorer_doc']}")
    print("\nProtocol 1 — Killourhy & Maxion replication (train reps 1-200, test 201-400, 250 impostors)")
    print(f"  published scaled Manhattan EER {_pct(PUBLISHED_EER)} ± 6.9% (31 features, with Return)")
    for v, ds in variants.items():
        r = evaluate(ds, split_km, lat if v == "no_return" else None)
        out["protocol1"][v] = r
        print(_line(f"{v} ({len(ds.names)} feat)", r["aggregate"]))

    print("\nProtocol 2 — BioPrint-realistic: enroll N reps of session 1, genuine = sessions 2-8 (350), 250 impostors")
    for v, ds in variants.items():
        out["protocol2"][v] = {}
        for n in args.sizes:
            r = evaluate(ds, lambda s, n=n: split_bioprint(s, n, "first"))
            out["protocol2"][v][n] = r
            tag = " ★" if n == ENROLL_TARGET else ""
            print(_line(f"{v} N={n}{tag}", r["aggregate"]))
    print("  sensitivity: enroll on the LAST N reps of session 1 (subject has practised the password)")
    for n in [x for x in args.sizes if x < 50]:
        r = evaluate(variants["no_return"], lambda s, n=n: split_bioprint(s, n, "last"))
        out["protocol2_window_last"][n] = r
        print(_line(f"no_return N={n} last", r["aggregate"]))

    out["latency"] = lat.summary()
    L = out["latency"]
    print(f"\nLatency (protocol 1, no_return): fit mean {L['fit_ms']['mean']:.3f} ms p95 {L['fit_ms']['p95']:.3f} ms"
          f" | score mean {L['score_ms']['mean']:.3f} ms p95 {L['score_ms']['p95']:.3f} ms (n={L['score_ms']['n']})")

    n = ENROLL_TARGET if ENROLL_TARGET in args.sizes else args.sizes[0]
    head = out["protocol2"]["no_return"][n]["aggregate"]
    e = head["eer"]["mean"]
    out["headline"] = {
        "protocol": f"protocol2 no_return N={n}",
        "eer": e,
        "frr_at_far_0.01": head["frr_at_far_0.01"]["mean"],
        "frr_at_far_0.05": head["frr_at_far_0.05"]["mean"],
        "frr_model": head["frr_model"]["mean"],
        "far_model": head["far_model"]["mean"],
        "gate_a": gate_a(e),
    }
    h = out["headline"]
    print(f"\nHEADLINE ({h['protocol']}): FRR {_pct(h['frr_at_far_0.01'])} at FAR 1%,"
          f" FRR {_pct(h['frr_at_far_0.05'])} at FAR 5%, EER {_pct(e)};"
          f" live threshold gives FRR {_pct(h['frr_model'])} / FAR {_pct(h['far_model'])}")
    print(f"Gate A (cross-session EER): {h['gate_a']}")
    print(f"  model thr outliers: {head['subjects_with_frr_model_over_20pct']} subjects FRR>20%,"
          f" {head['subjects_with_far_model_over_5pct']} subjects FAR>5%")

    dest = args.json or (RESULTS / f"cmu-{time.strftime('%Y%m%d-%H%M%S')}.json" if args.save else None)
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1, default=str))
        print(f"\nwrote {dest}")
    return out


if __name__ == "__main__":
    main()
