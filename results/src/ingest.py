"""
A5 — ingest BRSET v1.0.2 and mBRSET v1.0 (PhysioNet, credentialed).

What it does (and only this):
  1. verifies the PhysioNet SHA256SUMS.txt of each download (integrity),
  2. loads the two label CSVs, maps them onto ONE schema (results/DATA_RETRIEVAL.md §5c),
  3. applies the quality gate, builds frozen PATIENT-LEVEL 5-fold splits per dataset
     (StratifiedGroupKFold on the ICDR grade, grades 3+4 collapsed when a fold would
     otherwise get < 8 patients of a grade),
  4. writes results/data/interim/{labels_unified.csv, splits_brset.csv, splits_mbrset.csv,
     ingest_report.json} and prints AGGREGATE statistics only.

Compliance (PhysioNet DUA 1.5.0 + LLM guidance 2025-09-24): raw rows and images never
leave this machine and are never printed; this script prints counts and rates only. The
interim files are derived restricted data — keep them out of any public repo.

Expected layout (default; override with --brset / --mbrset):
  results/data/raw/physionet/brazilian-ophthalmological/1.0.2/   labels.csv, fundus_photos/, SHA256SUMS.txt
  results/data/raw/physionet/mbrset/1.0/                         labels_mbrset.csv, <images>/, SHA256SUMS.txt

Run:  python results/src/ingest.py [--brset DIR] [--mbrset DIR] [--skip-sha] [--seed 20260828]
"""
import sys, json, hashlib, pathlib, argparse, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "physionet"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

EXPECTED = {"brset": {"rows": 16266, "patients": 8524}, "mbrset": {"rows": 5164, "patients": 1291}}


# ------------------------------------------------------------------ sha256 --
def verify_sha256(root: pathlib.Path, max_files=None):
    """Verify SHA256SUMS.txt (PhysioNet format: '<hash>  <relative path>')."""
    f = root / "SHA256SUMS.txt"
    if not f.exists():
        return {"status": "no SHA256SUMS.txt", "checked": 0, "bad": []}
    bad, n = [], 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, rel = line.split(maxsplit=1)
        p = root / rel.strip()
        if not p.exists():
            bad.append(rel); continue
        d = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                d.update(chunk)
        if d.hexdigest() != h:
            bad.append(rel)
        n += 1
        if max_files and n >= max_files:
            break
    return {"status": "ok" if not bad else "MISMATCH", "checked": n, "bad": bad[:20]}


# --------------------------------------------------------------- loaders ---
def _yesno(s):
    return s.map(lambda v: 1 if str(v).strip().lower() in ("1", "yes", "true", "y") else
                 (0 if str(v).strip().lower() in ("0", "no", "false", "n") else np.nan))


def load_brset(d: pathlib.Path):
    df = pd.read_csv(d / "labels.csv")
    need = ["image_id", "patient_id", "DR_ICDR", "macular_edema", "exam_eye"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"BRSET labels.csv is missing {missing}: are you on v1.0.2? columns = {list(df.columns)[:12]}...")
    q_cols = [c for c in ("focus", "illumination", "image_field", "artifacts") if c in df.columns]
    u = pd.DataFrame({
        "dataset": "brset",
        "image_file": df["image_id"].astype(str),
        "pid": "b_" + df["patient_id"].astype(str),
        "eye": df["exam_eye"].map({1: "R", 2: "L"}),
        "dr_icdr": pd.to_numeric(df["DR_ICDR"], errors="coerce"),
        "dme": pd.to_numeric(df["macular_edema"], errors="coerce"),
        "diabetes": pd.to_numeric(df["diabetes"], errors="coerce") if "diabetes" in df else np.nan,
        "dm_years": pd.to_numeric(df.get("diabetes_time"), errors="coerce"),
        "insulin": pd.to_numeric(df.get("insulin_use"), errors="coerce"),
        "age": pd.to_numeric(df.get("patient_age"), errors="coerce"),
        "sex_male": df["patient_sex"].map({1: 1, 2: 0}) if "patient_sex" in df else np.nan,
        "neuropathy": np.nan,
        "device": df.get("camera", "unknown"),
        "quality_ok": (df[q_cols] != 2).all(axis=1).astype(int) if q_cols else 1,
    })
    return u, {"columns": list(df.columns)}


def load_mbrset(d: pathlib.Path):
    df = pd.read_csv(d / "labels_mbrset.csv")
    need = ["patient", "file", "final_icdr", "final_edema", "laterality"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"mBRSET labels_mbrset.csv is missing {missing}: columns = {list(df.columns)}")
    lat = df["laterality"].astype(str).str.strip().str.lower().map(
        lambda v: "R" if v.startswith("r") or v == "1" else ("L" if v.startswith("l") or v == "2" else None))
    qual = df["final_quality"].astype(str).str.strip().str.lower() if "final_quality" in df else None
    u = pd.DataFrame({
        "dataset": "mbrset",
        "image_file": df["file"].astype(str),
        "pid": "m_" + df["patient"].astype(str),
        "eye": lat,
        "dr_icdr": pd.to_numeric(df["final_icdr"], errors="coerce"),
        "dme": _yesno(df["final_edema"]),
        "diabetes": 1,                                   # mBRSET enrolled diabetics only
        "dm_years": pd.to_numeric(df.get("dm_time"), errors="coerce"),
        "insulin": pd.to_numeric(df.get("insulin"), errors="coerce"),
        "age": pd.to_numeric(df.get("age"), errors="coerce"),
        "sex_male": pd.to_numeric(df.get("sex"), errors="coerce"),
        "neuropathy": pd.to_numeric(df.get("neuropathy"), errors="coerce"),
        "device": "Phelcom Eyer",
        "quality_ok": (~qual.isin(["inadequate", "no", "0", "false"])).astype(int) if qual is not None else 1,
    })
    return u, {"columns": list(df.columns), "final_edema_coding": sorted(df["final_edema"].astype(str).unique().tolist())[:6],
               "laterality_coding": sorted(df["laterality"].astype(str).unique().tolist())[:6]}


# ---------------------------------------------------------------- splits ---
def make_splits(u: pd.DataFrame, seed: int, n_splits=5, min_per_fold=8):
    u = u.dropna(subset=["dr_icdr"]).copy()
    y = u["dr_icdr"].astype(int).values
    # collapse 3+4 when severe NPDR is too rare to stratify
    counts = pd.Series(y).value_counts()
    strat = y.copy()
    collapsed = False
    if counts.get(3, 0) < n_splits * min_per_fold:
        strat = np.where(strat >= 3, 3, strat); collapsed = True
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(u), -1)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(u)), strat, u["pid"].values)):
        fold[te] = k
    u["fold"] = fold
    assert (u.groupby("pid")["fold"].nunique() == 1).all(), "patient leakage across folds"
    return u[["dataset", "image_file", "pid", "fold"]], collapsed


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brset", default=str(RAW / "brazilian-ophthalmological" / "1.0.2"))
    ap.add_argument("--mbrset", default=str(RAW / "mbrset" / "1.0"))
    ap.add_argument("--skip-sha", action="store_true")
    ap.add_argument("--seed", type=int, default=20260828)
    a = ap.parse_args()
    t0 = time.time()
    report = {"seed": a.seed, "datasets": {}}
    frames = []
    for name, d, loader in (("brset", pathlib.Path(a.brset), load_brset), ("mbrset", pathlib.Path(a.mbrset), load_mbrset)):
        if not d.exists():
            print(f"[{name}] NOT FOUND at {d} — download per results/DATA_RETRIEVAL.md, then rerun.")
            continue
        sha = {"status": "skipped"} if a.skip_sha else verify_sha256(d)
        u, meta = loader(d)
        exp = EXPECTED[name]
        img_dirs = [p for p in d.iterdir() if p.is_dir()]
        n_img = sum(len(list(p.glob("*.jp*g"))) for p in img_dirs)
        splits, collapsed = make_splits(u, a.seed)
        splits.to_csv(INTERIM / f"splits_{name}.csv", index=False)
        u = u.merge(splits[["image_file", "fold"]], on="image_file", how="left")
        frames.append(u)
        dia = u[u["diabetes"] == 1] if u["diabetes"].notna().any() else u
        rep = {
            "dir": str(d), "sha256": sha, "rows": int(len(u)), "patients": int(u["pid"].nunique()),
            "expected": exp, "counts_match": bool(len(u) == exp["rows"] and u["pid"].nunique() == exp["patients"]),
            "images_on_disk": int(n_img), "image_dirs": [p.name for p in img_dirs],
            "quality_ok_rate": float(u["quality_ok"].mean()),
            "dr_icdr_counts": {int(k): int(v) for k, v in u["dr_icdr"].value_counts().sort_index().items()},
            "dr_any_rate_diabetics": float((dia["dr_icdr"] > 0).mean()) if len(dia) else None,
            "dme_rate": float(u["dme"].mean()) if u["dme"].notna().any() else None,
            "neuropathy_rate": float(u["neuropathy"].mean()) if u["neuropathy"].notna().any() else None,
            "dm_years_median": float(u["dm_years"].median()) if u["dm_years"].notna().any() else None,
            "splits": {"n_folds": 5, "grades_3_4_collapsed": collapsed,
                       "patients_per_fold": splits.groupby("fold")["pid"].nunique().tolist()},
            "meta": meta,
        }
        report["datasets"][name] = rep
        print(f"[{name}] rows {rep['rows']:,} / patients {rep['patients']:,} (expected {exp['rows']:,}/{exp['patients']:,}: "
              f"{'OK' if rep['counts_match'] else 'MISMATCH'}); images on disk {n_img:,}; sha256 {sha['status']}")
        print(f"         ICDR counts {rep['dr_icdr_counts']}; DME rate {rep['dme_rate']}; quality ok {100*rep['quality_ok_rate']:.1f} %; "
              f"neuropathy rate {rep['neuropathy_rate']}; folds {rep['splits']['patients_per_fold']}")
    if frames:
        allu = pd.concat(frames, ignore_index=True)
        allu["dr_any"] = (allu["dr_icdr"] > 0).astype("Int64")
        allu["dr_referable"] = ((allu["dr_icdr"] >= 2) | (allu["dme"] == 1)).astype("Int64")
        allu.to_csv(INTERIM / "labels_unified.csv", index=False)
        report["unified"] = {"rows": int(len(allu)), "columns": list(allu.columns)}
    report["time_s"] = time.time() - t0
    (INTERIM / "ingest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {INTERIM/'ingest_report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
