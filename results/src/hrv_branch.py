"""
HRV branch — first real per-modality result (step A6, HRV).

Data    Cerebral Vasoregulation in Diabetes (PhysioNet, CC BY 4.0).
        Day-1 head-up-tilt records; protocol opens with a supine baseline,
        so the first BASELINE_MIN minutes are used as the resting segment.
Labels  GE-71_Data_Summary_Table.csv: Group (DM/DMOH/Control),
        DR yes/no + grade, neuropathy, HbA1c.

Outputs
  1. Diabetes-status AUC from HRV features (28 diabetic vs 22 control)
     - the HRV branch's standalone number for the modality table
  2. Exploratory HRV->retinopathy association (10 yes vs 25 no):
     effect sizes + CI only; n is too small for a classifier claim
  3. hrv_features.csv for the fusion model later

Run:  python results/src/hrv_branch.py
"""
import pathlib, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent / "data" / "raw" / "physionet" / "cerebral-vasoregulation-in-diabetes-1.0.0"
REC_DIR = BASE / "Data" / "Labview" / "Converted" / "Head-up-tilt_Day1"
OUT = HERE.parent / "out"
SEED = 20260828
BASELINE_MIN = 5.0        # supine baseline at the start of the tilt protocol


def load_labels():
    df = pd.read_csv(BASE / "Data_Description" / "GE-71_Data_Summary_Table.csv",
                     encoding="latin-1")
    df = df[df["Inclusion"].astype(str).str.upper().str.startswith("YES")]
    out = pd.DataFrame({
        "subject": df["SUBJECT NUMBER"].astype(str).str.strip().str.upper(),
        "group": df["Group"].astype(str).str.strip(),
        "age": pd.to_numeric(df["Age"], errors="coerce"),
        "female": df["Gender"].astype(str).str.strip().str.lower().eq("f").astype(int),
        "diabetic": df["Group"].astype(str).str.strip().isin(["DM", "DMOH"]).astype(int),
        "dr": df["Diabetic retinopathy (yes/no)"].astype(str).str.strip().str.lower()
              .map({"yes": 1, "no": 0}),
        "neuropathy": df["Neuropathy"].astype(str).str.strip().str.lower()
                      .map({"yes": 1, "no": 0}),
        "hba1c": pd.to_numeric(df["HbA1c"], errors="coerce"),
    })
    return out


def extract_hrv(rec_path, fs_note=[]):
    """Read a WFDB record, find the ECG channel, return HRV features from the
    supine-baseline segment. Returns None on unusable records."""
    import wfdb
    import neurokit2 as nk

    rec = wfdb.rdrecord(str(rec_path))
    names = [s.strip().upper() for s in rec.sig_name]
    if "ECG" not in names:
        return None
    ecg = rec.p_signal[:, names.index("ECG")].astype(float)
    # TIMEBASE CORRECTION: headers claim 1000 Hz but the dataset's own
    # File_and_channels.csv documents 500 Hz for these Labview protocols.
    # Uncorrected, apparent supine HR is 140-152 bpm (implausible for ages
    # 55-75) and QRS half-max widths are 18-30 ms (3x too narrow); at 500 Hz
    # both land in physiological range. Audited 2026-09-01.
    fs = float(rec.fs) / 2.0
    fs_note.append(fs)

    n = int(min(len(ecg), BASELINE_MIN * 60 * fs))
    seg = ecg[:n]
    if np.all(~np.isfinite(seg)) or np.nanstd(seg) < 1e-9:
        return None
    seg = np.nan_to_num(seg, nan=np.nanmedian(seg))

    cleaned = nk.ecg_clean(seg, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    peaks = info["ECG_R_Peaks"]
    if len(peaks) < 60:                       # < ~1 beat/5s over 5 min = garbage
        return None

    rr = np.diff(peaks) / fs * 1000.0         # ms
    # physiological gate + artefact rejection (successive-difference > 20%)
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 50:
        return None
    keep = np.abs(np.diff(rr)) / rr[:-1] < 0.2
    rr_c = rr[1:][keep]
    if len(rr_c) < 50:
        return None

    # plausibility gate: supine HR must be physiological, else flag the record
    hr = 60000.0 / rr_c.mean()
    if not (35.0 <= hr <= 110.0):
        return None

    sd = np.diff(rr_c)
    feats = {
        "hr_mean": 60000.0 / rr_c.mean(),
        "sdnn": rr_c.std(ddof=1),
        "rmssd": np.sqrt(np.mean(sd ** 2)),
        "pnn50": 100.0 * np.mean(np.abs(sd) > 50),
        "cvnn": rr_c.std(ddof=1) / rr_c.mean() * 100,
        "n_beats": float(len(rr_c)),
    }
    # frequency domain via neurokit (Welch on interpolated RR)
    try:
        hf = nk.hrv_frequency(peaks, sampling_rate=fs, show=False)
        for k_src, k_dst in [("HRV_LF", "lf"), ("HRV_HF", "hf"),
                             ("HRV_LFHF", "lf_hf")]:
            v = float(hf[k_src].iloc[0]) if k_src in hf else np.nan
            feats[k_dst] = v
    except Exception:
        feats.update({"lf": np.nan, "hf": np.nan, "lf_hf": np.nan})
    return feats


def main():
    labels = load_labels()
    headers = sorted(REC_DIR.glob("*.hea"))
    print(f"records found: {len(headers)}   labelled subjects: {len(labels)}")

    fs_note, rows, skipped = [], [], []
    for h in headers:
        stem = h.stem                          # e.g. s0030DA
        subj = stem[:5].upper()                # S0030
        feats = extract_hrv(REC_DIR / stem, fs_note)
        if feats is None:
            skipped.append(stem)
            continue
        feats["subject"] = subj
        rows.append(feats)

    feat = pd.DataFrame(rows)
    # one record per subject (some subjects have repeats: keep first)
    feat = feat.drop_duplicates("subject", keep="first")
    df = feat.merge(labels, on="subject", how="inner")
    print(f"usable ECG baselines: {len(feat)}   joined with labels: {len(df)}")
    print(f"sampling rates seen: {sorted(set(fs_note))}")
    if skipped:
        print(f"skipped (no/poor ECG): {len(skipped)}: {', '.join(skipped[:8])}"
              + (" ..." if len(skipped) > 8 else ""))

    df.to_csv(OUT / "hrv_features.csv", index=False)

    FEATS = ["hr_mean", "sdnn", "rmssd", "pnn50", "cvnn", "lf_hf"]

    # ---- 1. group descriptives (the clinical sanity check) -----------------
    print("\n=== HRV by diabetes status (mean +/- sd) ===")
    for f in FEATS:
        a = df.loc[df.diabetic == 1, f].dropna()
        b = df.loc[df.diabetic == 0, f].dropna()
        print(f"  {f:8s} DM {a.mean():8.2f} ± {a.std():6.2f}   "
              f"Ctl {b.mean():8.2f} ± {b.std():6.2f}")

    # ---- 2. diabetes-status AUC (leave-one-out, n is small) ----------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import roc_auc_score

    X = df[FEATS].to_numpy(float)
    y = df["diabetic"].to_numpy()
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000))])
    p = cross_val_predict(pipe, X, y, cv=LeaveOneOut(),
                          method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    boots = []
    for _ in range(2000):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n=== HRV -> diabetes status (LOO-CV) ===")
    print(f"  n = {len(y)}  ({int(y.sum())} diabetic / {int((1-y).sum())} control)")
    print(f"  AUC = {auc:.3f}  [{lo:.3f}, {hi:.3f}]")

    # ---- 3. exploratory: HRV vs retinopathy (effect sizes only) ------------
    sub = df[df.dr.notna()]
    print(f"\n=== exploratory: HRV vs retinopathy (n={len(sub)}, "
          f"{int(sub.dr.sum())} DR+) — effect sizes, NOT a classifier ===")
    from scipy import stats
    dr_rows = []
    for f in FEATS:
        a = sub.loc[sub.dr == 1, f].dropna()
        b = sub.loc[sub.dr == 0, f].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        pooled = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var())
                         / (len(a)+len(b)-2))
        d = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
        u, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        dr_rows.append({"feature": f, "dr_mean": a.mean(), "nodr_mean": b.mean(),
                        "cohens_d": d, "mannwhitney_p": pval})
        print(f"  {f:8s} DR+ {a.mean():8.2f}  DR- {b.mean():8.2f}   "
              f"d = {d:+.2f}   p = {pval:.3f}")

    json.dump({
        "n_total": int(len(df)),
        "n_diabetic": int(y.sum()), "n_control": int((1 - y).sum()),
        "diabetes_auc_loo": float(auc), "diabetes_auc_ci95": [float(lo), float(hi)],
        "features": FEATS,
        "dr_exploratory": dr_rows,
        "baseline_minutes": BASELINE_MIN,
        "artefact_rule": "RR in 300-2000 ms; successive diff < 20%",
        "seed": SEED,
    }, open(OUT / "hrv_branch.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT/'hrv_features.csv'} and hrv_branch.json")


if __name__ == "__main__":
    main()
