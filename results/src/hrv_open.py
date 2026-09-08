"""
HRV branch on the two OPEN PhysioNet Novak-lab diabetes cohorts (CC BY 4.0):

  GE-71  Cerebral Vasoregulation in Diabetes  (already processed: results/out/hrv_features.csv,
         labels: DM vs control, DR yes/no, neuropathy, HbA1c; 500 Hz timebase fix applied)
  GE-75  Cerebral Perfusion and Cognitive Decline in T2DM (this script): 24-h Holter-type
         ME6000 ECG records (WFDB, 8 channels, 1000 Hz), summary table with Group (75 DM /
         13 control), diabetes duration, HbA1c, insulin, autonomic symptoms and an ETDRS-style
         retinopathy grade per eye (52 graded).

Pipeline (identical feature extraction to hrv_branch.py so the two cohorts pool):
  first 5 min of the record -> neurokit2 clean + R-peaks -> RR artefact gate -> SDNN, RMSSD,
  pNN50, CVNN, LF, HF, LF/HF. Channel choice: the ECG channel whose detected HR is
  physiological and whose RR series has the lowest artefact rate. Timebase plausibility gate:
  if HR at the header fs is implausible (>110 or <35 bpm) the record is retried at fs/2, and
  the correction is logged (the GE-71 cohort needed it; GE-75 is checked, not assumed).

Analyses (all subject-level, aggregate outputs only):
  1. DM vs control (pooled n ~ 125): logistic regression on z-scored HRV, leave-one-out AUC + bootstrap CI.
  2. DR+ vs DR- within diabetics (pooled): effect sizes (Cohen d), Mann-Whitney p with BH correction,
     LOO AUC.  This is the on-thesis test: non-retinal signal carrying retinopathy information.
  3. Autonomic symptoms (GE-75) and HbA1c correlation as secondary.

Run (nrdi-env):  python results/src/hrv_open.py
"""
import sys, json, pathlib, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
GE75 = ROOT / "data" / "raw" / "physionet" / "cerebral-perfusion-diabetes" / "1.0.0"
OUT = ROOT / "out"
BASELINE_MIN = 5
FEATS = ["hr_mean", "sdnn", "rmssd", "pnn50", "cvnn", "lf_hf"]


def load_ge75_labels():
    t = pd.read_csv(GE75 / "data_description" / "GE-75_data_summary_table.csv", encoding="latin-1")
    g = t["Retinopathy Grading"] if "Retinopathy Grading" in t else t["Diabetic Retinopathy (more advanced eye)"]
    adv = pd.to_numeric(t["Diabetic Retinopathy (more advanced eye)"], errors="coerce")
    out = pd.DataFrame({
        "subject": t["patient ID"].astype(str).str.strip().str.upper(),
        "cohort": "GE-75",
        "diabetic": t["Group"].astype(str).str.strip().str.upper().eq("DM").astype(int),
        "age": pd.to_numeric(t["age"], errors="coerce"),
        "female": t["sex"].astype(str).str.strip().str.lower().str.startswith("f").astype(int),
        "hba1c": pd.to_numeric(t["Hb A1C%"], errors="coerce"),
        "dm_years": pd.to_numeric(t["Diabetes Duration"], errors="coerce"),
        "insulin": t["INSULIN(Yes_or_No)"].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0}),
        "autonomic_symptoms": t["Neuropathy AUTONOMIC SYMPTOMS"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}),
        "dr_grade_adv_eye": adv,
    })
    # ETDRS-style scale in this table: 0 = none; 9 = unable to grade -> NaN
    out["dr"] = np.where(out["dr_grade_adv_eye"].isna() | (out["dr_grade_adv_eye"] >= 9), np.nan,
                         (out["dr_grade_adv_eye"] > 0).astype(float))
    return out


def read_wfdb16(path):
    """Minimal WFDB reader for these records (format 16, interleaved int16, gain 1/uV).
    wfdb.rdrecord rejects the header's non-standard base_date ('2006')."""
    hea = pathlib.Path(str(path) + ".hea").read_text().splitlines()
    name, n_sig, fs, n_samp = hea[0].split()[:4]
    n_sig, fs, n_samp = int(n_sig), float(fs), int(n_samp)
    sig = np.fromfile(pathlib.Path(str(path) + ".dat"), dtype="<i2")
    sig = sig[: (len(sig) // n_sig) * n_sig].reshape(-1, n_sig).astype(float)
    return sig, fs


def hrv_from_record(path):
    import neurokit2 as nk
    p_signal, fs0 = read_wfdb16(path)
    best = None
    for fs in (fs0, fs0 / 2.0):
        n = int(min(p_signal.shape[0], BASELINE_MIN * 60 * fs))
        for ch in range(p_signal.shape[1]):
            seg = p_signal[:n, ch].astype(float)
            if not np.isfinite(seg).any() or np.nanstd(seg) < 1e-9:
                continue
            seg = np.nan_to_num(seg, nan=np.nanmedian(seg))
            try:
                cleaned = nk.ecg_clean(seg, sampling_rate=fs)
                _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
            except Exception:
                continue
            peaks = np.asarray(info["ECG_R_Peaks"])
            if len(peaks) < 60:
                continue
            rr = np.diff(peaks) / fs * 1000.0
            rr = rr[(rr > 300) & (rr < 2000)]
            if len(rr) < 50:
                continue
            keep = np.abs(np.diff(rr)) / rr[:-1] < 0.2
            art = 1.0 - keep.mean()
            rr_c = rr[1:][keep]
            if len(rr_c) < 50:
                continue
            hr = 60000.0 / rr_c.mean()
            if not (35.0 <= hr <= 110.0):
                continue
            sd = np.diff(rr_c)
            f = {"hr_mean": hr, "sdnn": rr_c.std(ddof=1), "rmssd": float(np.sqrt(np.mean(sd ** 2))),
                 "pnn50": 100.0 * np.mean(np.abs(sd) > 50), "cvnn": rr_c.std(ddof=1) / rr_c.mean() * 100,
                 "n_beats": float(len(rr_c)), "artefact_rate": art, "channel": ch, "fs_used": fs}
            try:
                hf = nk.hrv_frequency(peaks, sampling_rate=fs, show=False)
                f["lf"], f["hf"], f["lf_hf"] = float(hf["HRV_LF"].iloc[0]), float(hf["HRV_HF"].iloc[0]), float(hf["HRV_LFHF"].iloc[0])
            except Exception:
                f["lf"] = f["hf"] = f["lf_hf"] = np.nan
            if best is None or art < best["artefact_rate"]:
                best = f
        if best is not None:
            break
    return best


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


PRED = {}


def loo_auc(X, y, seed=20260828, n_boot=1000, tag=None):
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[m], y[m].astype(int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    if tag:
        PRED[tag] = (np.asarray(y), np.asarray(p))
    rng = np.random.default_rng(seed); b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            b.append(roc_auc_score(y[i], p[i]))
    return {"n": int(len(y)), "n_pos": int(y.sum()), "auc": float(auc),
            "auc_ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def main():
    lab = load_ge75_labels()
    rows, notes = [], {"fs_halved": 0, "unusable": []}
    for _, r in lab.iterrows():
        hea = GE75 / "data" / "ecg" / f"{r.subject}ECG.hea"
        if not hea.exists():
            continue
        f = hrv_from_record(hea.with_suffix(""))
        if f is None:
            notes["unusable"].append(r.subject); continue
        if f["fs_used"] < 999:
            notes["fs_halved"] += 1
        rows.append({"subject": r.subject, **f})
    hrv75 = pd.DataFrame(rows)
    df75 = lab.merge(hrv75, on="subject", how="inner")
    print(f"GE-75: {len(df75)} subjects with usable ECG (of {lab.subject.nunique()} in table; "
          f"{len(notes['unusable'])} unusable; fs halved on {notes['fs_halved']}); DM {int(df75.diabetic.sum())}, "
          f"DR graded {int(df75.dr.notna().sum())}, DR+ {int(np.nansum(df75.dr))}")
    df75.to_csv(OUT / "hrv_features_ge75.csv", index=False)

    # pool with GE-71
    f71 = pd.read_csv(OUT / "hrv_features.csv") if (OUT / "hrv_features.csv").exists() else pd.DataFrame()
    if len(f71):
        f71 = f71.rename(columns={c: c for c in f71.columns}); f71["cohort"] = "GE-71"
        pooled = pd.concat([df75, f71], ignore_index=True, sort=False)
    else:
        pooled = df75
    res = {"ge75_n": int(len(df75)), "pooled_n": int(len(pooled)), "notes": notes, "features": FEATS}

    # 1. DM vs control
    X = pooled[FEATS].values.astype(float); res["dm_vs_control"] = loo_auc(X, pooled["diabetic"].values.astype(float), tag="dm_vs_control")
    # 2. DR+ vs DR- within diabetics
    d = pooled[(pooled.diabetic == 1) & pooled.dr.notna()]
    res["dr_within_dm"] = {"n": int(len(d)), "n_dr": int(d.dr.sum()), "auc": loo_auc(d[FEATS].values.astype(float), d.dr.values.astype(float), tag="dr_within_dm"),
                           "effects": {}}
    ps = []
    for f in FEATS:
        a, b = d.loc[d.dr == 1, f].dropna(), d.loc[d.dr == 0, f].dropna()
        if len(a) >= 5 and len(b) >= 5:
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            res["dr_within_dm"]["effects"][f] = {"d": cohen_d(a, b), "p": float(p), "median_dr": float(a.median()), "median_no_dr": float(b.median())}
            ps.append((f, p))
    if ps:
        order = sorted(ps, key=lambda t: t[1]); m = len(ps)
        for rank, (f, p) in enumerate(order, 1):
            res["dr_within_dm"]["effects"][f]["p_bh"] = float(min(1.0, p * m / rank))
    # 3. autonomic symptoms + HbA1c (GE-75 only)
    a = df75[df75.autonomic_symptoms.notna()]
    res["autonomic_symptoms_ge75"] = loo_auc(a[FEATS].values.astype(float), a.autonomic_symptoms.values.astype(float), tag="autonomic_symptoms")
    h = df75[df75.hba1c.notna()]
    res["hba1c_spearman"] = {f: {"rho": float(stats.spearmanr(h[f], h.hba1c, nan_policy="omit")[0]),
                                 "p": float(stats.spearmanr(h[f], h.hba1c, nan_policy="omit")[1])} for f in FEATS}
    np.savez(OUT / "hrv_pred.npz", **{f"{k}_{n}": v[i] for k, v in PRED.items() for i, n in enumerate(("y", "p"))})
    (OUT / "hrv_open.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("DM vs control:", res["dm_vs_control"])
    print("DR within DM:", {k: v for k, v in res["dr_within_dm"].items() if k != "effects"})
    for f, e in res["dr_within_dm"]["effects"].items():
        print(f"  {f:8s} d={e['d']:+.2f} p={e['p']:.3f} p_bh={e.get('p_bh', np.nan):.3f}")
    print("autonomic symptoms:", res["autonomic_symptoms_ge75"])
    print("wrote hrv_open.json + hrv_features_ge75.csv")


if __name__ == "__main__":
    main()
