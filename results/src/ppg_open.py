"""
PPG branch on PPG-BP (Liang et al. 2018, Sci Data; figshare 5459299; CC0):
219 subjects, 3 x 2.1-s fingertip PPG segments each (1 kHz), with age, sex, BMI, BP, HR,
hypertension stage and a diabetes flag (38 type-2 diabetics).

Two feature routes, both subject-level (segments averaged), 5-fold stratified CV repeated
with bootstrap CIs; the diabetes model is reported unadjusted AND adjusted for age/sex/BMI,
next to a demographics-only baseline, because diabetes is confounded by age in this cohort.

  A. Morphology (12 handcrafted features): pulse rise time, width at half height, systolic
     amplitude ratio, dicrotic notch ratio, reflection index, second-derivative ratios
     b/a, c/a, d/a, e/a and the ageing index (b-c-d-e)/a, crest time, HR.
  B. PaPaGei-S frozen embedding (512-d -> PCA 16): the 2.1-s segment is resampled to 125 Hz
     and tiled to PaPaGei's 10-s window (stated limitation: tiling repeats the same beats).
     Positive control: age regression r from the embedding (PWDB gave r = 0.96).

Run (system python, GPU optional):  python results/src/ppg_open.py
"""
import sys, json, pathlib, warnings
import numpy as np, pandas as pd
from scipy.signal import resample, find_peaks, butter, filtfilt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "raw" / "open" / "ppg_bp" / "Data File"
OUT = ROOT / "out"
SEED = 20260828
FS = 1000.0


def load_meta():
    t = pd.read_excel(BASE / "PPG-BP dataset.xlsx", header=1)
    t.columns = [str(c).strip() for c in t.columns]
    m = pd.DataFrame({"sid": t["subject_ID"].astype(int),
                      "female": t["Sex(M/F)"].astype(str).str.strip().str.upper().eq("F").astype(int),
                      "age": pd.to_numeric(t["Age(year)"], errors="coerce"),
                      "bmi": pd.to_numeric(t["BMI(kg/m^2)"], errors="coerce"),
                      "sbp": pd.to_numeric(t["Systolic Blood Pressure(mmHg)"], errors="coerce"),
                      "hr": pd.to_numeric(t["Heart Rate(b/m)"], errors="coerce"),
                      "diabetes": t["Diabetes"].astype(str).str.lower().str.contains("diab").astype(int),
                      "hypertension": (~t["Hypertension"].astype(str).str.strip().str.lower().isin(["normal", "nan"])).astype(int)})
    return m


def bandpass(x, lo=0.5, hi=12.0):
    b, a = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def morphology(x):
    """12 pulse-morphology features from one 2.1-s segment (mean over beats)."""
    x = bandpass(x - x.mean())
    peaks, _ = find_peaks(x, distance=int(0.4 * FS), prominence=np.ptp(x) * 0.3)
    feats = []
    for i in range(len(peaks)):
        p = peaks[i]
        lo = max(0, p - int(0.6 * FS)); hi = min(len(x) - 1, p + int(0.6 * FS))
        foot = lo + int(np.argmin(x[lo:p])) if p > lo else lo
        nxt = peaks[i + 1] if i + 1 < len(peaks) else None
        end = hi if nxt is None else min(nxt, hi)
        beat = x[foot:end]
        if len(beat) < int(0.3 * FS):
            continue
        amp = x[p] - x[foot]
        rise = (p - foot) / FS
        half = x[foot] + amp / 2
        above = np.where(beat >= half)[0]
        width = (above[-1] - above[0]) / FS if len(above) > 1 else np.nan
        # dicrotic notch: local minimum after the peak
        after = x[p:end]
        notch = None
        if len(after) > int(0.1 * FS):
            mins, _ = find_peaks(-after[int(0.08 * FS):], distance=int(0.05 * FS))
            if len(mins):
                notch = p + int(0.08 * FS) + mins[0]
        ri = (x[notch] - x[foot]) / amp if notch is not None else np.nan
        # second derivative waves a,b,c,d,e
        d2 = np.gradient(np.gradient(x[foot:end]))
        w = d2[: int(0.35 * FS)] if len(d2) > int(0.35 * FS) else d2
        ap, _ = find_peaks(w); an, _ = find_peaks(-w)
        a_ = w[ap[0]] if len(ap) else np.nan
        b_ = w[an[0]] if len(an) and (len(ap) and an[0] > ap[0]) else np.nan
        c_ = w[ap[1]] if len(ap) > 1 else np.nan
        d_ = w[an[1]] if len(an) > 1 else np.nan
        e_ = w[ap[2]] if len(ap) > 2 else np.nan
        with np.errstate(all="ignore"):
            feats.append([rise, width, amp / np.ptp(x), ri, b_ / a_, c_ / a_, d_ / a_, e_ / a_,
                          (b_ - c_ - d_ - e_) / a_, (p - foot) / (end - foot), 60.0 / ((end - foot) / FS)])
    if not feats:
        return None
    return np.nanmean(np.array(feats, float), axis=0)


MORPH = ["rise_time_s", "width_half_s", "amp_ratio", "reflection_index", "b_a", "c_a", "d_a", "e_a", "ageing_index", "crest_frac", "hr_est"]


def papagei_embed(segments):
    import torch
    sys.path.insert(0, str(ROOT / "vendor" / "papagei"))
    from models.resnet import ResNet1DMoE
    m = ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1, n_block=18, n_classes=512, n_experts=3)
    sd = torch.load(ROOT / "data" / "raw" / "papagei" / "papagei_s.pt", map_location="cpu", weights_only=True)
    m.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}); m.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); m.to(dev)
    X = []
    for x in segments:
        r = resample(x, int(len(x) / FS * 125))              # 2.1 s @ 125 Hz = 262 samples
        s = np.tile(r, int(np.ceil(1250 / len(r))))[:1250]
        s = (s - s.mean()) / (s.std() + 1e-8)
        X.append(s)
    X = torch.tensor(np.array(X, np.float32))[:, None, :]
    with torch.no_grad():
        E = np.concatenate([m(X[i:i + 64].to(dev))[0].cpu().numpy() for i in range(0, len(X), 64)])
    return E


def auc_ci(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


PRED = {}


def cv_auc(X, y, C=0.5, tag=None):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    p = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    if tag:
        PRED[tag] = (np.asarray(y), np.asarray(p))
    return {"auc": float(roc_auc_score(y, p)), "auc_ci": auc_ci(y, p), "n": int(len(y)), "n_pos": int(y.sum())}


def main():
    meta = load_meta()
    seg_rows, seg_sig, seg_sid = [], [], []
    for f in sorted((BASE / "0_subject").glob("*.txt")):
        sid = int(f.stem.split("_")[0])
        x = np.array(f.read_text().split(), float)
        if len(x) < 1500:
            continue
        mf = morphology(x)
        if mf is None:
            continue
        seg_rows.append([sid, *mf]); seg_sig.append(x[:2100]); seg_sid.append(sid)
    segs = pd.DataFrame(seg_rows, columns=["sid", *MORPH])
    subj = segs.groupby("sid").mean().reset_index().merge(meta, on="sid")
    print(f"PPG-BP: {len(segs)} segments, {len(subj)} subjects with morphology, diabetic {int(subj.diabetes.sum())}, hypertensive {int(subj.hypertension.sum())}")
    E = papagei_embed(seg_sig)
    Es = pd.DataFrame(E).groupby(np.array(seg_sid)).mean()
    Es = Es.loc[subj.sid.values].values
    pca = PCA(16, random_state=SEED).fit_transform(StandardScaler().fit_transform(Es))
    y = subj.diabetes.values.astype(int); demo = subj[["age", "female", "bmi"]].fillna(subj[["age", "female", "bmi"]].median()).values
    Xm = subj[MORPH].fillna(subj[MORPH].median()).values
    res = {"source": "PPG-BP (figshare 5459299, CC0), Liang et al. Sci Data 2018", "n_subjects": int(len(subj)), "n_diabetic": int(y.sum()),
           "diabetes": {"demographics_only": cv_auc(demo, y, tag="dm_demographics_only"), "morphology": cv_auc(Xm, y, tag="dm_morphology"), "morphology_plus_demo": cv_auc(np.hstack([Xm, demo]), y, tag="dm_morphology_plus_demo"),
                        "papagei_pca16": cv_auc(pca, y, tag="dm_papagei_pca16"), "papagei_plus_demo": cv_auc(np.hstack([pca, demo]), y, tag="dm_papagei_plus_demo")},
           "hypertension": {"morphology": cv_auc(Xm, subj.hypertension.values.astype(int), tag="htn_morphology"), "papagei_pca16": cv_auc(pca, subj.hypertension.values.astype(int))}}
    # positive control: age from the embedding (ridge, 5-fold)
    pa = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=10.0)), Es, subj.age.values, cv=KFold(5, shuffle=True, random_state=SEED))
    res["age_from_papagei"] = {"r": float(np.corrcoef(pa, subj.age.values)[0, 1]), "mae_years": float(np.mean(np.abs(pa - subj.age.values)))}
    res["morphology_effects_diabetes"] = {f: {"median_dm": float(subj.loc[y == 1, f].median()), "median_ctl": float(subj.loc[y == 0, f].median())} for f in MORPH}
    for k, v in res["diabetes"].items():
        print(f"  diabetes {k:22s} AUC {v['auc']:.3f} {v['auc_ci']}")
    print(f"  hypertension morphology AUC {res['hypertension']['morphology']['auc']:.3f}; age from PaPaGei r = {res['age_from_papagei']['r']:.2f}, MAE {res['age_from_papagei']['mae_years']:.1f} y")
    np.savez(OUT / "ppg_pred.npz", age=subj.age.values, age_pred=pa, **{f"{k}_{n}": v[i] for k, v in PRED.items() for i, n in enumerate(("y", "p"))})
    (OUT / "ppg_open.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote ppg_open.json")


if __name__ == "__main__":
    main()
