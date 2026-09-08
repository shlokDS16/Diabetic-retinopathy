"""
Systemic (tabular) branch on the open Istanbul EHR export (Mendeley doi:10.17632/rr4rzzrjfc.2,
CC BY 4.0): one row per diabetic patient with HbA1c, 1-year HbA1c change, age, sex, lipids,
creatinine, ICD-chapter comorbidity flags and antidiabetic drug classes.

Targets (as recorded in the export): retinopathy (0/1), diabetic_nueropathy (0/1), and the
JOINT outcome (none / DR only / DN only / both) -- the systemic counterpart of the paper's
joint DR+DN thesis. Features exclude the outcome-adjacent flags (eye/nervous-system chapters,
eye drugs, refraction) so the model cannot read the answer off a neighbouring code.

Models: L2 logistic regression (interpretable, odds ratios) and HistGradientBoosting; 5-fold
stratified CV with patient-level bootstrap CIs; calibration slope/intercept; SHAP-free feature
importance via permutation on the held-out folds. Aggregate outputs only ->
results/out/tabular_ehr.json.

Run (system python):  python results/src/tabular_ehr.py
"""
import json, pathlib, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "open" / "mendeley_glycemic_ehr" / "data_export_icin.csv"
OUT = ROOT / "out"
SEED = 20260828
LEAK = {"retinopathy", "diabetic_nueropathy", "neuropathies", "nervous_sys_dis", "cataract", "eye_other",
        "refraction_dis", "eye_ear_drugs", "Glycemic_control", "id"}
# Treatment proxies for the NEUROPATHY outcome (consequences of the diagnosis, like eye drugs for DR):
#   antiepileptics  -> gabapentin / pregabalin (ATC N03) are first-line neuropathic-pain drugs
#   other_digestive -> ATC A16 "other alimentary tract and metabolism products" = alpha-lipoic (thioctic) acid,
#                      prescribed in Turkey specifically for diabetic neuropathy
#   other_nervous_drugs -> residual N-chapter agents
DN_PROXY = {"antiepileptics", "other_digestive", "other_nervous_drugs"}
DN_PROXY_WIDE = DN_PROXY | {"analgesics", "pshycoanaleptics"}      # sensitivity: also drop pain / antidepressant classes


def load():
    df = pd.read_csv(SRC, sep=";", decimal=".", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def auc_ci(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def calib(y, p):
    from sklearn.linear_model import LogisticRegression as LR
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LR(C=1e6, max_iter=1000).fit(lp[:, None], y)
    return {"slope": float(m.coef_[0][0]), "intercept": float(m.intercept_[0]), "brier": float(brier_score_loss(y, p))}


PRED = {}


def evaluate(X, y, name, feats):
    y = y.astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {"n": int(len(y)), "prevalence": float(y.mean())}
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED)
    for tag, model in (("logistic", lr), ("hgb", gb)):
        t = time.time()
        p = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        PRED[f"{name}_{tag}"] = p
        res[tag] = {"auc": float(roc_auc_score(y, p)), "auc_ci": auc_ci(y, p), "auprc": float(average_precision_score(y, p)),
                    "calibration": calib(y, p), "time_s": time.time() - t}
        print(f"  {name:22s} {tag:8s} AUC {res[tag]['auc']:.3f} {res[tag]['auc_ci']} AUPRC {res[tag]['auprc']:.3f} "
              f"slope {res[tag]['calibration']['slope']:.2f} ({res[tag]['time_s']:.0f}s)", flush=True)
    # odds ratios from the logistic model on all data (interpretation only)
    lr.fit(X, y)
    coef = lr.named_steps["logisticregression"].coef_[0]
    sd = X.std(axis=0)
    res["logistic_odds_ratio_per_sd"] = {f: float(np.exp(c)) for f, c in sorted(zip(feats, coef), key=lambda t: -abs(t[1]))[:15]}
    # permutation importance of the boosted model on a held-out 20 %
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    gb.fit(Xtr, ytr)
    pi = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=4)
    res["hgb_permutation_importance"] = {feats[i]: float(pi.importances_mean[i]) for i in np.argsort(-pi.importances_mean)[:15]}
    return res


def main():
    df = load()
    y_dr = df["retinopathy"].fillna(0).astype(int).values
    y_dn = df["diabetic_nueropathy"].fillna(0).astype(int).values
    feats = [c for c in df.columns if c not in LEAK]
    feats_dn = [c for c in feats if c not in DN_PROXY]
    feats_dn_wide = [c for c in feats if c not in DN_PROXY_WIDE]
    mat = lambda fl: df[fl].fillna(df[fl].median()).values.astype(float)
    X, X_dn, X_dn_wide = mat(feats), mat(feats_dn), mat(feats_dn_wide)
    print(f"EHR: {len(df):,} patients, {len(feats)} features (DN model: {len(feats_dn)}); DR {y_dr.mean()*100:.1f} %, DN {y_dn.mean()*100:.1f} %, both {np.mean((y_dr==1)&(y_dn==1))*100:.1f} %")
    res = {"source": "Mendeley doi:10.17632/rr4rzzrjfc.2 (CC BY 4.0), Istanbul e-Nabiz EHR export", "n": int(len(df)),
           "features": feats, "features_dn": feats_dn, "excluded_as_leakage": sorted(LEAK & set(df.columns)),
           "excluded_as_dn_treatment_proxy": sorted(DN_PROXY & set(df.columns)),
           "targets": {"retinopathy": evaluate(X, y_dr, "retinopathy", feats),
                       "diabetic_neuropathy": evaluate(X_dn, y_dn, "diabetic_neuropathy", feats_dn)},
           "dn_sensitivity": {"with_treatment_proxies": evaluate(X, y_dn, "DN_with_proxies", feats),
                              "without_proxies_analgesics_psychoanaleptics": evaluate(X_dn_wide, y_dn, "DN_wide_exclusion", feats_dn_wide)}}
    X = X_dn                                   # joint 4-class model uses the proxy-free feature set
    # joint outcome: 4-class; report one-vs-rest AUCs from the boosted model
    joint = y_dr + 2 * y_dn
    res["joint_counts"] = {str(int(k)): int(v) for k, v in zip(*np.unique(joint, return_counts=True))}
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
    P = cross_val_predict(gb, X, joint, cv=skf, method="predict_proba")
    res["joint_ovr_auc"] = {["none", "DR only", "DN only", "both"][k]: float(roc_auc_score(joint == k, P[:, k])) for k in range(4)}
    # DR vs DN association (the co-occurrence the paper argues for)
    ct = pd.crosstab(y_dr, y_dn)
    from scipy.stats import fisher_exact
    orr, p = fisher_exact(ct.values)
    res["dr_dn_association"] = {"odds_ratio": float(orr), "p": float(p), "table": ct.values.tolist()}
    print("joint OvR AUC:", res["joint_ovr_auc"], "| DR-DN odds ratio", round(orr, 2))
    np.savez(OUT / "ehr_oof.npz", y_dr=y_dr, y_dn=y_dn, joint=joint, P_joint=P, **PRED)
    (OUT / "tabular_ehr.json").write_text(json.dumps(res, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)), encoding="utf-8")
    print("wrote tabular_ehr.json")


if __name__ == "__main__":
    main()
