"""
A4 — clinical-risk-factor baseline for diabetic retinopathy.

This is the number every later model is measured against. If the multimodal
wearable model cannot beat routinely-available clinical variables, there is no
result worth publishing.

Cohort   NHANES 2005-2006 (_D) + 2007-2008 (_E), the only cycles with retinal
         imaging graded to ETDRS by the University of Wisconsin Ocular
         Epidemiology Reading Center. Adults with self-reported diagnosed
         diabetes and a gradable fundus photograph.
Outcome  Any diabetic retinopathy — ETDRS level of the worse eye > 10.
         (10 = no retinopathy; 12-20 mild; 31-51 moderate/severe NPDR; 60+ PDR)
Model    Logistic regression on age, sex, diabetes duration, insulin use, HbA1c.

Note on survey weights: NHANES is a complex multistage sample. We fit an
UNWEIGHTED model because the goal is individual-level discrimination, not
population prevalence estimation. This is stated as a limitation rather than
silently ignored.

Run:  python results/src/baseline_clinical.py
"""
import pathlib, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "raw" / "nhanes"
OUT = HERE.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260828

# NHANES missing/sentinel codes
REFUSED, DONTKNOW, LESS_THAN_ONE_YEAR = 777, 999, 666


def load_cycle(sfx):
    def rd(name):
        return pd.read_sas(RAW / f"{name}_{sfx}.XPT", format="xport")

    demo = rd("DEMO")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1"]]
    diq = rd("DIQ")[["SEQN", "DIQ010", "DID040", "DIQ050"]]
    ghb = rd("GHB")[["SEQN", "LBXGH"]]
    bmx = rd("BMX")[["SEQN", "BMXBMI"]]
    op = rd("OPXRET")
    eye = [c for c in ("OPDDRET", "OPDSRET") if c in op.columns]
    op = op[["SEQN"] + eye].copy()
    # worse eye = higher ETDRS level
    op["etdrs"] = op[eye].max(axis=1)

    df = (demo.merge(diq, on="SEQN", how="left")
               .merge(ghb, on="SEQN", how="left")
               .merge(bmx, on="SEQN", how="left")
               .merge(op[["SEQN", "etdrs"]], on="SEQN", how="inner"))
    df["cycle"] = sfx
    return df


def build_cohort():
    df = pd.concat([load_cycle("D"), load_cycle("E")], ignore_index=True)
    n_all = len(df)

    # diagnosed diabetes only
    df = df[df.DIQ010 == 1]
    n_dm = len(df)

    # gradable photograph
    df = df[df.etdrs.notna()]
    n_grad = len(df)

    # --- diabetes duration -------------------------------------------------
    age_dx = df.DID040.copy()
    age_dx = age_dx.replace({REFUSED: np.nan, DONTKNOW: np.nan})
    age_dx = age_dx.where(age_dx != LESS_THAN_ONE_YEAR,
                          df.RIDAGEYR - 0.5)          # "less than 1 year"
    df = df.assign(duration=(df.RIDAGEYR - age_dx).clip(lower=0))

    df = df.assign(
        age=df.RIDAGEYR,
        female=(df.RIAGENDR == 2).astype(int),
        insulin=(df.DIQ050 == 1).astype(int),
        hba1c=df.LBXGH,
        bmi=df.BMXBMI,
        any_dr=(df.etdrs > 10).astype(int),
    )
    return df, dict(all_graded=n_all, diabetic=n_dm, gradable=n_grad)


FEATURES = ["age", "female", "duration", "insulin", "hba1c"]


def bootstrap_auc(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    out = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        out.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


if __name__ == "__main__":
    df, counts = build_cohort()
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["any_dr"].to_numpy()

    print("=== COHORT ===")
    print(f"  NHANES 2005-2008 records with fundus grading : {counts['all_graded']:,}")
    print(f"  with diagnosed diabetes                      : {counts['diabetic']:,}")
    print(f"  with gradable photograph (analysis cohort)   : {counts['gradable']:,}")
    print(f"  any retinopathy (ETDRS > 10)                 : {y.sum():,} "
          f"({100*y.mean():.1f}%)")
    print(f"  age            mean {df.age.mean():.1f}  sd {df.age.std():.1f}")
    print(f"  duration (yrs) mean {df.duration.mean():.1f}  sd {df.duration.std():.1f}"
          f"   missing {df.duration.isna().sum()}")
    print(f"  HbA1c (%)      mean {df.hba1c.mean():.2f} sd {df.hba1c.std():.2f}"
          f"   missing {df.hba1c.isna().sum()}")
    print(f"  on insulin     {df.insulin.sum():,} ({100*df.insulin.mean():.1f}%)")

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])

    # out-of-fold predictions require a true partition, so a plain
    # StratifiedKFold here; repeated CV is run separately below for variance.
    cv1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p = cross_val_predict(pipe, X, y, cv=cv1, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, p)
    lo, hi = bootstrap_auc(y, p)
    brier = brier_score_loss(y, p)

    cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_auc = cross_val_score(pipe, X, y, cv=cv2, scoring="roc_auc")

    print("\n=== BASELINE PERFORMANCE ===")
    print(f"  AUC (pooled out-of-fold)  {auc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  AUC (50 folds)            {fold_auc.mean():.4f} +/- {fold_auc.std():.4f}")
    print(f"  Brier                     {brier:.4f}")
    print(f"  prevalence                {y.mean():.4f}")

    # coefficients on the full fit, for interpretability
    pipe.fit(X, y)
    coef = pipe.named_steps["lr"].coef_[0]
    print("\n=== STANDARDISED COEFFICIENTS (odds ratio per 1 SD) ===")
    for f, c in sorted(zip(FEATURES, coef), key=lambda t: -abs(t[1])):
        print(f"  {f:10s} {c:+.4f}   OR {np.exp(c):.3f}")

    fpr, tpr, _ = roc_curve(y, p)
    np.savez(OUT / "baseline_roc.npz", fpr=fpr, tpr=tpr, y=y, p=p)
    (OUT / "baseline_clinical.json").write_text(json.dumps({
        "cohort": counts,
        "n_analysis": int(len(y)),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "features": FEATURES,
        "auc": float(auc), "auc_ci95": [lo, hi],
        "auc_fold_mean": float(fold_auc.mean()),
        "auc_fold_sd": float(fold_auc.std()),
        "brier": float(brier),
        "coefficients": {f: float(c) for f, c in zip(FEATURES, coef)},
        "seed": SEED,
        "note": "unweighted logistic regression; NHANES survey weights not applied "
                "because the target is individual discrimination, not population "
                "prevalence",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'baseline_clinical.json'} and baseline_roc.npz")
    print("\n>>> THIS AUC IS THE NUMBER THE MULTIMODAL MODEL MUST BEAT <<<")
