"""
Build results/RESULTS_MASTER.json: the single source of every number that may appear in the paper.

Every entry records value, 95 % CI where one exists, the source file and the key path inside it.
Derived entries (recomputed from our own stored predictions with the same code the figures use)
record the source files and the derivation instead of a key path.

Reads only: results/out/*.json, results/out/*.npz (our own outputs), sim/out/*.json, sim/out/uq_thermal_samples.npz.
Writes only: results/RESULTS_MASTER.json.  Never edits a results file.

Run: python results/src/build_results_master.py
"""
import hashlib, json, pathlib, datetime
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT, SIM = ROOT / "results" / "out", ROOT / "sim" / "out"
SEED = 20260828  # same seed as pubstyle.SEED / train_fundus

SOURCES = {
    "fundus_deepdrid": OUT / "fundus_deepdrid.json", "fundus_aptos": OUT / "fundus_aptos.json",
    "fundus_dme_idrid": OUT / "fundus_dme_idrid.json", "tabular_ehr": OUT / "tabular_ehr.json",
    "hrv_open": OUT / "hrv_open.json", "ppg_open": OUT / "ppg_open.json",
    "fundus_oof": OUT / "fundus_oof.npz", "fundus_idrid_pred": OUT / "fundus_idrid_pred.npz",
    "fundus_aptos_pred": OUT / "fundus_aptos_pred.npz", "dme_idrid_pred": OUT / "dme_idrid_pred.npz",
    "ehr_oof": OUT / "ehr_oof.npz", "hrv_pred": OUT / "hrv_pred.npz", "ppg_pred": OUT / "ppg_pred.npz",
    "verification": SIM / "verification.json", "surrogate_thermal": SIM / "surrogate_thermal.json",
    "thermal3d_full": SIM / "thermal3d_full.json", "optical_results": SIM / "optical_results.json",
    "bioimpedance3d_coarse": SIM / "bioimpedance3d_coarse.json", "pupil_model": SIM / "pupil_model.json",
    "opto_thermal": SIM / "opto_thermal.json", "periorbital_summary": SIM / "periorbital_summary.json",
    "sensor_layout": SIM / "sensor_layout.json", "uq_thermal_samples": SIM / "uq_thermal_samples.npz",
    "thermal_result": SIM / "thermal_result.json",
}
J = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in SOURCES.items() if p.suffix == ".json"}
rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/")


def get(d, key):
    # keys containing "/" are split on "/" so that key names with dots (e.g. "42.3") can be addressed
    if key in d:  # whole key first (handles top-level names that contain a dot, e.g. "gain_0.92")
        return d[key]
    for part in (key.split("/") if "/" in key else key.split(".")):
        d = d[int(part)] if isinstance(d, list) else d[part]
    return d


def E(src, key, ci=None, **kw):
    """Entry read straight from a results file."""
    e = {"value": get(J[src], key), "source": rel(SOURCES[src]), "key": key}
    if ci:
        e["ci95"] = get(J[src], ci); e["ci_key"] = ci
    e.update(kw); return e


def D(value, sources, derivation, **kw):
    """Entry derived from our own stored outputs (no single key path)."""
    e = {"value": value, "source": [rel(SOURCES[s]) for s in sources], "key": None, "derived": derivation}
    e.update(kw); return e


# ------------------------------------------------------------------ fundus helpers (same code as train_fundus / figF10)
def apply_thresholds(pred, th): return np.digitize(pred, np.sort(th))


def fit_thresholds(pred, y):
    th0 = np.array([0.5, 1.5, 2.5, 3.5])
    f = lambda th: -cohen_kappa_score(y, apply_thresholds(pred, th), weights="quadratic")
    r = minimize(f, th0, method="Nelder-Mead", options={"xatol": 1e-3, "maxiter": 400}); return np.sort(r.x)


def sens_spec(y, s, th, groups, n_boot=1000):
    yh = apply_thresholds(s, th) >= 2; yr = y >= 2
    f = lambda a, b: (b[a].mean(), (~b[~a]).mean()); se, sp = f(yr, yh)
    rng = np.random.default_rng(SEED); g = np.asarray(groups); ug = np.unique(g); idx_by = {k: np.where(g == k)[0] for k in ug}; bs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[k] for k in rng.choice(ug, len(ug), replace=True)]); bs.append(f(yr[idx], yh[idx]))
    bs = np.array(bs)
    return float(se), float(sp), np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), np.percentile(bs[:, 1], [2.5, 97.5]).tolist()


# npz files are our own outputs written by train_fundus.py / eval_aptos.py (object arrays hold image paths); allow_pickle is safe here.
Z = np.load(SOURCES["fundus_oof"], allow_pickle=True); I = np.load(SOURCES["fundus_idrid_pred"], allow_pickle=True)
A = np.load(SOURCES["fundus_aptos_pred"], allow_pickle=True); ZD = np.load(SOURCES["dme_idrid_pred"])
ZE = np.load(SOURCES["ehr_oof"]); ZH = np.load(SOURCES["hrv_pred"]); ZP = np.load(SOURCES["ppg_pred"]); U = np.load(SOURCES["uq_thermal_samples"])
th = np.array(J["fundus_deepdrid"]["thresholds"]); m = ~np.isnan(Z["oof"])

M = {"_meta": {}, "datasets": {}, "fundus_dr": {}, "fundus_external_idrid": {}, "fundus_external_aptos": {}, "dme": {},
     "ehr": {}, "hrv": {}, "ppg": {}, "anatomy_and_layout": {}, "thermal": {}, "verification": {}, "uq": {}, "optical": {},
     "opto_thermal": {}, "bioimpedance": {}, "pupil": {}, "design_verdicts": {}, "literature_anchors": {}, "figure_index": {}}

# ------------------------------------------------------------------ datasets
M["datasets"] = {
    "deepdrid": {"name": E("fundus_deepdrid", "dataset"), "n_images": E("fundus_deepdrid", "cv.n"), "n_patients": E("fundus_deepdrid", "cv.n_patients"),
                 "grade_counts": E("fundus_deepdrid", "cv.grade_counts"), "licence": "CC BY-SA 4.0 (from dataset field)", "doi": "VERIFY LIVE before citing (not stored in results files)"},
    "idrid": {"name": E("fundus_dme_idrid", "dataset"), "n_images_dr": E("fundus_deepdrid", "external_idrid.n"), "grade_counts": E("fundus_deepdrid", "external_idrid.grade_counts"),
              "n_dme_train": E("fundus_dme_idrid", "cv_train.n"), "n_dme_test": E("fundus_dme_idrid", "test_fixed_split.n"), "dme_counts_train": E("fundus_dme_idrid", "cv_train.counts"),
              "dme_counts_test": E("fundus_dme_idrid", "test_fixed_split.counts"), "note": E("fundus_deepdrid", "external_idrid.note"), "doi": "VERIFY LIVE before citing"},
    "aptos2019": {"name": E("fundus_aptos", "dataset"), "citation": E("fundus_aptos", "citation"), "source_on_disk": E("fundus_aptos", "source_on_disk"),
                  "n_images": E("fundus_aptos", "external_aptos.n"), "grade_counts": E("fundus_aptos", "external_aptos.grade_counts"), "note": E("fundus_aptos", "external_aptos.note"),
                  "doi": "Kaggle competition dataset; no DOI. Cite competition page. VERIFY LIVE"},
    "istanbul_ehr": {"source": E("tabular_ehr", "source"), "n": E("tabular_ehr", "n"), "n_features_dr": D(len(J["tabular_ehr"]["features"]), ["tabular_ehr"], "len(features)"),
                     "n_features_dn": D(len(J["tabular_ehr"]["features_dn"]), ["tabular_ehr"], "len(features_dn)"),
                     "excluded_as_leakage": E("tabular_ehr", "excluded_as_leakage"), "excluded_as_dn_treatment_proxy": E("tabular_ehr", "excluded_as_dn_treatment_proxy"),
                     "prevalence_retinopathy": E("tabular_ehr", "targets.retinopathy.prevalence"), "prevalence_neuropathy": E("tabular_ehr", "targets.diabetic_neuropathy.prevalence"),
                     "doi": "10.17632/rr4rzzrjfc.2 (from source field; VERIFY LIVE)"},
    "physionet_hrv": {"ge75_n": E("hrv_open", "ge75_n"), "pooled_n": E("hrv_open", "pooled_n"), "features": E("hrv_open", "features"),
                      "note": "Two open PhysioNet Novak cohorts (GE-71, GE-75) pooled; GE-75 = cerebral-perfusion-diabetes 1.0.0 per PROGRESS.md. DOIs VERIFY LIVE"},
    "ppg_bp": {"source": E("ppg_open", "source"), "n_subjects": E("ppg_open", "n_subjects"), "n_diabetic": E("ppg_open", "n_diabetic"), "doi": "figshare 5459299 (from source field; VERIFY LIVE)"},
    "mbrset_brset": {"status": "PhysioNet credentialing pending; not used in any result; limitation only"},
}

# ------------------------------------------------------------------ fundus DR (DeepDRiD)
cm = np.array(J["fundus_deepdrid"]["cv"]["confusion"])
se, sp, sec, spc = sens_spec(Z["grade"][m], Z["oof"][m], th, Z["pid"][m])
M["fundus_dr"] = {
    "model": E("fundus_deepdrid", "model"), "img_px": E("fundus_deepdrid", "img"), "epochs": E("fundus_deepdrid", "epochs"), "seed": E("fundus_deepdrid", "seed"),
    "thresholds": E("fundus_deepdrid", "thresholds", note="fitted on DeepDRiD OOF; applied unchanged to IDRiD and APTOS"),
    "cv_scheme": "5-fold, patient-grouped, out-of-fold (OOF)",
    "qwk": E("fundus_deepdrid", "cv.qwk", ci="cv.qwk_ci"), "auc_referable": E("fundus_deepdrid", "cv.auc_referable", ci="cv.auc_referable_ci"),
    "accuracy": E("fundus_deepdrid", "cv.accuracy", ci="cv.accuracy_ci"), "confusion": E("fundus_deepdrid", "cv.confusion", note="rows true 0-4, cols predicted 0-4"),
    "recall_per_grade": D((np.diag(cm) / cm.sum(1)).round(4).tolist(), ["fundus_deepdrid"], "diag(confusion)/rowsum (figF09 panel a)"),
    "sens_referable_at_oof_operating_point": D(se, ["fundus_oof", "fundus_deepdrid"], "figF10.sens_spec, patient-grouped bootstrap 1000", ci95=sec),
    "spec_referable_at_oof_operating_point": D(sp, ["fundus_oof", "fundus_deepdrid"], "figF10.sens_spec, patient-grouped bootstrap 1000", ci95=spc),
    "fold_n_val": D([f["n_val"] for f in J["fundus_deepdrid"]["folds"]], ["fundus_deepdrid"], "folds[*].n_val"),
    "fold_final_epoch_val_qwk": D([f["history"][-1]["val_qwk"] for f in J["fundus_deepdrid"]["folds"]], ["fundus_deepdrid"], "folds[*].history[-1].val_qwk"),
    "fold_best_val_qwk": D([max(e["val_qwk"] for e in f["history"]) for f in J["fundus_deepdrid"]["folds"]], ["fundus_deepdrid"], "max over epochs of folds[*].history[].val_qwk"),
    "epoch_time_s_mean": D(float(np.mean([e["s"] for f in J["fundus_deepdrid"]["folds"] for e in f["history"]])), ["fundus_deepdrid"], "mean of folds[*].history[].s (512-px cache)"),
    "grade_counts": E("fundus_deepdrid", "cv.grade_counts"),
}

# ------------------------------------------------------------------ IDRiD external
se, sp, sec, spc = sens_spec(I["grade"], I["pred"], th, np.arange(len(I["grade"])))
th_i = fit_thresholds(I["pred"], I["grade"])
M["fundus_external_idrid"] = {
    "n": E("fundus_deepdrid", "external_idrid.n"), "qwk": E("fundus_deepdrid", "external_idrid.qwk", ci="external_idrid.qwk_ci"),
    "auc_referable": E("fundus_deepdrid", "external_idrid.auc_referable", ci="external_idrid.auc_referable_ci"),
    "accuracy": E("fundus_deepdrid", "external_idrid.accuracy", ci="external_idrid.accuracy_ci"), "confusion": E("fundus_deepdrid", "external_idrid.confusion"),
    "bootstrap_note": E("fundus_deepdrid", "external_idrid.note"),
    "sens_referable_at_oof_operating_point": D(se, ["fundus_idrid_pred", "fundus_deepdrid"], "figF10.sens_spec, image-level bootstrap 1000", ci95=sec),
    "spec_referable_at_oof_operating_point": D(sp, ["fundus_idrid_pred", "fundus_deepdrid"], "figF10.sens_spec, image-level bootstrap 1000", ci95=spc),
    "mean_predicted_minus_true_grade": D(float(np.mean(apply_thresholds(I["pred"], th) - I["grade"])), ["fundus_idrid_pred", "fundus_deepdrid"], "mean(pred_grade - true_grade)"),
    "share_over_graded": D(float(np.mean(apply_thresholds(I["pred"], th) > I["grade"])), ["fundus_idrid_pred", "fundus_deepdrid"], "mean(pred_grade > true_grade)"),
    "share_under_graded": D(float(np.mean(apply_thresholds(I["pred"], th) < I["grade"])), ["fundus_idrid_pred", "fundus_deepdrid"], "mean(pred_grade < true_grade)"),
    "refit_thresholds_not_used": D(th_i.tolist(), ["fundus_idrid_pred"], "train_fundus.fit_thresholds on IDRiD (figF10 panel c; reported, not used)"),
    "refit_qwk_not_used": D(float(cohen_kappa_score(I["grade"], apply_thresholds(I["pred"], th_i), weights="quadratic")), ["fundus_idrid_pred"], "QWK with IDRiD-refitted thresholds (not shown on any figure; reported, not used)"),
}

# ------------------------------------------------------------------ APTOS external
se, sp, sec, spc = sens_spec(A["grade"], A["pred"], th, np.arange(len(A["grade"])))
cma = np.array(J["fundus_aptos"]["external_aptos"]["confusion"])
M["fundus_external_aptos"] = {
    "n": E("fundus_aptos", "external_aptos.n"), "thresholds_from": E("fundus_aptos", "thresholds_from"), "n_fold_models": E("fundus_aptos", "n_fold_models"),
    "qwk": E("fundus_aptos", "external_aptos.qwk", ci="external_aptos.qwk_ci"), "auc_referable": E("fundus_aptos", "external_aptos.auc_referable", ci="external_aptos.auc_referable_ci"),
    "accuracy": E("fundus_aptos", "external_aptos.accuracy", ci="external_aptos.accuracy_ci"), "confusion": E("fundus_aptos", "external_aptos.confusion"),
    "bootstrap_note": E("fundus_aptos", "external_aptos.note"),
    "sens_referable_at_oof_operating_point": D(se, ["fundus_aptos_pred", "fundus_deepdrid"], "figF10.sens_spec, image-level bootstrap 1000", ci95=sec),
    "spec_referable_at_oof_operating_point": D(sp, ["fundus_aptos_pred", "fundus_deepdrid"], "figF10.sens_spec, image-level bootstrap 1000", ci95=spc),
    "shift_true2_predicted3": D({"count": int(cma[2, 3]), "of": int(cma[2].sum())}, ["fundus_aptos"], "confusion[2][3] / row 2 sum"),
    "shift_true4_predicted3": D({"count": int(cma[4, 3]), "of": int(cma[4].sum())}, ["fundus_aptos"], "confusion[4][3] / row 4 sum"),
    "shift_true1_predicted2": D({"count": int(cma[1, 2]), "of": int(cma[1].sum())}, ["fundus_aptos"], "confusion[1][2] / row 1 sum"),
    "shift_true0_predicted1": D({"count": int(cma[0, 1]), "of": int(cma[0].sum())}, ["fundus_aptos"], "confusion[0][1] / row 0 sum"),
    "mean_predicted_minus_true_grade": D(float(np.mean(apply_thresholds(A["pred"], th) - A["grade"])), ["fundus_aptos_pred", "fundus_deepdrid"], "mean(pred_grade - true_grade)"),
    "share_over_graded": D(float(np.mean(apply_thresholds(A["pred"], th) > A["grade"])), ["fundus_aptos_pred", "fundus_deepdrid"], "mean(pred_grade > true_grade)"),
    "share_under_graded": D(float(np.mean(apply_thresholds(A["pred"], th) < A["grade"])), ["fundus_aptos_pred", "fundus_deepdrid"], "mean(pred_grade < true_grade)"),
    "refit_thresholds_not_used": E("fundus_aptos", "external_aptos_refit_thresholds.thresholds"),
    "refit_qwk_not_used": E("fundus_aptos", "external_aptos_refit_thresholds.qwk", note="reported, not used"),
    "grade_counts": E("fundus_aptos", "external_aptos.grade_counts", note="matches the official competition training-set counts"),
}

# ------------------------------------------------------------------ DME
ct = pd.crosstab(I["grade"], I["dme"]).reindex(index=range(5), columns=range(3), fill_value=0)
any_dme = I["dme"] >= 1
paths = np.array([str(p) for p in I["path"]]); test = np.array(["Testing" in p for p in paths])
assert test.sum() == len(ZD["y_test"]) and np.array_equal(I["dme"][test], ZD["y_test"])
M["dme"] = {
    "model": E("fundus_dme_idrid", "model"), "img_px": E("fundus_dme_idrid", "img"), "epochs": E("fundus_dme_idrid", "epochs"), "thresholds": E("fundus_dme_idrid", "thresholds"),
    "cv_train_n": E("fundus_dme_idrid", "cv_train.n"), "cv_qwk": E("fundus_dme_idrid", "cv_train.qwk", ci="cv_train.qwk_ci"),
    "cv_auc_any_dme": E("fundus_dme_idrid", "cv_train.auc_any_dme", ci="cv_train.auc_any_dme_ci"), "cv_confusion": E("fundus_dme_idrid", "cv_train.confusion"),
    "test_n": E("fundus_dme_idrid", "test_fixed_split.n"), "test_qwk": E("fundus_dme_idrid", "test_fixed_split.qwk", ci="test_fixed_split.qwk_ci"),
    "test_auc_any_dme": E("fundus_dme_idrid", "test_fixed_split.auc_any_dme", ci="test_fixed_split.auc_any_dme_ci"), "test_confusion": E("fundus_dme_idrid", "test_fixed_split.confusion"),
    "dr_by_dme_crosstab_all516": D(ct.values.tolist(), ["fundus_idrid_pred"], "crosstab(DR grade 0-4 rows, DME grade 0-2 cols) on all 516 IDRiD images (figF11 panel c)"),
    "any_dme_share_in_referable": D(float(any_dme[I["grade"] >= 2].mean()), ["fundus_idrid_pred"], "mean(dme>=1 | grade>=2)", n=int((I["grade"] >= 2).sum())),
    "any_dme_share_in_nonreferable": D(float(any_dme[I["grade"] < 2].mean()), ["fundus_idrid_pred"], "mean(dme>=1 | grade<2)", n=int((I["grade"] < 2).sum())),
    "pearson_r_dr_score_vs_dme_score_test": D(float(np.corrcoef(I["pred"][test], ZD["test_pred"])[0, 1]), ["fundus_idrid_pred", "dme_idrid_pred"], "corrcoef on the 103 test images (figF11 panel d)", n=int(test.sum())),
}

# ------------------------------------------------------------------ EHR
T = "tabular_ehr"
def ehr_block(prefix):
    b = {}
    for mdl in ("logistic", "hgb"):
        b[mdl] = {"auc": E(T, f"{prefix}.{mdl}.auc", ci=f"{prefix}.{mdl}.auc_ci"), "auprc": E(T, f"{prefix}.{mdl}.auprc"),
                  "calibration_slope": E(T, f"{prefix}.{mdl}.calibration.slope"), "calibration_intercept": E(T, f"{prefix}.{mdl}.calibration.intercept"),
                  "brier": E(T, f"{prefix}.{mdl}.calibration.brier")}
    b["n"] = E(T, f"{prefix}.n"); b["prevalence"] = E(T, f"{prefix}.prevalence")
    b["logistic_odds_ratio_per_sd_top"] = E(T, f"{prefix}.logistic_odds_ratio_per_sd", note="ordered by |log OR|; top 15 stored")
    b["hgb_permutation_importance_top"] = E(T, f"{prefix}.hgb_permutation_importance", note="delta AUC on held-out 20 %; top 15 stored")
    return b
M["ehr"] = {
    "cv_scheme": "5-fold OOF for AUC/calibration; permutation importance on a held-out 20 % split; logistic ORs on all data (per figF12 docstring)",
    "retinopathy": ehr_block("targets.retinopathy"),
    "diabetic_neuropathy_main_excludes_treatment_proxies": ehr_block("targets.diabetic_neuropathy"),
    "dn_sensitivity_with_treatment_proxies": ehr_block("dn_sensitivity.with_treatment_proxies"),
    "dn_sensitivity_without_proxies_analgesics_psychoanaleptics": ehr_block("dn_sensitivity.without_proxies_analgesics_psychoanaleptics"),
    "joint_counts": E(T, "joint_counts", note="0 none, 1 DR only, 2 DN only, 3 both"),
    "joint_ovr_auc": E(T, "joint_ovr_auc", note="4-class boosted model, one-vs-rest, 5-fold OOF"),
    "dr_dn_odds_ratio": E(T, "dr_dn_association.odds_ratio"), "dr_dn_fisher_p": E(T, "dr_dn_association.p"), "dr_dn_table": E(T, "dr_dn_association.table"),
    "headline_dn_auc": {"value": get(J[T], "targets.diabetic_neuropathy.hgb.auc"), "ci95": get(J[T], "targets.diabetic_neuropathy.hgb.auc_ci"), "source": rel(SOURCES[T]),
                        "key": "targets.diabetic_neuropathy.hgb.auc", "note": "HEADLINE. Treatment proxies (antiepileptics N03, other_digestive = ATC A16, other_nervous_drugs) excluded. With proxies: dn_sensitivity.with_treatment_proxies.hgb.auc"},
}

# ------------------------------------------------------------------ HRV
H = "hrv_open"
M["hrv"] = {
    "features": E(H, "features"), "ge75_n": E(H, "ge75_n"), "pooled_n": E(H, "pooled_n"),
    "dm_vs_control": {"n": E(H, "dm_vs_control.n"), "n_pos": E(H, "dm_vs_control.n_pos"), "auc": E(H, "dm_vs_control.auc", ci="dm_vs_control.auc_ci", note="leave-one-out logistic")},
    "dr_within_dm": {"n": E(H, "dr_within_dm.n"), "n_dr": E(H, "dr_within_dm.n_dr"), "auc": E(H, "dr_within_dm.auc.auc", ci="dr_within_dm.auc.auc_ci", note="NULL finding, keep"),
                     "effects": E(H, "dr_within_dm.effects", note="Cohen d, Mann-Whitney p, BH-corrected p, medians per feature"),
                     "min_p_bh": D(min(v["p_bh"] for v in J[H]["dr_within_dm"]["effects"].values()), [H], "min over features of effects[*].p_bh"),
                     "max_abs_d": D(max(abs(v["d"]) for v in J[H]["dr_within_dm"]["effects"].values()), [H], "max |d| over features")},
    "autonomic_symptoms_ge75": {"n": E(H, "autonomic_symptoms_ge75.n"), "n_pos": E(H, "autonomic_symptoms_ge75.n_pos"), "auc": E(H, "autonomic_symptoms_ge75.auc", ci="autonomic_symptoms_ge75.auc_ci")},
    "hba1c_spearman": E(H, "hba1c_spearman", note="n = ge75_n diabetics with HbA1c"),
}

# ------------------------------------------------------------------ PPG
P = "ppg_open"
M["ppg"] = {
    "n_subjects": E(P, "n_subjects"), "n_diabetic": E(P, "n_diabetic"),
    "diabetes_auc_by_route": {k: E(P, f"diabetes.{k}.auc", ci=f"diabetes.{k}.auc_ci") for k in J[P]["diabetes"]},
    "diabetes_max_auc_any_route": D(max(v["auc"] for v in J[P]["diabetes"].values()), [P], "max over diabetes.*.auc (NULL finding, keep)"),
    "hypertension_auc_by_route": {k: E(P, f"hypertension.{k}.auc", ci=f"hypertension.{k}.auc_ci", note="positive control") for k in J[P]["hypertension"]},
    "hypertension_n_pos": E(P, "hypertension.morphology.n_pos"),
    "age_from_papagei_r": E(P, "age_from_papagei.r"), "age_from_papagei_mae_years": E(P, "age_from_papagei.mae_years"),
    "morphology_medians_diabetes": E(P, "morphology_effects_diabetes"),
    "cv_scheme": "5-fold CV, subject level; 3 x 2.1-s fingertip PPG per subject (figF13 docstring)",
}

# ------------------------------------------------------------------ anatomy, mesh, layout
S = "periorbital_summary"; L = "sensor_layout"
M["anatomy_and_layout"] = {
    "mesh_nodes": E(S, "nodes"), "mesh_tets": E(S, "tets"), "mesh_boundary_tris": E(S, "boundary_tris"),
    "n_tissue_regions": D(len(J[S]["regions"]), [S], "len(regions)"), "region_volumes_mm3": D({k: v["volume_mm3"] for k, v in J[S]["regions"].items()}, [S], "regions[*].volume_mm3"),
    "head_layers_mm": E(S, "head.layers_mm"), "units": E(S, "units"),
    "anthropometry": E(L, "anthropometry"), "frame": E(L, "frame"), "derived_geometry": E(L, "derived"),
    "n_sensing_components": D(len(J[L]["sensors"]), [L], "len(sensors) (figF02 title)"),
    "sensor_kinds": D(sorted({s["kind"] for s in J[L]["sensors"]}), [L], "set(sensors[*].kind)"),
}

# ------------------------------------------------------------------ thermal
T3 = "thermal3d_full"; sw = J[T3]["choroid_sweep"]; base = sw[0]
cap = {str(r["choroid_frac"]): {"d_apex_C": r["apex_C"] - base["apex_C"], "d_thermopile_90deg_C": r["baa_C"] - base["baa_C"], "d_thermopile_5deg_C": r["dci_C"] - base["dci_C"],
                                "d_canthus_C": r["canthus_C"] - base["canthus_C"]} for r in sw}
r50 = [r for r in sw if abs(r["choroid_frac"] - 0.5) < 1e-9][0]
M["thermal"] = {
    "model": "3-D Pennes bioheat, P1 tetrahedra, tissue_db.json properties, 25 C ambient (thermal3d_full.json)", "mesh": E(T3, "mesh"),
    "T_range_C": E(T3, "T_range_C"), "corneal_apex_C": E(T3, "corneal_apex_C"), "medial_canthus_skin_C": E(T3, "medial_canthus_skin_C"), "temple_skin_C": E(T3, "temple_skin_C"),
    "thermopile_90deg_reading_C": E(T3, "thermopile_BAA_90deg.reading_C"), "thermopile_90deg_spot_mm": E(T3, "thermopile_BAA_90deg.spot_mm"),
    "thermopile_5deg_reading_C": E(T3, "thermopile_DCI_5deg.reading_C"), "thermopile_5deg_spot_mm": E(T3, "thermopile_DCI_5deg.spot_mm"),
    "published_corneal_apex_range_C": E(T3, "validation.corneal_apex_published_C"), "published_inner_canthus_range_C": E(T3, "validation.inner_canthus_published_C"),
    "apex_below_published_band_C": D(J[T3]["validation"]["corneal_apex_published_C"][0] - J[T3]["corneal_apex_C"], [T3], "published lower bound - simulated apex"),
    "canthus_below_published_band_C": D(J[T3]["validation"]["inner_canthus_published_C"][0] - J[T3]["medial_canthus_skin_C"], [T3], "published lower bound - simulated canthus"),
    "choroid_sweep": E(T3, "choroid_sweep"), "choroid_sweep_deltas_C": D(cap, [T3], "reading(frac) - reading(1.0) per readout"),
    "apex_deficit_at_50pct_choroid_C": D(r50["apex_C"] - base["apex_C"], [T3], "choroid_sweep[0.5].apex_C - choroid_sweep[1.0].apex_C"),
    "thermopile_90deg_capture_fraction": D((r50["baa_C"] - base["baa_C"]) / (r50["apex_C"] - base["apex_C"]), [T3], "d_baa / d_apex at choroid 0.5 (constant 0.282-0.284 across the sweep)"),
    "thermopile_5deg_capture_fraction": D((r50["dci_C"] - base["dci_C"]) / (r50["apex_C"] - base["apex_C"]), [T3], "d_dci / d_apex at choroid 0.5 (0.139-0.141 across the sweep)"),
    "skin_perfusion_sweep": E(T3, "skin_perfusion_sweep"),
}

# ------------------------------------------------------------------ verification
V = "verification"
M["verification"] = {
    "mms_coarse": {"nodes": E(V, "mms_coarse.nodes"), "h_mm": E(V, "mms_coarse.h_mm"), "L2_rel": E(V, "mms_coarse.L2_rel"), "Linf_K": E(V, "mms_coarse.Linf_K")},
    "mms_full": {"nodes": E(V, "mms_full.nodes"), "h_mm": E(V, "mms_full.h_mm"), "L2_rel": E(V, "mms_full.L2_rel"), "Linf_K": E(V, "mms_full.Linf_K")},
    "observed_order": E(V, "observed_order", note="P1 tetrahedra, theoretical 2"),
    "gci": E(V, "gci", note="Roache GCI, fine/coarse/Richardson per quantity of interest, percent"),
    "gci_max_percent": D(max(v["GCI_fine_percent"] for v in J[V]["gci"].values()), [V], "max over gci[*].GCI_fine_percent (1.9985 -> say 'at most 2.0 %', not 'below 2 %')"),
}

# ------------------------------------------------------------------ UQ
UQ = "surrogate_thermal"; X, Y = U["X"], U["Y"]; params = [p[0] for p in J[UQ]["params"]]; outs = J[UQ]["outputs"]
ia, ich, ja = params.index("T_amb"), params.index("choroid_frac"), outs.index("apex_C")
lo_a, hi_a = J[UQ]["params"][ia][1], J[UQ]["params"][ia][2]
slope_unit, b0 = np.polyfit(X[:, ia], Y[:, ja], 1)
ST = np.array(J[UQ]["ST"]); S1 = np.array(J[UQ]["S1"])
M["uq"] = {
    "n_base": E(UQ, "n_base"), "n_solves": E(UQ, "n_solves"), "params_ranges": E(UQ, "params"), "outputs": E(UQ, "outputs"),
    "S1": E(UQ, "S1", note="rows = params, cols = outputs"), "ST": E(UQ, "ST", note="rows = params, cols = outputs"),
    "ST_ambient_range": D([float(ST[ia].min()), float(ST[ia].max())], [UQ], "min/max of ST[T_amb] over the 4 outputs"),
    "ST_choroid_ocular_readouts": D({o: float(ST[ich, j]) for j, o in enumerate(outs)}, [UQ], "ST[choroid_frac] per output"),
    "S1_choroid_per_output": D({o: float(S1[ich, j]) for j, o in enumerate(outs)}, [UQ], "S1[choroid_frac] per output"),
    "output_stats": E(UQ, "output_stats"), "surrogate_type": E(UQ, "surrogate.type"), "surrogate_rmse_holdout_C": E(UQ, "surrogate.rmse_holdout"),
    "sample_matrix_scaling": D("X in uq_thermal_samples.npz is in the unit hypercube [0,1]; physical value = lo + X*(hi-lo)", ["uq_thermal_samples", UQ], "checked X.min()/X.max()"),
    "apex_slope_per_unit_hypercube": D(float(slope_unit), ["uq_thermal_samples"], "polyfit(X[:,T_amb], Y[:,apex]) with X in [0,1]; THIS is what figF05 panel c currently prints as 'per C'"),
    "apex_slope_per_C_ambient": D(float(slope_unit / (hi_a - lo_a)), ["uq_thermal_samples", UQ], "slope_per_unit / (T_amb range 28-18)"),
    "apex_r2_ambient_alone": D(float(np.corrcoef(X[:, ia], Y[:, ja])[0, 1] ** 2), ["uq_thermal_samples"], "R^2 of apex on ambient alone"),
}

# ------------------------------------------------------------------ optical
O = "optical_results"; ss = J[O]["ppg"]["spacing_sweep_880"]
M["optical"] = {
    "nphoton": E(O, "nphoton"), "voxel_res_mm": E(O, "ppg.res_mm"), "pd_spacing_mm": E(O, "ppg.pd_spacing_mm"),
    "ppg_by_wavelength": {wl: {"detected_fraction": E(O, f"ppg.wavelengths.{wl}.detected_fraction"), "arterial_attenuation_share": E(O, f"ppg.wavelengths.{wl}.arterial_attenuation_share"),
                               "dermal_blood_attenuation_share": E(O, f"ppg.wavelengths.{wl}.dermal_blood_attenuation_share"), "ac_dc_perfusion_index": E(O, f"ppg.wavelengths.{wl}.ac_dc_perfusion_index"),
                               "ac_dc_artery_only": E(O, f"ppg.wavelengths.{wl}.ac_dc_artery_only"), "banana_weighted_depth_mm": E(O, f"ppg.wavelengths.{wl}.banana_weighted_depth_mm"),
                               "mean_partial_path_mm": E(O, f"ppg.wavelengths.{wl}.mean_partial_path_mm"), "attenuation_share": E(O, f"ppg.wavelengths.{wl}.attenuation_share")} for wl in ("660", "880")},
    "spacing_sweep_880": E(O, "ppg.spacing_sweep_880"),
    "detected_fraction_ratio_3mm_over_12mm": D(ss[0]["detected_fraction"] / ss[3]["detected_fraction"], [O], "spacing_sweep_880[3 mm].detected_fraction / [12 mm] (37.5x; CAPTIONS.md says 40x)"),
    "artery_share_of_pulse_at_12mm": D(ss[3]["artery_share_of_pulse"], [O], "spacing_sweep_880[12 mm].artery_share_of_pulse"),
    "led": {"led_to_cornea_mm": E(O, "led.led_to_cornea_mm"), "cone_half_angle_deg": E(O, "led.cone_half_angle_deg"), "irradiance_W_m2_per_W_led": E(O, "led.irradiance_W_m2_per_W_led"),
            "iec62471_cornea_limit_W_m2": E(O, "led.iec62471_cornea_limit_W_m2"), "max_led_optical_power_mW": E(O, "led.max_led_optical_power_mW"),
            "absorbed_fraction_per_tissue": E(O, "led.absorbed_fraction_per_tissue"), "absorbed_total_fraction": E(O, "led.absorbed_total_fraction")},
}

# ------------------------------------------------------------------ opto-thermal
OT = "opto_thermal"
M["opto_thermal"] = {"steady_rise_C_by_led_power_mW": E(OT, "led_power_mW", note="keys 5.0, 20.0, 42.3 mW; sites apex, lid, canthus, max"),
                     "apex_rise_at_iec_limit_C": E(OT, "led_power_mW/42.3/apex"), "lid_rise_at_iec_limit_C": E(OT, "led_power_mW/42.3/lid"), "canthus_rise_at_iec_limit_C": E(OT, "led_power_mW/42.3/canthus"),
                     "absorbed_field_max_W_m3_per_W": E(OT, "absorbed_field_max_W_m3_per_W"),
                     "apex_rise_as_fraction_of_50pct_choroid_deficit": D(J[OT]["led_power_mW"]["42.3"]["apex"] / abs(r50["apex_C"] - base["apex_C"]), [OT, T3], "0.214 / 0.543")}

# ------------------------------------------------------------------ bioimpedance
B = "bioimpedance3d_coarse"; swp = J[B]["sweep"]
M["bioimpedance"] = {
    "mesh": E(B, "mesh", note="coarse mesh"), "I_drive_A": E(B, "I_drive_A"), "pad_radius_mm": E(B, "pad_radius_mm"), "z_contact_ohm_m2": E(B, "z_contact_ohm_m2"),
    "configuration": E(B, "configuration", note="local tetrapolar around ONE eye (drive brow_lat->nose_inf, sense brow_med-nose_sup) and bipolar brow_med-nose_sup. The cross-eye tetrapolar of design verdict 6 is NOT in this file."),
    "electrodes": E(B, "electrodes"),
    "sweep_by_frequency_Hz": {f: {"tetrapolar_abs_ohm": E(B, f"sweep.{f}.tetrapolar.abs"), "tetrapolar_phase_deg": E(B, f"sweep.{f}.tetrapolar.phase_deg"),
                                  "bipolar_tissue_abs_ohm": E(B, f"sweep.{f}.bipolar.Z_tissue.abs"), "bipolar_tissue_phase_deg": E(B, f"sweep.{f}.bipolar.Z_tissue.phase_deg"),
                                  "bipolar_contact_abs_ohm": E(B, f"sweep.{f}.bipolar.Z_contact.abs"), "bipolar_total_abs_ohm": E(B, f"sweep.{f}.bipolar.Z_total.abs"),
                                  "reciprocity_residual": E(B, f"sweep.{f}.reciprocity_residual"), "geselowitz_residual": E(B, f"sweep.{f}.geselowitz_residual")} for f in swp},
    "max_reciprocity_residual": D(max(v["reciprocity_residual"] for v in swp.values()), [B], "max over frequencies"),
    "max_geselowitz_residual": D(max(v["geselowitz_residual"] for v in swp.values()), [B], "max over frequencies"),
    "bipolar_contact_share_of_total_50kHz": D(swp["50000"]["bipolar"]["Z_contact"]["abs"] / swp["50000"]["bipolar"]["Z_total"]["abs"], [B], "|Z_contact| / |Z_total| at 50 kHz"),
    "tissue_share_50kHz": E(B, "tissue_share_50kHz"), "orbit_share_50kHz": E(B, "orbit_share_50kHz", note="orbit + eye + lid share of the tetrapolar transfer impedance"),
    "edema_50kHz": E(B, "edema_50kHz", note="keys 0.1/0.2/0.3 = +10/20/30 % orbital water; tetrapolar and bipolar dZ percent"),
    "thermal_coupling_dZ_percent_50kHz": E(B, "thermal_coupling_50kHz.dZ_abs_percent", note="sigma(T) over the thermal field vs uniform"),
}

# ------------------------------------------------------------------ pupil
PU = "pupil_model"
M["pupil"] = {"model": E(PU, "model"), "healthy": E(PU, "healthy"), "gain_0_92": E(PU, "gain_0.92", note="autonomic gain 0.92 (published diabetic group, see literature_anchors)"),
              "gain_sweep": E(PU, "gain_sweep"), "stimulus_sweep": E(PU, "stimulus_sweep", note="0 lx amplitude 0.095 mm: a residual, not zero. Say 'no measurable reflex (0.1 mm)'"),
              "camera": E(PU, "camera")}

# ------------------------------------------------------------------ eight design verdicts -> master paths
M["design_verdicts"] = {
    "1_thermopile_90deg_fov_too_wide": ["thermal.thermopile_90deg_spot_mm", "thermal.thermopile_90deg_capture_fraction", "thermal.thermopile_5deg_capture_fraction"],
    "2_940nm_illuminator_evokes_no_reflex": ["pupil.stimulus_sweep"],
    "3_aim_thermopile_at_ocular_surface_not_canthus": ["uq.ST_choroid_ocular_readouts", "uq.S1_choroid_per_output"],
    "4_electrode_contact_model": ["bioimpedance.electrodes", "bioimpedance.bipolar_contact_share_of_total_50kHz"],
    "5_3mm_ppg_reads_dermal_plexus_not_artery": ["optical.ppg_by_wavelength", "optical.spacing_sweep_880", "optical.detected_fraction_ratio_3mm_over_12mm"],
    "6_cross_eye_tetrapolar_blind_to_orbit": {"paths": ["bioimpedance.orbit_share_50kHz", "bioimpedance.edema_50kHz"],
                                             "WARNING": "Numbers for the cross-eye configuration (PROGRESS.md session 4: |Z| 35.4 ohm, orbit net -1.5 %, -0.06 % per +10 % edema) survive only in PROGRESS.md narrative; no results file holds them. The current JSON holds the redesigned LOCAL tetrapolar only. Either re-run the cross-eye case and save it, or state verdict 6 qualitatively."},
    "7_illuminator_heating_confound": ["opto_thermal.apex_rise_at_iec_limit_C", "opto_thermal.apex_rise_as_fraction_of_50pct_choroid_deficit"],
    "8_ambient_temperature_dominates_compensate": ["uq.ST_ambient_range", "uq.apex_slope_per_C_ambient"],
}

# ------------------------------------------------------------------ literature anchors used on figures (not our results)
M["literature_anchors"] = {
    "published_corneal_apex_C": E(T3, "validation.corneal_apex_published_C", note="citation to be added in the paper; VERIFY"),
    "published_inner_canthus_C": E(T3, "validation.inner_canthus_published_C", note="citation to be added; VERIFY"),
    "iec62471_cornea_limit_W_m2": E(O, "led.iec62471_cornea_limit_W_m2"),
    "asme_vv20_gci_5pct_line": {"value": 5.0, "source": "results/src/figF04_verification.py", "key": "typed constant on panel b", "note": "typed in the figure script, not a results number"},
    "clinical_npdr_corneal_deficit_C": E("thermal_result", "clinical_npdr_deficit_C", note="from the earlier axisymmetric model file; Chandrasekar 2021 PMID 34464609; VERIFY citation"),
    "pupil_calibration_source": {"value": "PMID 8462391, healthy amplitude 2.44 mm and MCV 7.2 mm/s; diabetic group corresponds to gain 0.92", "source": "PROGRESS.md session 4 narrative", "key": None, "note": "VERIFY before citing"},
}

# ------------------------------------------------------------------ figure index
M["figure_index"] = {
    "F1": ["anatomy_and_layout", "verification.observed_order", "verification.gci_max_percent", "uq.n_solves"],
    "F2": ["anatomy_and_layout"], "F3": ["thermal"], "F4": ["verification", "opto_thermal", "optical.led"], "F5": ["uq"], "F6": ["optical"],
    "F7": ["bioimpedance"], "F8": ["pupil"], "F9": ["fundus_dr", "fundus_external_idrid", "fundus_external_aptos"],
    "F10": ["fundus_dr", "fundus_external_idrid", "fundus_external_aptos"], "F11": ["dme"], "F12": ["ehr"], "F13": ["hrv", "ppg"],
}

# ------------------------------------------------------------------ meta
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
M["_meta"] = {
    "built": datetime.datetime.now().isoformat(timespec="seconds"), "builder": "results/src/build_results_master.py",
    "rule": "This file is the ONLY source of numbers for the manuscript. Entries carry value, ci95 where one exists, source file and key path. Derived entries carry source files and the derivation.",
    "sources": {k: {"path": rel(p), "sha256_16": sha(p), "mtime": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes"), "bytes": p.stat().st_size} for k, p in SOURCES.items()},
    "no_quantum": "results/src/concept_quantum.py contributed to no entry in this file.",
    "figure_cross_check_2026-09-08": [
        "F5 panel c: x-axis is labelled 'ambient temperature (C)' but plots the unit-hypercube sample (0-1); the printed slope 3.17 'C per C ambient' is per unit of the [0,1] variable. Physical slope = 0.317 C per C (uq.apex_slope_per_C_ambient). CAPTIONS.md repeats 3.2. Fix figF05_uq.py in Phase 5 by unscaling X with params ranges; the colourbar (0-1) has the same issue (choroid range 0.3-1.0).",
        "F6 / CAPTIONS.md: 'detected photons fall 40x' between 3 mm and 12 mm spacing; the ratio is 37.5x (optical.detected_fraction_ratio_3mm_over_12mm). Say 'about 37-fold'. Figure panel itself plots the raw values (no disagreement on canvas).",
        "F4 / F1 / CAPTIONS.md: 'GCI < 2 %' while the largest GCI is 1.9985 % (verification.gci_max_percent). Say 'at most 2.0 %'.",
        "F8 / CAPTIONS.md: '940-nm illuminator evokes no reflex'; the model gives a 0.095 mm residual at 0 lx (pupil.stimulus_sweep). Say 'no measurable reflex (0.1 mm)'.",
        "Design verdict 6 (cross-eye tetrapolar blind to the orbit): its numbers exist only in PROGRESS.md session-4 narrative; bioimpedance3d_coarse.json holds the redesigned local tetrapolar only. Not citable as numbers until re-run and saved.",
        "PROGRESS.md session-4 narrative numbers (MMS order 1.86, GCI 0.05/0.17/0.14/1.6 %, Sobol S1 0.12/0.15/0.01, ST 0.71-1.00, tetrapolar 35.4 ohm, sigma(T) +0.93 %) are superseded by the 2026-09-04 JSONs used here (order 1.79; GCI 0.035/0.036/0.008/2.0 %; S1 0.095/0.129/0.007; ST 0.74-1.01; local tetrapolar 427 ohm at 50 kHz; sigma(T) +4.9 %). Use only this file.",
        "All on-canvas numbers read from JSON (F1-F4, F6-F13) agree with this file; all data files predate the figure build (2026-09-07 19:56).",
    ],
    "cross_checks_passed": "Confusion matrices, QWK, AUC and accuracy for DeepDRiD OOF, IDRiD and APTOS; DME OOF/test AUC and QWK; all EHR, HRV and PPG AUCs; PaPaGei age r/MAE were recomputed from the stored predictions and match the JSON values (build run).",
}

out = ROOT / "results" / "RESULTS_MASTER.json"
out.write_text(json.dumps(M, indent=1, default=float), encoding="utf-8")
print("wrote", rel(out), out.stat().st_size, "bytes")
