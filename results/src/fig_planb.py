"""
Figure R2 -- Plan-B results panel (open data), journal style via figstyle.py.

Panels
  a  Fundus DR grade (DeepDRiD, patient-level 5-fold): confusion matrix of the OOF
     predictions + QWK with CI; external IDRiD QWK if present.
  b  Referable-DR ROC: DeepDRiD OOF and IDRiD external.
  c  Systemic EHR (n = 77,724): ROC for retinopathy and diabetic neuropathy (HGB, 5-fold OOF
     are not stored, so the panel shows AUC bars with CIs + calibration slope text).
  d  HRV -> retinopathy within diabetics (open PhysioNet cohorts): effect sizes (Cohen d)
     with BH-corrected p, and the LOO AUC.
  e  PPG-BP: diabetes AUCs for each feature route vs demographics (the honest null) and
     the hypertension positive control.

Run (system python):  python results/src/fig_planb.py
"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
try:
    import figstyle as FS
    FS.apply() if hasattr(FS, "apply") else None
    COL = getattr(FS, "MODEL_COLORS", {})
except Exception:
    COL = {}
ACC = "#b04a20"; C_FUND = COL.get("fundus", "#2a6f97"); C_EHR = COL.get("static", "#6a4c93"); C_HRV = COL.get("hrv", "#1f8a82"); C_PPG = COL.get("ppg", "#b8336a")


def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    fundus, ehr, hrv, ppg = load("fundus_deepdrid.json"), load("tabular_ehr.json"), load("hrv_open.json"), load("ppg_open.json")
    fig, axs = plt.subplots(2, 3, figsize=(182 / 25.4, 110 / 25.4))
    axs = axs.ravel()

    # a: confusion matrix
    ax = axs[0]
    if fundus:
        cm = np.array(fundus["cv"]["confusion"], float); cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xlabel("predicted ICDR grade"); ax.set_ylabel("true grade")
        t = f"DeepDRiD OOF: QWK {fundus['cv']['qwk']:.2f} [{fundus['cv']['qwk_ci'][0]:.2f}, {fundus['cv']['qwk_ci'][1]:.2f}]"
        if "external_idrid" in fundus:
            t += f"\nIDRiD external: QWK {fundus['external_idrid']['qwk']:.2f}"
        ax.set_title(t, fontsize=7)
    ax.text(-0.25, 1.08, "a", transform=ax.transAxes, fontweight="bold")

    # b: referable ROC
    ax = axs[1]
    if fundus and (OUT / "fundus_oof.npz").exists():
        # allow_pickle: our own npz (object array of image paths written by train_fundus.py), not external input
        z = np.load(OUT / "fundus_oof.npz", allow_pickle=True); m = ~np.isnan(z["oof"])
        fpr, tpr, _ = roc_curve(z["grade"][m] >= 2, z["oof"][m]); ax.plot(fpr, tpr, color=C_FUND, label=f"DeepDRiD OOF, AUC {fundus['cv']['auc_referable']:.3f}")
    if fundus and "external_idrid" in fundus and (OUT / "fundus_idrid_pred.npz").exists():
        z = np.load(OUT / "fundus_idrid_pred.npz", allow_pickle=True)
        fpr, tpr, _ = roc_curve(z["grade"] >= 2, z["pred"]); ax.plot(fpr, tpr, color=ACC, label=f"IDRiD external, AUC {fundus['external_idrid']['auc_referable']:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.6); ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity"); ax.set_title("Referable DR (grade ≥ 2)", fontsize=7); ax.legend(fontsize=5.5, loc="lower right")
    ax.text(-0.25, 1.08, "b", transform=ax.transAxes, fontweight="bold")

    # c: EHR AUCs
    ax = axs[2]
    if ehr:
        names, vals, los, his = [], [], [], []
        for tgt, lab in (("retinopathy", "DR"), ("diabetic_neuropathy", "DN")):
            for mdl, ml in (("logistic", "logit"), ("hgb", "HGB")):
                r = ehr["targets"][tgt][mdl]; names.append(f"{lab}\n{ml}"); vals.append(r["auc"]); los.append(r["auc"] - r["auc_ci"][0]); his.append(r["auc_ci"][1] - r["auc"])
        x = np.arange(len(names)); ax.bar(x, vals, color=[C_EHR, C_EHR, ACC, ACC], yerr=[los, his], capsize=2, width=0.6)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6); ax.set_ylim(0.5, 0.9); ax.set_ylabel("AUC (5-fold)")
        ax.set_title(f"Systemic EHR, n = {ehr['n']:,}; DR–DN OR {ehr['dr_dn_association']['odds_ratio']:.2f}", fontsize=7)
    ax.text(-0.25, 1.08, "c", transform=ax.transAxes, fontweight="bold")

    # d: HRV effects
    ax = axs[3]
    if hrv and hrv.get("dr_within_dm", {}).get("effects"):
        eff = hrv["dr_within_dm"]["effects"]; names = list(eff); d = [eff[k]["d"] for k in names]; pbh = [eff[k].get("p_bh", 1) for k in names]
        y = np.arange(len(names)); ax.barh(y, d, color=[ACC if p < 0.05 else C_HRV for p in pbh]); ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([n.upper() for n in names], fontsize=6); ax.set_xlabel("Cohen d, DR+ vs DR− (diabetics)")
        a = hrv["dr_within_dm"].get("auc") or {}
        ax.set_title(f"HRV → DR within diabetics, n = {hrv['dr_within_dm']['n']}\n({hrv['dr_within_dm']['n_dr']} DR+)" + (f", LOO AUC {a['auc']:.2f}" if a else ""), fontsize=7)
    else:
        ax.text(0.5, 0.5, "HRV run pending", ha="center", va="center", transform=ax.transAxes)
    ax.text(-0.25, 1.08, "d", transform=ax.transAxes, fontweight="bold")

    # e: PPG
    ax = axs[4]
    if ppg:
        keys = ["demographics_only", "morphology", "papagei_pca16", "papagei_plus_demo"]; labels = ["demogr.", "morphology", "PaPaGei", "PaPaGei+demo"]
        vals = [ppg["diabetes"][k]["auc"] for k in keys]; lo = [ppg["diabetes"][k]["auc"] - ppg["diabetes"][k]["auc_ci"][0] for k in keys]; hi = [ppg["diabetes"][k]["auc_ci"][1] - ppg["diabetes"][k]["auc"] for k in keys]
        x = np.arange(len(keys)); ax.bar(x, vals, color=C_PPG, yerr=[lo, hi], capsize=2, width=0.6)
        ax.bar([len(keys)], [ppg["hypertension"]["morphology"]["auc"]], color=ACC, width=0.6)
        ax.axhline(0.5, color="k", lw=0.6, ls=":"); ax.set_xticks(list(x) + [len(keys)]); ax.set_xticklabels(labels + ["HTN control"], fontsize=6, rotation=20)
        ax.set_ylim(0.2, 0.9); ax.set_ylabel("AUC (5-fold)"); ax.set_title(f"PPG-BP, n = {ppg['n_subjects']} ({ppg['n_diabetic']} diabetic)", fontsize=7)
    ax.text(-0.25, 1.08, "e", transform=ax.transAxes, fontweight="bold")

    # f: DME head on IDRiD
    ax = axs[5]; dme = load("fundus_dme_idrid.json")
    if dme:
        cm = np.array(dme["test_fixed_split"]["confusion"], float); cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xlabel("predicted DME grade"); ax.set_ylabel("true grade")
        t = dme["test_fixed_split"]
        ax.set_title(f"IDRiD DME test (n = {t['n']}): QWK {t['qwk']:.2f} [{t['qwk_ci'][0]:.2f}, {t['qwk_ci'][1]:.2f}]\nany-DME AUC {t['auc_any_dme']:.2f}", fontsize=7)
    else:
        ax.axis("off")
    ax.text(-0.25, 1.08, "f", transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "figR2_planB.png", dpi=600); fig.savefig(FIG / "figR2_planB.pdf")
    print("wrote", FIG / "figR2_planB.png")


if __name__ == "__main__":
    main()
