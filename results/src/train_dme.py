"""
Macular-edema (DME) head on IDRiD disease grading (CC BY 4.0) -- the only open fundus set
with a DME grade (0 = none, 1 = hard exudates outside 1 DD of the fovea, 2 = within 1 DD).

IDRiD ships a fixed split: 413 training / 103 test images. We train on the training images
(5-fold CV inside for model selection and OOF metrics) and report the FIXED test split once,
with image-level bootstrap CIs (IDRiD has no patient ids). Backbone and recipe identical to
train_fundus.py (ConvNeXt-Tiny, ordinal regression), initialised from ImageNet (not from the
DR fold models, to keep the two heads independent).

Run (system python, GPU):  python results/src/train_dme.py [--epochs 12]
"""
import sys, json, pathlib, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, roc_auc_score
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train_fundus as TF                                   # noqa: E402

OUT = TF.OUT


def summarise_dme(y, pred, th, name):
    yh = TF.apply_thresholds(pred, th)
    idx = np.arange(len(y))
    qwk = lambda a, b: cohen_kappa_score(a, TF.apply_thresholds(b, th), weights="quadratic")
    auc = lambda a, b: roc_auc_score(a >= 1, b)
    return {"n": int(len(y)), "qwk": float(qwk(y, pred)), "qwk_ci": TF.bootstrap(qwk, y, pred, idx),
            "auc_any_dme": float(auc(y, pred)), "auc_any_dme_ci": TF.bootstrap(auc, y, pred, idx),
            "confusion": pd.crosstab(y, yh).reindex(index=range(3), columns=range(3), fill_value=0).values.tolist(),
            "counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    args = ap.parse_args(); torch.manual_seed(TF.SEED); np.random.seed(TF.SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idr = TF.load_idrid(TF.ROOT / "data" / "raw" / "open" / "idrid")
    idr["path"] = TF.build_cache(idr)
    idr = idr.rename(columns={"grade": "dr_grade"}); idr["grade"] = idr["dme"]      # FundusDS reads .grade
    tr, te = idr[idr.split.str.contains("Training")].reset_index(drop=True), idr[idr.split.str.contains("Testing")].reset_index(drop=True)
    print(f"IDRiD DME: train {len(tr)} {tr.grade.value_counts().sort_index().to_dict()}, test {len(te)} {te.grade.value_counts().sort_index().to_dict()}")
    skf = StratifiedKFold(5, shuffle=True, random_state=TF.SEED)
    oof = np.full(len(tr), np.nan); folds = []; test_preds = []
    for k, (a, b) in enumerate(skf.split(tr, tr.grade)):
        pred, hist = TF.train_fold(tr.iloc[a], tr.iloc[b], args, dev, f"dme{k}")
        oof[b] = pred; folds.append({"fold": k, "history": hist})
        model = TF.timm.create_model(args.model, pretrained=False, num_classes=1).to(dev)
        model.load_state_dict(torch.load(OUT / f"fundus_dme{k}.pt", map_location=dev, weights_only=True))
        test_preds.append(TF.predict(model, DataLoader(TF.FundusDS(te, args.img, False), batch_size=32, num_workers=args.workers), dev))
    th = np.sort(TF.minimize(lambda t: -cohen_kappa_score(tr.grade.values, TF.apply_thresholds(oof, t), weights="quadratic"),
                             np.array([0.5, 1.5]), method="Nelder-Mead").x)
    res = {"dataset": "IDRiD disease grading, DME 0-2 (CC BY 4.0)", "model": args.model, "img": args.img, "epochs": args.epochs,
           "thresholds": th.tolist(), "cv_train": summarise_dme(tr.grade.values, oof, th, "cv"),
           "test_fixed_split": summarise_dme(te.grade.values, np.mean(test_preds, axis=0), th, "test"), "folds": folds}
    print(f"DME  CV (413): QWK {res['cv_train']['qwk']:.3f} {res['cv_train']['qwk_ci']}, any-DME AUC {res['cv_train']['auc_any_dme']:.3f}")
    print(f"DME test (103): QWK {res['test_fixed_split']['qwk']:.3f} {res['test_fixed_split']['qwk_ci']}, any-DME AUC {res['test_fixed_split']['auc_any_dme']:.3f} {res['test_fixed_split']['auc_any_dme_ci']}")
    (OUT / "fundus_dme_idrid.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    np.savez(OUT / "dme_idrid_pred.npz", oof=oof, y_train=tr.grade.values, test_pred=np.mean(test_preds, axis=0), y_test=te.grade.values)
    print("wrote fundus_dme_idrid.json")


if __name__ == "__main__":
    main()
