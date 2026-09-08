"""
Fundus DR-grade branch on open data (Plan B, 2026-09-07).

Train: DeepDRiD regular fundus (CC BY-SA 4.0), training + validation splits pooled
       (the challenge evaluation split has no public labels). Per-image label = the
       graded eye's ICDR level (0-4). PATIENT-LEVEL StratifiedGroupKFold (5 folds).
Test : IDRiD disease grading (CC BY 4.0) as an external set (--idrid DIR), and any
       other CSV of (path, grade) via --extra name=csv.
Model: timm ConvNeXt-Tiny (ImageNet-22k -> 1k) with an ordinal regression head
       (single output, SmoothL1 on the grade) -- the standard recipe behind the
       published APTOS/DeepDRiD QWK ~0.9. Rounding thresholds are optimised on the
       out-of-fold predictions (Nelder-Mead on QWK), never on test.
Metrics: quadratic weighted kappa (QWK), accuracy, referable-DR AUC (grade >= 2),
       with patient-level bootstrap 95 % CIs. Everything is written to
       results/out/fundus_deepdrid.json + fundus_oof.npz; nothing is printed that is
       not an aggregate.

Run (system python, GPU):  python results/src/train_fundus.py [--smoke] [--epochs 10] [--img 384]
"""
import sys, json, pathlib, time, argparse, math
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score, roc_auc_score, accuracy_score
from scipy.optimize import minimize
import timm

ROOT = pathlib.Path(__file__).resolve().parents[1]
DD = ROOT / "data" / "raw" / "open" / "deepdrid" / "regular_fundus_images"
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
SEED = 20260828


# ------------------------------------------------------------------ data ---
def load_deepdrid():
    rows = []
    for split in ("training", "validation"):
        df = pd.read_csv(DD / f"regular-fundus-{split}" / f"regular-fundus-{split}.csv")
        for _, r in df.iterrows():
            eye = "left" if "_l" in str(r.image_id) else "right"
            g = r[f"{eye}_eye_DR_Level"]
            if pd.isna(g):
                continue
            p = DD / f"regular-fundus-{split}" / "Images" / str(r.patient_id) / f"{r.image_id}.jpg"
            if p.exists():
                rows.append({"path": str(p), "pid": int(r.patient_id), "grade": int(g), "quality": r.get("Overall quality", np.nan)})
    return pd.DataFrame(rows)


def load_idrid(d):
    """IDRiD 'B. Disease Grading': 2. Groundtruths/*.csv with columns Image name, Retinopathy grade, Risk of macular edema."""
    d = pathlib.Path(d)
    rows = []
    for csv in d.rglob("*.csv"):
        if "Groundtruth" not in str(csv.parent) and "Groundtruths" not in str(csv.parent):
            continue
        df = pd.read_csv(csv)
        sub = "a. Training Set" if "Training" in csv.name else "b. Testing Set"
        for _, r in df.iterrows():
            name = str(r.iloc[0]).strip()
            # IDRiD reuses the same file names in "a. Training Set" and "b. Testing Set":
            # match inside the split's own image folder, never across splits
            imgs = [q for q in d.rglob(f"{name}.jpg") if sub in str(q.parent)]
            if imgs:
                rows.append({"path": str(imgs[0]), "pid": f"{sub[:1]}_{name}", "grade": int(r.iloc[1]), "dme": int(r.iloc[2]), "split": sub})
    return pd.DataFrame(rows)


CACHE = ROOT / "data" / "interim" / "fundus_cache"


def build_cache(df, size=512):
    """One-time: crop the black border and resize every image to `size` px (JPEG q95).
    Cuts epoch time ~15x (the raw DeepDRiD frames are 1736 x 1824)."""
    from concurrent.futures import ThreadPoolExecutor
    CACHE.mkdir(parents=True, exist_ok=True)
    def one(path):
        src = pathlib.Path(path)
        dst = CACHE / (src.parent.name + "_" + src.stem + f"_{size}.jpg")
        if not dst.exists():
            im = crop_black(Image.open(src).convert("RGB")).resize((size, size), Image.BICUBIC)
            im.save(dst, quality=95)
        return str(dst)
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, df.path.tolist()))


class FundusDS(Dataset):
    def __init__(self, df, img, train):
        self.df, self.img, self.train = df.reset_index(drop=True), img, train
        import torchvision.transforms as T
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
               T.ColorJitter(0.2, 0.2, 0.1, 0.02)] if train else []
        self.tf = T.Compose([T.Resize((img, img)), *aug, T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r.path).convert("RGB")          # cached, already cropped + resized
        return self.tf(im), torch.tensor(float(r.grade))


def crop_black(im, thr=12):
    """Crop the black border around the fundus circle (Ben Graham-style)."""
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a > thr)
    if len(ys) < 100:
        return im
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


# --------------------------------------------------------------- metrics ---
def apply_thresholds(pred, th):
    return np.digitize(pred, np.sort(th))


def fit_thresholds(pred, y):
    th0 = np.array([0.5, 1.5, 2.5, 3.5])
    f = lambda th: -cohen_kappa_score(y, apply_thresholds(pred, th), weights="quadratic")
    r = minimize(f, th0, method="Nelder-Mead", options={"xatol": 1e-3, "maxiter": 400})
    return np.sort(r.x)


def bootstrap(fn, y, pred, groups, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    g = np.asarray(groups); ug = np.unique(g); idx_by_g = {k: np.where(g == k)[0] for k in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by_g[k] for k in pick])
        try:
            vals.append(fn(y[idx], pred[idx]))
        except ValueError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise(y, pred, th, groups):
    yh = apply_thresholds(pred, th)
    qwk = lambda a, b: cohen_kappa_score(a, apply_thresholds(b, th), weights="quadratic")
    auc = lambda a, b: roc_auc_score(a >= 2, b)
    acc = lambda a, b: accuracy_score(a, apply_thresholds(b, th))
    out = {"n": int(len(y)), "n_patients": int(len(np.unique(groups))),
           "qwk": float(qwk(y, pred)), "qwk_ci": bootstrap(qwk, y, pred, groups),
           "auc_referable": float(auc(y, pred)), "auc_referable_ci": bootstrap(auc, y, pred, groups),
           "accuracy": float(acc(y, pred)), "accuracy_ci": bootstrap(acc, y, pred, groups),
           "grade_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
           "confusion": pd.crosstab(y, yh).reindex(index=range(5), columns=range(5), fill_value=0).values.tolist()}
    return out


# ----------------------------------------------------------------- train ---
def train_fold(tr, va, args, dev, tag):
    model = timm.create_model(args.model, pretrained=True, num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * math.ceil(len(tr) / args.bs))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    loss_fn = nn.SmoothL1Loss()
    dl_tr = DataLoader(FundusDS(tr, args.img, True), batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(FundusDS(va, args.img, False), batch_size=args.bs * 2, shuffle=False, num_workers=args.workers)
    best, best_pred, hist = -1.0, None, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tl = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = model(x).squeeze(1)
                loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y)
        pred = predict(model, dl_va, dev)
        q = cohen_kappa_score(va.grade.values, apply_thresholds(pred, [0.5, 1.5, 2.5, 3.5]), weights="quadratic")
        hist.append({"epoch": ep, "train_loss": tl / len(tr), "val_qwk": float(q), "s": time.time() - t0})
        print(f"  [{tag}] epoch {ep+1}/{args.epochs} loss {tl/len(tr):.3f} val QWK {q:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if q > best:
            best, best_pred = q, pred
            torch.save(model.state_dict(), OUT / f"fundus_{tag}.pt")
    return best_pred, hist


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); out = []
    for x, _ in dl:
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            p = model(x.to(dev)).squeeze(1).float()
            pf = model(torch.flip(x.to(dev), dims=[3])).squeeze(1).float()      # horizontal-flip TTA
        out.append(((p + pf) / 2).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--idrid", default=str(ROOT / "data" / "raw" / "open" / "idrid"))
    ap.add_argument("--eval-only", action="store_true", help="skip training; reuse fundus_fold*.pt and fundus_deepdrid.json")
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_deepdrid()
    print(f"DeepDRiD: {len(df)} labelled images, {df.pid.nunique()} patients, grades {df.grade.value_counts().sort_index().to_dict()}, device {dev}")
    t = time.time(); df["path"] = build_cache(df); print(f"cache ready ({time.time()-t:.0f}s)")
    if args.smoke:
        args.epochs, args.img = 1, 256
        df = pd.concat([g.sample(min(len(g), 40), random_state=SEED) for _, g in df.groupby("grade")]).reset_index(drop=True)
        print(f"SMOKE: {len(df)} images")
    if args.eval_only:
        res = json.loads((OUT / "fundus_deepdrid.json").read_text(encoding="utf-8")); th = np.array(res["thresholds"]); folds = res["folds"]
        oof = np.load(OUT / "fundus_oof.npz", allow_pickle=True)["oof"]      # our own file
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    if not args.eval_only:
        oof = np.full(len(df), np.nan); folds = []
    for k, (tr, va) in enumerate(sgkf.split(df, df.grade, df.pid)):
        if args.eval_only:
            break
        assert not set(df.pid.iloc[tr]) & set(df.pid.iloc[va])
        pred, hist = train_fold(df.iloc[tr], df.iloc[va], args, dev, f"fold{k}")
        oof[va] = pred; folds.append({"fold": k, "n_val": int(len(va)), "history": hist})
        if args.smoke:
            break
    m = ~np.isnan(oof)
    if not args.eval_only:
        th = fit_thresholds(oof[m], df.grade.values[m])
    res = res if args.eval_only else {"dataset": "DeepDRiD regular fundus (training+validation pooled), CC BY-SA 4.0", "model": args.model,
           "img": args.img, "epochs": args.epochs, "seed": SEED, "thresholds": th.tolist(),
           "cv": summarise(df.grade.values[m], oof[m], th, df.pid.values[m]), "folds": folds}
    print(f"\nOOF (patient-level 5-fold): QWK {res['cv']['qwk']:.3f} {res['cv']['qwk_ci']}, referable AUC {res['cv']['auc_referable']:.3f} {res['cv']['auc_referable_ci']}, acc {res['cv']['accuracy']:.3f}")
    np.savez(OUT / "fundus_oof.npz", oof=oof, grade=df.grade.values, pid=df.pid.values, path=df.path.values)
    # external test: IDRiD, ensemble of the fold models
    idr = load_idrid(args.idrid) if pathlib.Path(args.idrid).exists() else pd.DataFrame()
    if len(idr):
        idr["path"] = build_cache(idr)
        dl = DataLoader(FundusDS(idr, args.img, False), batch_size=args.bs * 2, num_workers=args.workers)
        preds = []
        for k in range(len(folds)):
            model = timm.create_model(args.model, pretrained=False, num_classes=1).to(dev)
            model.load_state_dict(torch.load(OUT / f"fundus_fold{k}.pt", map_location=dev, weights_only=True)); preds.append(predict(model, dl, dev))
        pe = np.mean(preds, axis=0)
        res["external_idrid"] = summarise(idr.grade.values, pe, th, idr.pid.values)
        res["external_idrid"]["note"] = "IDRiD has no patient ids: bootstrap is image-level"
        print(f"IDRiD external: QWK {res['external_idrid']['qwk']:.3f} {res['external_idrid']['qwk_ci']}, referable AUC {res['external_idrid']['auc_referable']:.3f}")
        np.savez(OUT / "fundus_idrid_pred.npz", pred=pe, grade=idr.grade.values, dme=idr.dme.values, path=idr.path.values)
    (OUT / ("fundus_deepdrid_smoke.json" if args.smoke else "fundus_deepdrid.json")).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote results")


if __name__ == "__main__":
    main()
