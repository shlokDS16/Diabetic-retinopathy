"""
Second external test for the DeepDRiD-trained fundus DR model: APTOS 2019 Blindness
Detection (Aravind Eye Hospital, India; 3,662 labelled fundus photographs, ICDR grades 0-4).

Sources accepted (first found wins, provenance recorded in the JSON):
  A. official competition download  -> data/raw/open/aptos2019/train.csv + train_images/*.png
     (kaggle competitions download -c aptos2019-blindness-detection; requires accepting the
      competition rules once on kaggle.com; rules permit non-commercial academic research)
  B. community mirror mariaherrerot/aptos2019 (same images, re-split; licence "Unknown" on
     Kaggle, so the paper cites the competition, not the mirror)
     (kaggle datasets download -d mariaherrerot/aptos2019  or  kagglehub.dataset_download)

Loader is layout-agnostic: every CSV with an `id_code` and `diagnosis` column is read, images
are located by stem anywhere under the root, duplicates by id_code are dropped. APTOS has no
patient ids, so the bootstrap is image-level (same caveat as IDRiD). No training happens
here: the five DeepDRiD fold models and the DeepDRiD-fitted thresholds are applied unchanged.

Run (system python, GPU):  python results/src/eval_aptos.py [--root <dir>]
"""
import os, sys, json, pathlib, argparse, subprocess, zipfile, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train_fundus as TF                                   # noqa: E402

OUT = TF.OUT
DEFAULT_ROOT = TF.ROOT / "data" / "raw" / "open" / "aptos2019"
IMG_EXT = (".png", ".jpg", ".jpeg")


def try_download(root: pathlib.Path):
    """Attempt A (competition) then B (mirror) with the kaggle CLI. Token is read by the CLI
    from USERPROFILE/.kaggle/access_token or KAGGLE_API_TOKEN; nothing is copied here."""
    home = pathlib.Path.home() / ".kaggle"
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN")) or (home / "access_token").exists() or (home / "kaggle.json").exists()
    root.mkdir(parents=True, exist_ok=True)
    attempts = [("mirror", [sys.executable, "-m", "kaggle", "datasets", "download", "-d", "mariaherrerot/aptos2019", "-p", str(root)])]
    if has_token:      # the official competition needs a token + one-time rules acceptance on kaggle.com; the public mirror downloads anonymously
        attempts.insert(0, ("competition", [sys.executable, "-m", "kaggle", "competitions", "download", "-c", "aptos2019-blindness-detection", "-p", str(root)]))
    else:
        print("[download] no Kaggle token found: skipping the official competition, using the public mirror", flush=True)
    for tag, cmd in attempts:
        print(f"[download] trying {tag}: {' '.join(cmd)}", flush=True)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=3 * 3600)
        except subprocess.TimeoutExpired:
            print(f"[download] {tag} timed out", flush=True); continue
        if r.returncode == 0:
            for z in root.glob("*.zip"):
                print(f"[download] extracting {z.name} ({z.stat().st_size/1e9:.1f} GB)", flush=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(root)
                z.unlink()
            (root / "SOURCE.txt").write_text(f"{tag}\n{' '.join(cmd)}\n{time.strftime('%Y-%m-%d %H:%M')}\n", encoding="utf-8")
            return tag
        print(f"[download] {tag} failed: {(r.stderr or r.stdout).strip()[:300]}", flush=True)
    return None


def load_aptos(root: pathlib.Path) -> pd.DataFrame:
    stems = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            stems.setdefault(p.stem, p)
    rows = []
    for csv in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        if "id_code" not in cols or "diagnosis" not in cols:
            continue                                        # e.g. the unlabelled competition test.csv
        for _, r in df.iterrows():
            k = str(r[cols["id_code"]]).strip()
            if k in stems and pd.notna(r[cols["diagnosis"]]):
                rows.append({"id_code": k, "path": str(stems[k]), "grade": int(r[cols["diagnosis"]]), "csv": csv.name})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates("id_code").reset_index(drop=True)
        df["pid"] = df.id_code                             # no patient ids in APTOS
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT)); ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    args = ap.parse_args(); root = pathlib.Path(args.root)
    df = load_aptos(root) if root.exists() else pd.DataFrame()
    source = (root / "SOURCE.txt").read_text(encoding="utf-8").split()[0] if (root / "SOURCE.txt").exists() else "pre-existing"
    if not len(df):
        source = try_download(root)
        if source is None:
            sys.exit("APTOS not on disk and download failed. Put the token in USERPROFILE/.kaggle/access_token "
                     "(see results/DATA_RETRIEVAL.md, APTOS section) or download manually into " + str(root))
        df = load_aptos(root)
    if not len(df):
        sys.exit(f"No labelled APTOS images found under {root}")
    print(f"APTOS 2019 ({source}): {len(df)} labelled images, grades {df.grade.value_counts().sort_index().to_dict()}, csvs {sorted(df.csv.unique())}")
    res_dd = json.loads((OUT / "fundus_deepdrid.json").read_text(encoding="utf-8"))
    th = np.array(res_dd["thresholds"]); n_folds = len(res_dd["folds"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = time.time(); df["path"] = TF.build_cache(df); print(f"cache ready ({time.time()-t:.0f}s)", flush=True)
    dl = DataLoader(TF.FundusDS(df, args.img, False), batch_size=args.bs, num_workers=args.workers)
    preds = []
    for k in range(n_folds):
        model = TF.timm.create_model(args.model, pretrained=False, num_classes=1).to(dev)
        model.load_state_dict(torch.load(OUT / f"fundus_fold{k}.pt", map_location=dev, weights_only=True))
        preds.append(TF.predict(model, dl, dev)); print(f"  fold {k} done", flush=True)
    pe = np.mean(preds, axis=0)
    res = {"dataset": "APTOS 2019 Blindness Detection (Aravind Eye Hospital, India), ICDR 0-4", "source_on_disk": source,
           "citation": "Asia Pacific Tele-Ophthalmology Society, APTOS 2019 Blindness Detection, Kaggle 2019",
           "model": args.model, "img": args.img, "thresholds_from": "DeepDRiD OOF (unchanged)", "n_fold_models": n_folds,
           "external_aptos": TF.summarise(df.grade.values, pe, th, df.pid.values)}
    res["external_aptos"]["note"] = "APTOS has no patient ids: bootstrap is image-level"
    # secondary: thresholds re-fitted on APTOS itself (reported separately, NOT the headline)
    th_a = TF.fit_thresholds(pe, df.grade.values)
    res["external_aptos_refit_thresholds"] = {"thresholds": th_a.tolist(), "qwk": float(TF.cohen_kappa_score(df.grade.values, TF.apply_thresholds(pe, th_a), weights="quadratic"))}
    e = res["external_aptos"]
    print(f"APTOS external: QWK {e['qwk']:.3f} {e['qwk_ci']}, referable AUC {e['auc_referable']:.3f} {e['auc_referable_ci']}, acc {e['accuracy']:.3f}; refit-threshold QWK {res['external_aptos_refit_thresholds']['qwk']:.3f}")
    (OUT / "fundus_aptos.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    np.savez(OUT / "fundus_aptos_pred.npz", pred=pe, grade=df.grade.values, id_code=df.id_code.values)
    print("wrote fundus_aptos.json")


if __name__ == "__main__":
    main()
