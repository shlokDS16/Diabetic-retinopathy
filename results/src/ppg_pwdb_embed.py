"""
PPG branch validation (step A6, PPG): PaPaGei embeddings over PWDB.

Site   SupTemporal — the superficial temporal artery, i.e. the exact site of
       the device's temple-arm PPG sensor. PWDB simulates our placement.
Data   PWDB (Charlton et al.), 4,374 virtual subjects, ODC public domain.
       Each subject provides one steady-state cardiac cycle of PPG.
Test   Tile the cycle at the subject's own HR into a 10-s stream, resample to
       125 Hz (PaPaGei's contract), z-score, embed with frozen PaPaGei-S.
       Then linear-probe the 512-d embeddings for age and carotid-femoral PWV
       (the arterial-stiffness axis used as the diabetes-ageing proxy).

If the probes recover age/PWV well, the premise "PaPaGei embeddings carry
vascular-ageing information at our measurement site" is validated on data we
did not generate ourselves.

Caveat stated up front: PWDB is a healthy-ageing model. Stiffness here is a
PROXY axis for diabetic vascular change, not diabetes itself.

Run:  python results/src/ppg_pwdb_embed.py
"""
import pathlib, sys, json, zipfile, io, re, time
import numpy as np
import pandas as pd

BASE = pathlib.Path(r"C:\Users\Shlok\Downloads\Patent and research paper on diabetic retinopathy")
D = BASE / "results" / "data" / "raw" / "pwdb"
OUT = BASE / "results" / "out"
SEED = 20260828
FS_TARGET, SEG_S = 125.0, 10.0
SITE = "SupTemporal"


def discover(z):
    """Find per-subject PPG files for the site inside PWs_csv.zip."""
    names = z.namelist()
    cand = [n for n in names if SITE.lower() in n.lower() and "ppg" in n.lower()]
    if not cand:
        cand = [n for n in names if "ppg" in n.lower()]
    return names, cand


def main():
    import torch
    sys.path.insert(0, str(BASE / "results" / "vendor" / "papagei"))
    from models.resnet import ResNet1DMoE

    zpath = D / "PWs_csv.zip"
    if not zpath.exists():
        sys.exit("PWs_csv.zip not downloaded yet")

    z = zipfile.ZipFile(zpath)
    names, cand = discover(z)
    print(f"zip members: {len(names)}; candidate PPG files: {len(cand)}")
    for n in cand[:5]:
        print("   ", n)

    # ---- load PPG matrix ---------------------------------------------------
    # PWDB csv layout: one file per site/signal, rows = subjects (verified below)
    target = cand[0]
    raw = z.read(target).decode()
    df = pd.read_csv(io.StringIO(raw))          # row 0 is a header
    subj_col = df.columns[0]                    # "Subject Number"
    subjects = df[subj_col].to_numpy()
    df = df.drop(columns=[subj_col])
    print(f"loaded {target}: shape {df.shape} (+{subj_col!r} column)")

    hp = pd.read_csv(D / "pwdb_haemod_params.csv")
    hp.columns = [c.strip() for c in hp.columns]
    n_subj = len(hp)
    M = df.to_numpy(float)
    assert M.shape[0] == n_subj, f"row/subject mismatch: {M.shape} vs {n_subj}"
    # align haemod rows to the waveform file's subject order
    hp = hp.set_index("Subject Number").loc[subjects].reset_index()
    print(f"waveform matrix: {M.shape} (subjects x samples), aligned to "
          f"haemod params by subject id")

    hr = hp["HR [bpm]"].to_numpy(float)
    age = hp["age [years]"].to_numpy(float)
    pwv = hp["PWV_cf [m/s]"].to_numpy(float)

    # ---- build 10-s 125 Hz streams -----------------------------------------
    # Each row is one cardiac cycle sampled at PWDB's dt (500 Hz onset-aligned;
    # trailing NaNs pad to the longest cycle). Cycle duration = 60/HR.
    n_out = int(FS_TARGET * SEG_S)
    X = np.empty((len(M), 1, n_out), dtype=np.float32)
    for i, row in enumerate(M):
        w = row[np.isfinite(row)]
        period_s = 60.0 / hr[i]
        n_cyc = int(np.ceil(SEG_S / period_s)) + 1
        # resample one cycle to its true duration at 125 Hz, then tile
        n_cycle_samples = max(8, int(round(period_s * FS_TARGET)))
        t_src = np.linspace(0, 1, len(w))
        t_dst = np.linspace(0, 1, n_cycle_samples)
        cyc = np.interp(t_dst, t_src, w)
        stream = np.tile(cyc, n_cyc)[:n_out]
        mu, sd = stream.mean(), stream.std()
        X[i, 0] = (stream - mu) / (sd if sd > 1e-9 else 1.0)

    print(f"streams: {X.shape}, all finite: {np.isfinite(X).all()}")

    # ---- embed -------------------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2,
                    groups=1, n_block=18, n_classes=512, n_experts=3)
    sd_ = torch.load(BASE/"results"/"data"/"raw"/"papagei"/"papagei_s.pt",
                     map_location="cpu", weights_only=True)
    m.load_state_dict({(k[7:] if k.startswith("module.") else k): v
                       for k, v in sd_.items()})
    m.eval().to(dev)

    embs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i+256]).to(dev)
            embs.append(m(xb)[0].cpu().numpy())
    E = np.concatenate(embs)
    print(f"embeddings: {E.shape} in {time.perf_counter()-t0:.1f}s on {dev}")

    # ---- linear probes -----------------------------------------------------
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from scipy.stats import pearsonr

    res = {}
    cv = KFold(5, shuffle=True, random_state=SEED)
    for name, y in [("age", age), ("pwv_cf", pwv)]:
        pipe = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=10.0))])
        p = cross_val_predict(pipe, E, y, cv=cv)
        r = pearsonr(p, y)[0]
        mae = np.abs(p - y).mean()
        ss = 1 - np.sum((p-y)**2)/np.sum((y-y.mean())**2)
        res[name] = {"r": float(r), "r2": float(ss), "mae": float(mae)}
        print(f"  probe {name:8s}  r = {r:.3f}   R2 = {ss:.3f}   MAE = {mae:.2f}")

    np.savez_compressed(OUT / "pwdb_suptemporal_embeddings.npz",
                        E=E, age=age, pwv=pwv, hr=hr,
                        subject=hp["Subject Number"].to_numpy())
    json.dump({"site": SITE, "n": int(len(E)), "device": dev,
               "probes": res, "source_file": target, "seed": SEED},
              open(OUT / "ppg_pwdb_probe.json", "w"), indent=2)
    print(f"\nwrote pwdb_suptemporal_embeddings.npz and ppg_pwdb_probe.json")


if __name__ == "__main__":
    main()
