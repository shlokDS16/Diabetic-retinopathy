"""
Download the NHANES files needed for the clinical-risk-factor baseline.

Cycles 2005-2006 (suffix _D) and 2007-2008 (suffix _E) are the only two that
carry retinal imaging with ETDRS grading, done by the University of Wisconsin
Ocular Epidemiology Reading Center.

Files:
    DEMO    demographics (age, sex, ethnicity, survey weights)
    DIQ     diabetes questionnaire (diagnosis, age at diagnosis, insulin, oral meds)
    OPXRET  retinal imaging / retinopathy grading
    GHB     glycohemoglobin (HbA1c)
    BMX     body measures (BMI)
    BPX     blood pressure

Public domain, no registration. Run:  python results/src/fetch_nhanes.py
"""
import pathlib, urllib.request, urllib.error, hashlib, json, sys

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "nhanes"
RAW.mkdir(parents=True, exist_ok=True)

CYCLES = {"D": "2005", "E": "2007"}   # start year, per CDC path scheme
FILES = ["DEMO", "DIQ", "OPXRET", "GHB", "BMX", "BPX"]
URL = "https://wwwn.cdc.gov/nchs/data/nhanes/public/{cycle}/datafiles/{name}_{sfx}.xpt"
UA = {"User-Agent": "Mozilla/5.0 (research; NHANES public-domain retrieval)"}


def fetch(name, sfx):
    dest = RAW / f"{name}_{sfx}.XPT"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest, dest.stat().st_size, "cached"
    url = URL.format(cycle=CYCLES[sfx], name=name, sfx=sfx)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        return None, 0, f"HTTP {e.code}"
    except Exception as e:
        return None, 0, f"{type(e).__name__}"
    if not data.startswith(b"HEADER"):
        # CDC serves its 404 page with HTTP 200; magic bytes are the only honest check
        return None, 0, "not-SAS"
    dest.write_bytes(data)
    return dest, len(data), "downloaded"


if __name__ == "__main__":
    manifest = {}
    print(f"{'file':14s} {'status':12s} {'size':>12s}  sha256[:16]")
    print("-" * 60)
    for sfx in CYCLES:
        for name in FILES:
            dest, size, status = fetch(name, sfx)
            key = f"{name}_{sfx}"
            if dest is None:
                print(f"{key:14s} {status:12s} {'-':>12s}")
                manifest[key] = {"status": status}
                continue
            h = hashlib.sha256(dest.read_bytes()).hexdigest()
            print(f"{key:14s} {status:12s} {size:12,d}  {h[:16]}")
            manifest[key] = {"status": status, "bytes": size, "sha256": h,
                             "url": URL.format(cycle=CYCLES[sfx], name=name, sfx=sfx)}

    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = sum(1 for v in manifest.values() if v.get("bytes", 0) > 1000)
    print(f"\n{ok}/{len(manifest)} files retrieved -> {RAW}")
    print(f"manifest -> {RAW/'manifest.json'}")
