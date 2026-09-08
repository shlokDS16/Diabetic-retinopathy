# DATA_RETRIEVAL.md — BRSET + mBRSET from PhysioNet (verified against live pages 2026-09-04)

Scope: exact retrieval, compliance and ingest checklist for the two credentialed PhysioNet datasets.
Every fact carries the URL it was read from. Items marked **[UNVERIFIED]** could not be confirmed from a public page (PhysioNet hides the Files table, size and wget line of restricted projects until you are logged in and have signed the DUA).

Local machine facts (checked on this PC 2026-09-04): `wget` is **not** installed (PowerShell's `wget` is only an alias of `Invoke-WebRequest`; Git Bash has no `wget.exe`). Real `curl.exe` exists at `C:\Windows\system32\curl.exe`. Python 3.14 is at `C:\Python314\python.exe`.

---

## 0. Quick path (what to do tonight, in order)

1. Log in at https://physionet.org/login/ .
2. Check https://physionet.org/settings/credentialing/ shows credentialed, and https://physionet.org/settings/training/ shows "CITI Data or Specimens Only Research" as accepted (see section 2).
3. Open https://physionet.org/content/brazilian-ophthalmological/1.0.2/ , scroll to the bottom "Files" box, click **"sign the data use agreement for the project"** (the link goes to `/sign-dua/brazilian-ophthalmological/1.0.2/`), read, tick, sign.
4. Repeat on https://physionet.org/content/mbrset/1.0/ (link goes to `/sign-dua/mbrset/1.0/`).
5. Reload each project page; the Files section now shows "Total uncompressed size", the file table, the ZIP button and the wget command. Write the sizes into section 1 below.
6. Install wget (`winget install -e --id JernejSimoncic.Wget`) and run the two wget commands in section 3 into `data/raw/` (which must be git-ignored).
7. Verify with `SHA256SUMS.txt` (section 5), then run the ingest checks.

---

## 1. Dataset facts

### 1a. BRSET — A Brazilian Multilabel Ophthalmological Dataset

| Item | Value | Source |
|---|---|---|
| Canonical URL (latest) | https://physionet.org/content/brazilian-ophthalmological/ (resolves to 1.0.2) | https://physionet.org/content/brazilian-ophthalmological/ |
| Latest version | **1.0.2**, published July 27, 2026 | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ |
| Version history | 1.0.0 (Mar 8, 2023); 1.0.1 (Aug 14, 2024); 1.0.2 (Jul 27, 2026) | https://physionet.org/content/brazilian-ophthalmological/1.0.1/ and .../1.0.2/ |
| 1.0.2 release notes | "Comprehensive review of `exam_eye` column mapping"; typo fixes `Illuminaton`→`illumination`, `drusens`→`drusen`, `insuline`→`insulin` | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ |
| 1.0.1 release notes | `SAH`→`hypertension` in comorbidities; `ilumination`→`illumination`; review of AMD labels; review of DR labels (DR_ICDR, DR_SDRG) | https://physionet.org/content/brazilian-ophthalmological/1.0.1/ |
| Consequence | **Use 1.0.2 only.** Any code written against 1.0.0/1.0.1 column names (`insuline_use`, `drusens`, `ilumination`) will break, and the eye-laterality column was remapped. | derived from the two release notes above |
| Images / patients | **16,266 fundus photographs from 8,524 patients**, collected 2010–2020 | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ ; https://github.com/luisnakayama/BRSET |
| Image format | JPEG, "no preprocessing applied"; macula-centred, 45-degree field, optic disc visible; fluorescein angiograms, non-retinal images and duplicates excluded | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ |
| Resolution | Variable. Height 874–2304 px, width 951–2984 px (paper). Canon CR-2: 10,592 images (65.1 %); Nikon NF505: 5,674 (34.9 %) | https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000454 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11239107/ |
| Cameras | Nikon NF505 and Canon CR-2 (labels.csv `camera` values are "Canon CR" and "NIKON NF5050") | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ |
| Folder structure | `fundus_photos/` (16,266 JPEGs) + `labels.csv` (+ PhysioNet-standard `LICENSE.txt`, `SHA256SUMS.txt`) | folder names: https://physionet.org/content/brazilian-ophthalmological/ ; standard files: pattern confirmed on public project https://physionet.org/files/mitdb/1.0.0/ **[UNVERIFIED for this project until logged in]** |
| Total size (GB) | **Not displayed publicly.** Estimate **[UNVERIFIED]**: 16,266 JPEGs at 1–3 MB each ≈ 15–45 GB. Read the exact "Total uncompressed size" line in the Files section after signing the DUA and record it here: `______ GB` | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ (line hidden pre-DUA) |
| Licence | **PhysioNet Credentialed Health Data License 1.5.0** | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ |
| DUA | **PhysioNet Credentialed Health Data Use Agreement 1.5.0** | https://physionet.org/content/brazilian-ophthalmological/view-dua/1.0.2/ |
| Required training | "CITI Data or Specimens Only Research" | https://physionet.org/content/brazilian-ophthalmological/view-required-training/1.0.2/ |
| Ethics | UNIFESP IRB, CAAE 33842220.7.0000.5505, individual consent waived | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ |
| DOI (v1.0.2) | https://doi.org/10.13026/mysn-8b26 | https://physionet.org/content/brazilian-ophthalmological/ |
| DOI (latest, version-independent) | https://doi.org/10.13026/z9zv-g542 | https://physionet.org/content/brazilian-ophthalmological/ |

**labels.csv columns (v1.0.2)** — source: https://physionet.org/content/brazilian-ophthalmological/1.0.2/

| Group | Columns / coding |
|---|---|
| Identifiers | `image_id` (file name in `fundus_photos/`), `patient_id` |
| Demographics | `patient_age`, `patient_sex` (1 = male, 2 = female), `nationality` |
| Acquisition | `camera` ("Canon CR" / "NIKON NF5050"), `exam_eye` (1 = right, 2 = left — remapped in 1.0.2) |
| Diabetes | `diabetes` (diagnosis), `diabetes_time` (duration, years; ~88 % missing because only recorded for diabetics), `insulin_use` (~89 % missing, same reason), `comorbidities` (free text incl. "hypertension") |
| **HbA1c** | **Not present** (not in PhysioNet column list, not in the PLOS paper) |
| Anatomy | `optic_disc`, `vessels`, `macula` (1 = normal, 2 = abnormal) |
| **DR grade** | `DR_ICDR` 0–4 (0 none, 1 mild NPDR, 2 moderate NPDR, 3 severe NPDR, 4 PDR incl. post-laser status); `DR_SDRG` 0–4 (Scottish grading) |
| Quality | `focus`, `illumination`, `image_field`, `artifacts` (1 = adequate/normal, 2 = inadequate/abnormal) |
| Pathology binaries (0/1) | `diabetic_retinopathy`, **`macular_edema`**, `scar`, `nevus`, `amd`, `vascular_occlusion`, `hypertensive_retinopathy`, `drusen` (was `drusens` before 1.0.2), `hemorrhage`, `retinal_detachment`, `myopic_fundus`, `increased_cup_disc`, `other` |

Missing-rate figures for `insulin_use` / `diabetes_time` are from the PLOS paper: https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000454 . Macular edema is defined per ICDR as exudates or apparent thickening within one disc diameter of the fovea (https://pmc.ncbi.nlm.nih.gov/articles/PMC11239107/). The paper reports 2,579 patients (15.8 %) with diabetes; of those about 25.5 % have some DR grade (https://pmc.ncbi.nlm.nih.gov/articles/PMC11239107/) — confirm counts from the CSV after download.

**Citations the paper must use (BRSET)** — source: https://physionet.org/content/brazilian-ophthalmological/

- Dataset: Nakayama, L. F., Goncalves, M., Zago Ribeiro, L., Santos, H., Ferraz, D., Malerbi, F., Celi, L. A., & Regatieri, C. (2026). A Brazilian Multilabel Ophthalmological Dataset (BRSET) (version 1.0.2). PhysioNet. https://doi.org/10.13026/mysn-8b26
- Original paper: Nakayama LF, Restrepo D, Matos J, Ribeiro LZ, Malerbi FK, Celi LA, Regatieri CS. BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos. PLOS Digit Health. 2024;3(7):e0000454. https://doi.org/10.1371/journal.pdig.0000454
- PhysioNet platform (the page now asks for this instead of the 2000 Goldberger paper): Pollard, T., Moody, B. E., Lehman, L., et al. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z

### 1b. mBRSET — a Mobile Brazilian Retinal Dataset

| Item | Value | Source |
|---|---|---|
| Canonical URL | https://physionet.org/content/mbrset/ → **slug `mbrset`, version `1.0`** (not 1.0.0) | https://physionet.org/content/mbrset/1.0/ |
| Latest version | **1.0**, published June 26, 2024; single release ("first public release; future expansions planned") | https://physionet.org/content/mbrset/ |
| Images / patients | **5,164 images from 1,291 patients with diabetes**, Itabuna Diabetes Campaign, Bahia, Brazil, November 2022 | https://physionet.org/content/mbrset/1.0/ |
| Image format / resolution | JPEG, **1600 × 1600 px**, 45-degree field, 12-megapixel sensor, no preprocessing | https://physionet.org/content/mbrset/1.0/ |
| Camera | **Phelcom Eyer** — portable handheld smartphone-based fundus camera built on a Samsung Galaxy S10 | https://physionet.org/content/mbrset/1.0/ |
| Folder structure | `labels_mbrset.csv` + one directory of `.jpg` images keyed by the `file` column. Exact image-folder name **[UNVERIFIED]** (paper only says "a directory with the corresponding images in a .jpg format"); record it after download: `______/` | https://physionet.org/content/mbrset/1.0/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11846882/ |
| Total size (GB) | **Not displayed publicly.** Estimate **[UNVERIFIED]**: 5,164 × ~0.5–1.2 MB ≈ 3–6 GB. Record actual: `______ GB` | https://physionet.org/content/mbrset/1.0/ (hidden pre-DUA) |
| Licence | PhysioNet Credentialed Health Data License 1.5.0 | https://physionet.org/content/mbrset/1.0/ |
| DUA | PhysioNet Credentialed Health Data Use Agreement 1.5.0 | https://physionet.org/content/mbrset/view-dua/1.0/ |
| Required training | "CITI Data or Specimens Only Research" | https://physionet.org/content/mbrset/1.0/ |
| DOI (v1.0) | https://doi.org/10.13026/qxpd-1y65 | https://physionet.org/content/mbrset/ |
| DOI (latest) | https://doi.org/10.13026/m10j-dj56 | https://physionet.org/content/mbrset/ |
| Code repo | https://github.com/luisnakayama/mBRSET (uses Python 3.9.7; its internal CSV is named `dataframe_brsetmobile.csv`, which is NOT the PhysioNet file name) | https://github.com/luisnakayama/mBRSET |

**labels_mbrset.csv columns (v1.0)** — source: https://physionet.org/content/mbrset/1.0/ (full list quoted from the page)

`patient, age, sex, dm_time, insulin, insulin_time, oraltreatment_dm, systemic_hypertension, insurance, educational_level, alcohol_consumption, smoking, obesity, vascular_disease, acute_myocardial_infarction, nephropathy, neuropathy, diabetic_foot, file, laterality, final_artifacts, final_quality, final_icdr, final_edema`

| Group | Columns / coding |
|---|---|
| Identifiers | `patient` (patient id — use for splits), `file` (image file name), `laterality` (left / right eye) |
| Demographics | `age`, `sex` (0 = female, 1 = male), `insurance`, `educational_level` (1–7, illiterate → complete tertiary) |
| Diabetes | `dm_time` (duration, years), `insulin` (0/1), `insulin_time` (years), `oraltreatment_dm` |
| **HbA1c** | **Not present** (not in column list, not in the Scientific Data paper) |
| Comorbidities (self-reported, 0/1) | `systemic_hypertension`, `smoking`, `obesity`, `vascular_disease`, `acute_myocardial_infarction`, `nephropathy`, **`neuropathy`** (self-reported), `diabetic_foot`, `alcohol_consumption` |
| Quality | `final_quality` (acceptable in 94.3 % of images), `final_artifacts` (artifacts present in 82.7 %) |
| **DR grade** | `final_icdr` 0–4: 0 no DR (76.8 %), 1 mild NPDR (5.6 %), 2 moderate NPDR (11.6 %), 3 severe NPDR (1.7 %), 4 PDR (4.3 %) |
| **DME** | `final_edema` yes/no (present in 8.7 % of images) |

Prevalence figures and inter-rater agreement (weighted kappa 0.863 ICDR, 0.618 edema) from https://pmc.ncbi.nlm.nih.gov/articles/PMC11846882/ . Check the exact string coding of `final_edema` ("yes"/"no" vs 0/1) and `laterality` in the CSV after download before writing parsers.

**Citations the paper must use (mBRSET)** — source: https://physionet.org/content/mbrset/1.0/

- Dataset: Nakayama, L. F., Zago Ribeiro, L., Restrepo, D., Santos Barboza, N., Dias Fiterman, R., Vieira Sousa, M. l., Pereira, A. D. A., Regatieri, C., Malerbi, F. K., & Andrade, R. (2024). mBRSET, a Mobile Brazilian Retinal Dataset (version 1.0). PhysioNet. https://doi.org/10.13026/qxpd-1y65
- Original paper: Wu, C., et al. A portable retina fundus photos dataset for clinical, demographic, and diabetic retinopathy prediction. Scientific Data 12, 323 (2025). https://doi.org/10.1038/s41597-025-04627-3 (author order per https://pmc.ncbi.nlm.nih.gov/articles/PMC11846882/ ; copy the full author list from the PMC page when writing the bibliography)
- PhysioNet platform: Pollard, T., Moody, B. E., Lehman, L., et al. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z

Related but not required: "Embedding-Based Representations for BRSET and mBRSET" v1.0.0 (Mar 30, 2026), credentialed, precomputed DINOv3/ConvNeXt embeddings as six CSVs — https://physionet.org/content/embedding-brset-mbrset/1.0.0/ . Useful as a baseline if raw-image training is too slow, same DUA regime.

---

## 2. Signing the DUA — exact click path and what to check first

**What the public page shows (both projects, identical wording)** — read from the raw HTML of https://physionet.org/content/brazilian-ophthalmological/1.0.2/ and https://physionet.org/content/mbrset/1.0/ :

> Files — This is a restricted-access resource. To access the files, you must fulfill all of the following requirements: be a credentialed user; complete required training: CITI Data or Specimens Only Research (You may submit your training here); sign the data use agreement for the project.

The three lines are a checklist; each renders green once satisfied for your logged-in account. "You may submit your training here" links to https://physionet.org/settings/training/ ; "sign the data use agreement for the project" links to `/sign-dua/brazilian-ophthalmological/1.0.2/` and `/sign-dua/mbrset/1.0/` respectively.

**The official three-step process** (https://physionet.org/news/post/395/):
1. Credentialing: https://physionet.org/settings/credentialing/
2. Training: https://physionet.org/settings/training/ — upload the CITI **report**, click **"Submit Training"** (button label from https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/).
3. DUA: "sign the Data Use Agreement in the 'Files' section of the relevant project". Signed agreements are listed at https://physionet.org/settings/agreements/ .

**Click path per project (tonight):**
1. Log in. Open the project page (URLs above).
2. Scroll to the very bottom, to the Files box. Third-party walkthroughs describe it as "a red box reading: 'sign the data use agreement for the project'. Click that to agree." (https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/).
3. The sign page shows the DUA 1.5.0 text; a MIT-LCP maintainer describes signing as "just a quick click of a button" (https://github.com/MIT-LCP/mimic-code/discussions/1640). Exact checkbox/button labels **[UNVERIFIED]** — read the page.
4. If the sign page instead says **"In order to access a restricted-access resource, you must first be a credentialed user."** (https://github.com/MIT-LCP/mimic-code/discussions/1640), your credentialing is not yet approved — go to step 2 below.
5. After signing, reload the project page: the Files section now shows the size, ZIP button, wget command and file table.

**If training shows pending / insufficient:**
- Open https://physionet.org/settings/training/ (login required; without login it redirects to the login page — confirmed https://physionet.org/settings/training/). It lists each submitted training with its status. If the "CITI Data or Specimens Only Research" row is not accepted, the DUA link stays blocked.
- Requirements for the CITI report (https://physionet.org/about/citi-course/): course "Data or Specimens Only Research" with modules "Data or Specimens Only Research" and "Conflicts of Interest"; affiliate as "Massachusetts Institute of Technology Affiliates" (free, no MIT address needed); answer questionnaire items 1, 2, 3 and 5 ("Yes" for conflicts of interest); upload the **training report** (Records → View-Print-Share), **not** the certificate. DrivenData adds that the course needs an overall score of 90 % or higher (https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/) **[UNVERIFIED on physionet.org]**.
- Common rejection reasons: uploading the certificate instead of the report; report missing the Conflicts of Interest module; name on report not matching PhysioNet profile.
- Review time: PhysioNet FAQ says "several days to one week (business days)" (https://physionet.org/about/faqs/); other guides say 24–48 h but "not a guaranteed turnaround" (https://casrai.org/guides/physionet-credentialed-access-restricted-data). If credentialing is stuck more than a week, your reference may have an unanswered email from credentialing@physionet.org (https://physionet.org/about/faqs/).
- Contact: the FAQ page lists support routes; do not resubmit duplicates.

---

## 3. Download commands (Windows)

PhysioNet's standard Files-section command for a project is `wget -r -N -c -np <files URL>` (verified verbatim on an open project: `wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/`, https://physionet.org/content/mitdb/1.0.0/). For credentialed projects the same template adds `--user <username> --ask-password`; the exact line is printed on each project page after you sign the DUA **[template UNVERIFIED for these two projects until logged in — copy the line from the page]**.

The two `/files/` roots are confirmed to exist and to be auth-gated (HTTP 403 when anonymous, checked 2026-09-04):
- https://physionet.org/files/brazilian-ophthalmological/1.0.2/
- https://physionet.org/files/mbrset/1.0/

### 3a. Install wget (not present on this PC)

```powershell
winget install -e --id JernejSimoncic.Wget
# package id and 1.21.4 version confirmed at https://winstall.app/apps/JernejSimoncic.Wget
# alternative: choco install wget
```
Open a new terminal afterwards so PATH refreshes. Verify with `wget --version`.

### 3b. wget (recommended; resumable, mirrors folder tree)

```powershell
# run from the project root; data\raw must be in .gitignore
New-Item -ItemType Directory -Force data\raw | Out-Null
Set-Location data\raw

wget -r -N -c -np --user <physionet_username> --ask-password https://physionet.org/files/brazilian-ophthalmological/1.0.2/
wget -r -N -c -np --user <physionet_username> --ask-password https://physionet.org/files/mbrset/1.0/
```
Flags: `-r` recursive, `-N` timestamping (skip unchanged), `-c` continue partial files, `-np` no parent. Output lands in `data\raw\physionet.org\files\<slug>\<version>\`. Re-running the same command resumes after an interruption. Add `--cut-dirs=2 -nH` if you want `data\raw\<slug>\<version>\` instead.

Never put the password on the command line (`--password=`) — it ends up in shell history. `--ask-password` prompts interactively.

### 3c. Browser ZIP

PhysioNet offers a "Download the ZIP file (size)" button in the Files section of projects (seen on https://physionet.org/content/mitdb/1.0.0/). Whether it is offered for these two projects is **[UNVERIFIED]** until after DUA signing; large image projects sometimes omit it. If present, it is the simplest option but is not resumable — prefer wget for anything above ~5 GB.

### 3d. curl.exe (already installed at C:\Windows\system32\curl.exe)

curl does not mirror directories. Use it for single files, e.g. the label CSV and checksums, after the DUA:
```powershell
curl.exe -u <physionet_username> -o labels.csv https://physionet.org/files/brazilian-ophthalmological/1.0.2/labels.csv
curl.exe -u <physionet_username> -o SHA256SUMS.txt https://physionet.org/files/brazilian-ophthalmological/1.0.2/SHA256SUMS.txt
```
(`-u user` without `:password` prompts for the password.)

### 3e. PowerShell Invoke-WebRequest (no install needed; single files)

```powershell
$cred = Get-Credential   # enter PhysioNet username/password in the dialog; nothing is written to disk
Invoke-WebRequest -Uri "https://physionet.org/files/mbrset/1.0/labels_mbrset.csv" -Credential $cred -OutFile labels_mbrset.csv
```

### 3f. Python requests (basic auth; for scripted per-file pulls driven by SHA256SUMS.txt)

```python
# scripts/fetch_physionet.py  -- credentials from environment only; never hard-code, never commit
import os, sys, pathlib, requests
from getpass import getpass

BASE = "https://physionet.org/files/"
user = os.environ.get("PHYSIONET_USER") or input("PhysioNet username: ")
pw   = os.environ.get("PHYSIONET_PASS") or getpass("PhysioNet password: ")
slug_version = sys.argv[1]            # e.g. brazilian-ophthalmological/1.0.2
dest = pathlib.Path("data/raw") / slug_version
dest.mkdir(parents=True, exist_ok=True)

s = requests.Session(); s.auth = (user, pw)
sums = s.get(f"{BASE}{slug_version}/SHA256SUMS.txt"); sums.raise_for_status()
(dest / "SHA256SUMS.txt").write_text(sums.text)
for line in sums.text.splitlines():
    if not line.strip(): continue
    digest, rel = line.split(maxsplit=1); rel = rel.lstrip("*")
    out = dest / rel
    if out.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True)
    with s.get(f"{BASE}{slug_version}/{rel}", stream=True) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20): f.write(chunk)
```
Set `PHYSIONET_USER`/`PHYSIONET_PASS` only in the current shell session (`$env:PHYSIONET_PASS = ...`), never in `.env` files inside the repo, never in CI secrets shared with non-credentialed people.

### 3g. Download-time estimate at ~50 Mbit/s

50 Mbit/s ≈ 6.25 MB/s ≈ 22 GB/hour sustained.
- mBRSET (est. 3–6 GB) ≈ **8–16 min**
- BRSET (est. 15–45 GB) ≈ **40 min–2 h**
Replace with real figures once the "Total uncompressed size" line is visible. PhysioNet's FAQ notes that "data exceeding several tens of GB requires considerable download time" and suggests AWS/GCP mirrors where offered (https://physionet.org/about/faqs/); cloud mirrors for these two projects are **[UNVERIFIED]**.

---

## 4. Compliance (DUA 1.5.0 + PhysioNet LLM guidance)

Sources: DUA text https://physionet.org/content/brazilian-ophthalmological/view-dua/1.0.2/ and https://physionet.org/content/mbrset/view-dua/1.0/ ; licence clauses https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/ ; LLM guidance https://physionet.org/news/post/llm-responsible-use/ (Sept 24, 2025); FAQ https://physionet.org/about/faqs/ .

**Allowed**
- Local processing on your own machine(s) for "lawful use in scientific research" (DUA cl. 6).
- LLM-assisted **code** writing (the code never contains data). PhysioNet: "Local LLM models can be used without restrictions" (FAQ). Locally deployed LLMs are the recommended approach (LLM post).
- Cloud/API LLM services only if you can verify zero data retention, no training on the data and no human review; PhysioNet warns that "zero data retention" claims may be insufficient and that "If a service's data handling practices are unclear or cannot be fully verified, do not use the service." (LLM post). Practical rule for this project: **no image, no CSV row, no derived per-patient record goes into Claude, ChatGPT, Colab, Kaggle, Drive, GitHub or any hosted service.** Aggregate statistics and model weights trained on the data are not restricted by the DUA text, but keep saliency maps / example images out of shared documents unless they are the authors' own published figures.

**Not allowed**
- Sharing access with anyone (DUA cl. 3: "I will not share access to PhysioNet restricted data with anyone else"). FAQ: "each user must obtain individual access rights. Sharing data within teams or classes is not permitted." → **Aryan is not credentialed: he may not receive the images, the CSVs, a shared drive link, a database dump, or a screen-share of individual records.** He can work on code, on synthetic/public data (e.g. APTOS/EyePACS on Kaggle), and on aggregate results. If he needs the data, he must complete CITI + credentialing + sign both DUAs himself.
- Any re-identification attempt (cl. 1) and any publication that could disclose identity (cl. 2).
- Sending data "through APIs or using it on online platforms" (LLM post quoting the DUA).
- Uploading to public repos: the code repo must `.gitignore` `data/`, any `*.csv` derived from labels, notebooks with data outputs, and `SHA256SUMS.txt` copies are fine.

**Security / retention**
- Keep "physical and electronic security" (cl. 4): encrypted disk (BitLocker), no copies on shared/removable media, no cloud sync folders (OneDrive, Google Drive) — check that `Downloads` and the project folder are not OneDrive-synced.
- Report any suspected identifying information to PHI-report@physionet.org (cl. 5).
- The DUA 1.5.0 does **not** state a retention or deletion deadline; obligations "continue after termination" (cl. 10). Delete raw data when the project ends as good practice, and document that in the paper's data statement.

**Publication requirements**
- Cite the dataset record, the original paper, and PhysioNet (section 1 citations). The project pages request the 2026 Nature Health PhysioNet citation.
- Code release: "If I openly disseminate my results, I will also contribute the code used to produce those results to a repository that is open to the research community." (cl. 9) — plan a public GitHub release of code (without data) at submission time.
- Data-availability statement wording: state that BRSET v1.0.2 and mBRSET v1.0 are available on PhysioNet under the Credentialed Health Data License 1.5.0 to credentialed users, and that no data are redistributed.
- Training must stay current (cl. 7): keep the CITI report; CITI completions are commonly valid for 3 years **[UNVERIFIED on physionet.org]**.

---

## 5. Post-download ingest checklist (A5)

### 5a. Verify integrity with SHA256SUMS.txt

PhysioNet ships a `SHA256SUMS.txt` in every project root (confirmed on https://physionet.org/files/mitdb/1.0.0/ ; format is `<sha256>  <relative path>` per line). Expect `SHA256SUMS.txt` in both `brazilian-ophthalmological/1.0.2/` and `mbrset/1.0/` **[UNVERIFIED until downloaded]**.

PowerShell (no extra tools):
```powershell
Set-Location "data\raw\physionet.org\files\brazilian-ophthalmological\1.0.2"
$bad = 0; $n = 0
Get-Content SHA256SUMS.txt | ForEach-Object {
  if ($_ -match '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
    $n++; $want = $matches[1].ToLower(); $rel = $matches[2]
    if (-not (Test-Path $rel)) { "MISSING $rel"; $bad++; return }
    $got = (Get-FileHash -Algorithm SHA256 $rel).Hash.ToLower()
    if ($got -ne $want) { "MISMATCH $rel"; $bad++ }
  }
}
"checked $n files, $bad problems"
```
Git Bash: `sha256sum -c SHA256SUMS.txt --quiet` (`sha256sum` ships with Git for Windows). Repeat for `mbrset\1.0`.

### 5b. Expected counts

| Dataset | Images | Patients | Source |
|---|---|---|---|
| BRSET v1.0.2 | 16,266 files in `fundus_photos/`, 16,266 rows in `labels.csv` | 8,524 unique `patient_id` | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ |
| mBRSET v1.0 | 5,164 `.jpg` files, 5,164 rows in `labels_mbrset.csv` | 1,291 unique `patient` | https://physionet.org/content/mbrset/1.0/ |

```python
import pandas as pd, pathlib
b = pd.read_csv("data/raw/.../brazilian-ophthalmological/1.0.2/labels.csv")
assert len(b) == 16266 and b.patient_id.nunique() == 8524, (len(b), b.patient_id.nunique())
imgs = {p.name for p in pathlib.Path("data/raw/.../fundus_photos").iterdir()}
assert set(b.image_id.astype(str)) <= imgs or set(b.image_id.astype(str) + ".jpg") <= imgs  # check id vs filename form
m = pd.read_csv("data/raw/.../mbrset/1.0/labels_mbrset.csv")
assert len(m) == 5164 and m.patient.nunique() == 1291, (len(m), m.patient.nunique())
```
Also record: dtype/coding of `final_edema` and `laterality` (strings), `exam_eye` (1/2), and how many BRSET rows have `diabetes` set — DR grades are only meaningful within the diabetic subset for prevalence reporting.

### 5c. Label schema to standardise across both sets

| Target | BRSET column | mBRSET column | Unified |
|---|---|---|---|
| DR grade (ICDR 0–4) | `DR_ICDR` | `final_icdr` | `dr_icdr` int 0–4 |
| DR binary | `diabetic_retinopathy` (0/1) or `DR_ICDR>0` | `final_icdr>0` | `dr_any` |
| Referable DR (common in literature: ICDR ≥ 2 or DME) | derive | derive | `dr_referable` |
| DME | `macular_edema` (0/1) | `final_edema` (yes/no → 0/1) | `dme` |
| Patient id | `patient_id` | `patient` | `pid` (prefix `b_` / `m_`) |
| Eye | `exam_eye` 1=R, 2=L | `laterality` | `eye` in {R, L} |
| Quality gate | `focus`, `illumination`, `image_field`, `artifacts` (2 = inadequate) | `final_quality`, `final_artifacts` | `quality_ok` |
| Diabetes duration | `diabetes_time` | `dm_time` | `dm_years` |
| Insulin | `insulin_use` | `insulin` (+ `insulin_time`) | `insulin` |
| Neuropathy | — | `neuropathy` (self-reported) | `neuropathy` (mBRSET only) |
| HbA1c | — | — | not available in either dataset |
| Device | `camera` | constant "Phelcom Eyer" | `device` |

Note BRSET ICDR grade 4 is "Proliferative diabetic retinopathy and post-laser status" (https://physionet.org/content/brazilian-ophthalmological/), i.e. treated eyes are folded into grade 4.

### 5d. Patient-level splits (mandatory)

Both datasets contain both eyes (and sometimes repeat visits) of the same patient; the mBRSET authors used a "grouped stratification strategy to ensure a representative distribution of classes and no patient data leakage" with 70/10/20 splits (https://pmc.ncbi.nlm.nih.gov/articles/PMC11846882/), and BRSET's paper used stratified 70/30 (https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000454). Neither project ships an official split file **[UNVERIFIED — check the file table after download]**, so create and freeze one:

```python
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for k, (_, te) in enumerate(sgkf.split(df, y=df["dr_icdr"], groups=df["pid"])):
    df.loc[df.index[te], "fold"] = k
assert df.groupby("pid")["fold"].nunique().max() == 1   # every patient in exactly one fold
df[["image_file", "pid", "fold"]].to_csv("data/splits/brset_folds_seed42.csv", index=False)
```
Keep BRSET and mBRSET splits separate (different devices; mBRSET is the natural external/handheld test set). Stratify on `dr_icdr`; if grade-3 counts per fold are too small (mBRSET has only ~1.7 % grade 3), stratify on a collapsed label {0, 1, 2, 3+4} instead. The split CSV contains only file names, patient ids and fold numbers — still treat it as derived restricted data and keep it out of the public repo.

---

## 6. Things I could not verify from public pages (fill in after login)

1. Exact "Total uncompressed size" for each project and whether a ZIP button is offered.
2. The exact wget line as printed (expected `wget -r -N -c -np --user <you> --ask-password https://physionet.org/files/<slug>/<version>/`).
3. Presence and names of `SHA256SUMS.txt`, `LICENSE.txt`, any `RECORDS`/split files.
4. mBRSET image folder name and the string codings of `final_edema` / `laterality`.
5. Whether `settings/training` shows the CITI row as "accepted" for your account (only you can see it).
6. Sign-DUA page control labels (checkbox/button wording).

Sources consulted (all fetched 2026-09-04): https://physionet.org/content/brazilian-ophthalmological/ ; https://physionet.org/content/brazilian-ophthalmological/1.0.2/ ; https://physionet.org/content/brazilian-ophthalmological/1.0.1/ ; https://physionet.org/content/brazilian-ophthalmological/view-dua/1.0.2/ ; https://physionet.org/content/brazilian-ophthalmological/view-required-training/1.0.2/ ; https://physionet.org/content/mbrset/ ; https://physionet.org/content/mbrset/1.0/ ; https://physionet.org/content/mbrset/view-dua/1.0/ ; https://physionet.org/files/brazilian-ophthalmological/1.0.2/ (403) ; https://physionet.org/files/mbrset/1.0/ (403) ; https://physionet.org/about/citi-course/ ; https://physionet.org/settings/training/ (login redirect) ; https://physionet.org/news/post/llm-responsible-use/ ; https://physionet.org/news/post/395/ ; https://physionet.org/about/faqs/ ; https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/ ; https://physionet.org/content/mitdb/1.0.0/ ; https://physionet.org/files/mitdb/1.0.0/ ; https://physionet.org/content/embedding-brset-mbrset/1.0.0/ ; https://github.com/luisnakayama/BRSET ; https://github.com/luisnakayama/mBRSET ; https://github.com/MIT-LCP/mimic-code/discussions/1640 ; https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000454 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11239107/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11846882/ ; https://casrai.org/guides/physionet-credentialed-access-restricted-data ; https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/ ; https://winstall.app/apps/JernejSimoncic.Wget .

## APTOS 2019 (second external fundus test) — added 2026-09-07

**Where the Kaggle token goes.** Never into the project folder, never into a `.env` (the
project is going public on publication, DUA cl. 9). The Kaggle CLI 2.x reads it from ONE of:

1. File (preferred, persistent): `%USERPROFILE%\.kaggle\access_token` containing only the token
   string, no quotes, no newline. Create it from PowerShell (paste the token in place of the X's):

   ```powershell
   Set-Content -Path "$env:USERPROFILE\.kaggle\access_token" -Value "XXXXXXXXXXXXXXXX" -NoNewline -Encoding ascii
   ```

2. Environment variable for one session only: `$env:KAGGLE_API_TOKEN = "XXXX"` (PowerShell).

Verify: `python -m kaggle competitions list` (the `kaggle.exe` shim is in
`%APPDATA%\Python\Python314\Scripts`, which is not on PATH, so use `python -m kaggle`).

**Two sources, in order of preference for the paper:**

| | Official competition | Community mirror `mariaherrerot/aptos2019` |
|---|---|---|
| Needs token | yes | no (public dataset, downloads anonymously) |
| Needs one click on kaggle.com | yes: open https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules and accept (Late Submission / "I Understand and Accept") | no |
| Licence statement | competition rules: non-commercial academic research permitted | "Unknown" on the dataset card |
| Files | `train.csv` (id_code, diagnosis) + `train_images/*.png`, 3,662 labelled | same images, re-split into train/val/test CSVs |
| Command | `python -m kaggle competitions download -c aptos2019-blindness-detection -p "<root>"` | `python -m kaggle datasets download -d mariaherrerot/aptos2019 -p "<root>"` |

`<root>` = `results\data\raw\open\aptos2019`. `results/src/eval_aptos.py` does all of this
itself: it tries the competition when a token exists, falls back to the mirror, unzips, writes
`SOURCE.txt` recording which one was used, and evaluates the five DeepDRiD fold models with the
DeepDRiD thresholds unchanged. Output: `results/out/fundus_aptos.json`, `fundus_aptos_pred.npz`.

Citation either way: Asia Pacific Tele-Ophthalmology Society, "APTOS 2019 Blindness Detection",
Kaggle, 2019, https://www.kaggle.com/competitions/aptos2019-blindness-detection. If the mirror
was used, the Methods say so and state that it is a re-split of the competition training set.
Images are never redistributed.
