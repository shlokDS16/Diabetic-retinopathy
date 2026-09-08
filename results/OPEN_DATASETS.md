# Open substitute datasets for the DR / neuropathy multimodal paper

Compiled 2026-09-07 (presentation 2026-09-09). Context: BRSET v1.0.2 and mBRSET v1.0 on PhysioNet are
credentialed (CITI training + PhysioNet Credentialed Health Data License 1.5.0; re-verified today at
https://physionet.org/content/brazilian-ophthalmological/1.0.1/), so everything below is what can be trained on
without a pending review.

Verification legend
- VERIFIED: page fetched today and the stated facts come from it (or from its JSON API / data descriptor).
- PARTIAL: page is live (title returned) but Kaggle renders content client-side, so licence/columns come from
  secondary sources named inline.
- UNVERIFIED: could not fetch; stated from a secondary source or memory. Treat as a claim to re-check.

Method: every URL below was requested with WebFetch; Kaggle competition/dataset pages return only their title
(client-side rendering), figshare pages 403 but the figshare v2 API works, nature.com/springer redirect to a login
gate (used PMC / Semantic Scholar / arXiv instead), MDPI 403 (used HF/Zenodo/IEEE DataPort policy pages instead).

---

## A. Fundus datasets with DR grading

### A1. DeepDRiD (Shanghai) - VERIFIED
- URL: https://github.com/deepdrdoc/DeepDRiD (data files are in the repo tree; no account)
- Licence: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) - read from the repo LICENSE file.
- Size: 2,000 regular fundus images from 500 patients (4 per patient: macula- and disc-centred, both eyes) plus
  256 ultra-widefield images from 128 patients (counts from the Patterns paper via search snippet; the README
  itself does not state them - PARTIAL on counts).
- Grading: DR severity 0-4 (ICDR); image-quality labels (overall, artefact, clarity, field definition). No DME label.
- Metadata: patient id, eye/laterality, image field encoded in filenames/CSVs (structure "patient/eye/field";
  exact CSV column names not extracted - PARTIAL). No age/sex/HbA1c.
- Download: `git clone https://github.com/deepdrdoc/DeepDRiD` (repo splits into training/validation/evaluation).
- Account: none.
- Publication condition (CC BY-SA): attribute; derivative *datasets* must carry the same licence. Cite
  Liu R. et al., "DeepDRiD: Diabetic Retinopathy-Grading and Image Quality Estimation Challenge", Patterns 3, 100512 (2022),
  doi:10.1016/j.patter.2022.100512.
- Why first: only fully open set with real patient identifiers, so patient-level splits are defensible.

### A2. IDRiD (Nanded, India) - VERIFIED
- URL: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid (free IEEE account
  needed). No-account mirror with identical zips and explicit CC BY 4.0: https://zenodo.org/records/17219542
  (uploader states they are not the creator; files A/B/C zips, 1.0 GB).
- Licence: CC BY 4.0 (stated on Zenodo mirror, on the HF mirror amin-nejad/idrid-disease-grading and in the
  IEEE DataPort open-access policy: "content will be made available subject to the terms of the Creative Commons
  Attribution (CC-BY) License"). IEEE DataPort page itself does not print the licence name.
- Size: 516 images for grading (413 train / 103 test); 81 with pixel-level lesion masks; 4288x2848, Kowa VX-10a.
- Grading: DR 0-4 (ICDR) AND DME 0-2 per image. Graded by a retinal specialist.
- Metadata: none (no patient id, age, sex). Images are from an Indian clinic, which matches the VIT context.
- Download: Zenodo direct zips; or IEEE DataPort after login; HF: `huggingface-cli download amin-nejad/idrid-disease-grading --repo-type dataset`.
- Cite: Porwal P. et al., "Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy
  Screening Research", Data 3(3):25 (2018); and IEEE DataPort doi:10.21227/H25W98.
- Why second: only open set with an ordinal DME label alongside ICDR; Indian population; CC BY.

### A3. SUSTech-SYSU (Zhongshan Ophthalmic Centre) - VERIFIED
- URL: https://figshare.com/articles/dataset/The_SUSTech-SYSU_dataset_for_automated_exudate_detection_and_diabetic_retinopathy_grading/12570770
  (page 403s to bots; API https://api.figshare.com/v2/articles/12570770 confirms one 437.67 MB zip).
- Licence: CC BY 4.0 (figshare API `license.name`); metadata under CC0 per the Sci Data article.
- Size: 1,219 images (603 DR, 631 healthy per PMC7679367); Topcon TRC-50DX, 2880x2136.
- Grading: DR grade under three protocols - International (0-5, incl. laser-treated), AAO, Scottish. No separate
  DME label. Exudate masks; optic-disc box; fovea coordinates. Three ophthalmologists; ICC 0.91.
- Metadata: left/right eye label only; no patient id.
- Download: direct figshare file link (no account).
- Cite: Lin L. et al., Sci Data 7, 409 (2020), doi:10.1038/s41597-020-00755-0.

### A4. APTOS 2019 (Aravind Eye Hospital, India) - PARTIAL
- URL: https://www.kaggle.com/competitions/aptos2019-blindness-detection (live; content client-rendered).
- Licence: Kaggle competition rules ("Competition Use and Non-Commercial & Academic Research" per secondary
  summary; the clause text itself could not be fetched - UNVERIFIED verbatim). Research publication is the normal
  use; redistribution prohibited, so do not re-upload images.
- Size: 3,662 labelled train images (1,805/370/999/193/295 for grades 0-4); 1,928 test images unlabelled.
- Grading: 0-4 ICDR. Columns: id_code, diagnosis. No patient id, laterality, or metadata.
- Download: `kaggle competitions download -c aptos2019-blindness-detection` (Kaggle account + accept rules).
  Mirror (exists; licence not readable): `kaggle datasets download -d mariaherrerot/aptos2019`.
- Precedent: dozens of journal papers (see D).

### A5. Kaggle Diabetic Retinopathy Detection 2015 (EyePACS) - PARTIAL
- URL: https://www.kaggle.com/competitions/diabetic-retinopathy-detection (live). TFDS card VERIFIED:
  https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection (35,126 train / 10,906 val /
  42,670 test; label 0-4; manual Kaggle download required).
- Licence: Kaggle competition rules (accept required; research use customary; verbatim UNVERIFIED).
- Metadata: image names `<patient>_left/right.jpeg` give a patient id and laterality -> patient-level splits
  possible. No age/sex/HbA1c. Test labels were released after the competition (88,702 labelled total).
- Download: `kaggle competitions download -c diabetic-retinopathy-detection` (~89 GB original).
- Label noise is well documented (Voets 2019). Good for pretraining, weak as the primary reported set.

### A6. DDR (147 hospitals, China) - VERIFIED (repo) / PARTIAL (counts)
- URL: https://github.com/nkicsl/DDR-dataset ; files on Google Drive / Baidu.
- Licence: MIT (repo). Unusual for clinical images, but it is what the owners publish.
- Size: 13,673 images, 6 classes (ICDR 0-4 + ungradable); 757 with lesion boxes/masks (from Li 2019 via search).
- Metadata: none. Cite Li T. et al., Information Sciences 501:511-522 (2019).
- Kaggle mirror exists (title only): `kaggle datasets download -d mariaherrerot/ddrdataset`.

### A7. ODIR-5K - VERIFIED (challenge page) / licence UNCLEAR
- URL: https://odir2019.grand-challenge.org/dataset/ (download page returned 403 to the fetcher).
- Size: 5,000 patients, left+right images (~10k); 8 patient-level labels N/D/G/C/A/H/M/O ("D" = diabetic
  retinopathy present, no grade); age, sex (sex per Kaggle mirror descriptions; the challenge page names age only),
  per-eye diagnostic keywords; patient id.
- Licence: none stated anywhere fetched; the Kaggle mirror (andrewmvd/ocular-disease-recognition-odir5k, live)
  is what most papers cite. Flag as "terms unclear" in the manuscript.
- Download: `kaggle datasets download -d andrewmvd/ocular-disease-recognition-odir5k`.

### A8. Messidor-2 + adjudicated grades - VERIFIED (ADCIS) / PARTIAL (Kaggle grades)
- URL: https://www.adcis.net/en/third-party/messidor2/ - 874 examinations, 1,748 images, paired per exam
  (patient-level split possible). Form with personal information required; "can be used, free of charge, for
  research and educational purposes. Copy, redistribution, and any unauthorized commercial use are prohibited."
  Publications must acknowledge LaTIM and the Messidor partners and cite Decenciere et al. 2014 and Abramoff et al. 2013.
- Official data has NO DR ground truth. Grades: https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades
  (live; adjudicated DR grade, DME, gradability per Krause et al. 2018 - contents UNVERIFIED today).
- Risk for a 2-day build: form turnaround is manual.

### A9. RFMiD - VERIFIED
- URL: https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid (IEEE login);
  HF mirror ctmedtech/RFMID states CC BY 4.0, 3,200 images, 46 labels incl. DR (binary, no grade).
- Cite Pachade S. et al., Data 6(2):14 (2021). No metadata. Useful only as an extra DR-vs-normal external test.

### A10. Request-form sets (not for a 2-day build)
- DRTiD (https://github.com/FDU-VTS/DRTiD, VERIFIED): 3,100 two-field images, DR grades, research-only,
  Google/Wenjuanxing form. Cite Hou et al., BIBM 2022.
- FGADR (https://csyizhou.github.io/FGADR/, VERIFIED): 1,842 seg + 1,000 grade images; signed Research Use
  Agreement by email; non-commercial; Grade-set still pending legal approval.
- Retinal-Lesions (https://github.com/WeiQijie/retinal-lesions, VERIFIED): 1,593 images, DR0-4 by 45
  ophthalmologists, 8 lesion classes; Google Form; licence not stated.
- e-ophtha (https://www.adcis.net/en/third-party/e-ophtha/, VERIFIED): lesion masks only (82 EX, 381 MA), no DR grade.
- MMAC 2023: myopic maculopathy, not DR - excluded.

### A11. Hugging Face mirrors - legitimacy check
- Legitimate (licence carried through): amin-nejad/idrid-disease-grading (CC BY 4.0), ctmedtech/RFMID (CC BY 4.0).
- Questionable: bumbledeep/eyepacs and youssefedweqd/Diabetic_Retinopathy_Detection re-label Kaggle EyePACS as
  "MIT" with no original terms - the competition rules forbid redistribution; do not cite these as the source.
  ctmedtech/EYEPACS at least states "usage restricted under Kaggle & EyePACS terms". sngsfydy/aptos: no licence,
  3 classes, 2,538 rows - not the original.
- Synthetic-metadata trap: macular/diabetic-retinopathy-grading-africa and the
  "diabetic-retinopathy-multimodal-progression-africa" cards are APTOS images with age/sex/duration that are
  generated (`synthetic = True`, Apache-2.0). Unusable as clinical metadata; useful only for pipeline testing.
- No HF mirror of BRSET, mBRSET, DeepDRiD, DDR, SUSTech-SYSU or ODIR was found in searches for
  "diabetic retinopathy", "APTOS", "IDRiD", "fundus".

---

## B. Non-retinal signals with diabetes / complication labels (open, no credentialing)

### B1. 24-h ECG + metabolic dataset, male T2DM inpatients (Southeast University, Nanjing) - VERIFIED
- URL: https://data.mendeley.com/datasets/9c47vwvtss/4 ; descriptor PMC10405204 (Front. Physiol. 2023).
- Licence: CC BY 4.0. Size: 60 subjects, 24-h Holter ECG at 250 Hz (.mat), RR-interval series by sleep stage.
- Clinical table: HbA1c, FBG, lipids, renal function, and 0/1 flags for diabetic nephropathy,
  retinopathy/cataract, peripheral neuropathy, CAD, atherosclerosis, carotid plaque. No duration; no controls.
- Download: direct from Mendeley (no account). Cite doi:10.17632/9c47vwvtss.4 and the Frontiers article.
- Fit: the only open set where HRV can be related to neuropathy AND retinopathy labels. Small n.

### B2. PPG-BP (Guilin People's Hospital) - VERIFIED
- URL: https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299 (API confirms files: PPG-BP Database.zip,
  Table 1.xlsx, sdata201820.pdf).
- Licence: CC0. Size: 219 subjects, 657 fingertip PPG segments (2.1 s, 1 kHz, 905 nm).
- Metadata (Table 1): sex, age, height, weight, SBP/DBP, heart rate, BMI, hypertension stage, Diabetes,
  cerebral infarction, cerebrovascular disease (field list from the arXiv 2308.01930 methods; "Diabetes" column
  used as label there). Diabetic count: paper kept 27 diabetic / 59 non-diabetic after exclusions; whole-table
  count UNVERIFIED (open Table 1.xlsx).
- Cite: Liang Y. et al., Sci Data 5, 180020 (2018), doi:10.1038/sdata.2018.20.

### B3. Plantar Thermogram Database for Diabetic Foot (INAOE, Mexico) - VERIFIED
- URL: https://ieee-dataport.org/open-access/plantar-thermogram-database-study-diabetic-foot-complications
  (free IEEE account). Licence: CC BY per IEEE DataPort open-access policy (not printed on the page).
- Size: 167 subjects (122 DM, 45 control), 334 foot thermograms + per-angiosome crops; RGB PNG plus raw
  temperature CSV; sex encoded in file names. No HbA1c/duration.
- Cite: Hernandez-Contreras D. et al., IEEE Access 7:161296 (2019); DataPort doi:10.21227/tm4t-9n15.
- Caveat: plantar, not periorbital; only usable as a "thermal signature of diabetes" proxy branch.

### B4. PhysioNet open diabetes cohorts (Novak lab) - VERIFIED
- Cerebral Vasoregulation in Diabetes: https://physionet.org/content/cerebral-vasoreg-diabetes/1.0.0/
  CC BY 4.0; 37 DM / 49 controls; ECG, continuous BP, respiration, TCD; tilt/Valsalva/sit-to-stand; summary
  table includes retinopathy level and labs.
- Cerebral Perfusion and Cognitive Decline in T2DM: https://physionet.org/content/cerebral-perfusion-diabetes/1.0.0/
  CC BY 4.0; 70 DM / 70 controls; 24-h ECG, TCD, MRI, insole pressure; HbA1c and retinopathy level.
- Cerebromicrovascular Disease in Elderly with Diabetes: listed open on physionet.org/content/?topic=diabetes
  (page URL 404 at the slug I tried - UNVERIFIED).
- Fit: ECG/BP with diabetes and retinopathy variables in one cohort; elderly; heavy files.

### B5. D1NAMO (HES-SO) - VERIFIED
- URL: https://zenodo.org/records/1421616 ; CC BY-SA 4.0; 10.2 GB; 20 healthy + 9 T1D; Zephyr BioHarness ECG,
  breathing, accelerometer, CGM, food photos. Wearable-grade ECG but n=9 diabetic.

### B6. Dryad HRV in T2DM without neuropathy - VERIFIED
- https://datadryad.org/dataset/doi:10.5061/dryad.5d8jv ; Dryad = CC0; 34 T2DM / 34 controls; derived spectral
  indices only (Rest.xlsx, Stand.xlsx), no raw signals. PLOS ONE 2016.

### B7. Not openly available
- Pupillometry for DAN: "Dynamic Pupillary Dataset for Autonomic Neuropathy in Type 1 Diabetes" (IEEE DataPort,
  doi:10.21227/gknz-g927, 49 T1D patients, 90 videos) requires an IEEE DataPort *subscription* - VERIFIED, not open.
- Facial/periorbital thermography of diabetics: no downloadable dataset found (only papers).
- Bioimpedance with diabetes labels: none found.
- BIG IDEAs Lab wearable + CGM (PhysioNet, open) is normoglycemic only - not useful for labels.

---

## C. Tabular clinical datasets with DR / neuropathy outcomes

| # | Dataset | Provenance | Licence | n | DR/neuropathy outcome | Systemic vars | Verdict |
|---|---|---|---|---|---|---|---|
| C1 | Dryad doi:10.5061/dryad.6kg1sd7 (Zhuang et al., BMJ Open 2019) VERIFIED | Guangdong Provincial People's Hospital, single centre | CC0 | 413 T2DM | DR stage, DME | duration, LDL, eGFR, UACR (HbA1c not confirmed in abstract) | Q1-suitable; peer-reviewed source |
| C2 | Mendeley doi:10.17632/rr4rzzrjfc.2 "ML for Prediction of Glycemic Control in DM" VERIFIED | Akdeniz Univ.; Istanbul e-Nabiz EHR; ethics approval stated | CC BY 4.0 | not stated on page | retinopathy, neuropathy, nephropathy flags | HbA1c at dx and 1-yr change, insulin types/doses, lipids, age, sex | Q1-suitable with the caveat that the companion paper must be located and cited; check n before use |
| C3 | Mendeley 9c47vwvtss clinical table (see B1) VERIFIED | Southeast Univ. hospital | CC BY 4.0 | 60 | retinopathy, neuropathy flags | HbA1c, FBG, lipids | small but clean |
| C4 | UCI Diabetic Retinopathy Debrecen VERIFIED https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen | Antal & Hajdu 2014, features from Messidor images | CC BY 4.0 | 1,151 | binary DR | none (image-derived features only) | fine for a sanity baseline, not for systemic modelling |
| C5 | Kaggle "diabetes complications" style sets, Mendeley "Type-2 Diabetes" (Pabna), Mendeley "Diabetes Dataset" (Iraqi, HbA1c) | mixed; Iraqi set has HbA1c but no DR label; Pabna set has no DR label | various | - | none/unclear | HbA1c | unsuitable for Q1 as outcome data |
| C6 | Mendeley gmkngwww2p "Diabetic Peripheral Neuropathy" | RCT of a physiotherapy intervention (84 pts) | ? | 84 | DPN (all cases) | - | wrong design; skip |
| C7 | NHANES | already in use | public domain | - | retinopathy (self-report/questionnaire cycles), neuropathy (monofilament, 1999-2004) | HbA1c, duration, insulin | keep |

Dryad searches "diabetic retinopathy HbA1c" (0 hits) and "retinopathy diabetes" (16 hits, C1 the only
individual-level clinical file) were run through the Dryad v2 API today.

---

## D. Precedent and "good enough" numbers

Fundus (top 3 from A)
- DeepDRiD: Liu et al., Patterns 2022 - challenge sub-task 1 (DR grading, regular fundus) quadratic weighted
  kappa 0.9033-0.9303 across the nine finalists. A QWK of about 0.88-0.90 on a patient-level split is a credible
  student result; report also referable-DR AUC.
- IDRiD: Porwal et al., Medical Image Analysis 59:101561 (2020) - joint DR+DME accuracy on the 103-image test set
  was ~0.63 for the winning team; CANet (Li et al., IEEE TMI 2020) reports 65.1% joint accuracy, DR accuracy ~0.65
  and DME accuracy ~0.81 range. Do not expect high numbers on 103 images; report bootstrap CIs.
- APTOS 2019: Kaggle private leaderboard top QWK ~0.936 (UNVERIFIED, from memory); journal papers report QWK
  0.92 (EfficientNet-B0 + attention gates, arXiv 2604.17341), 0.93 (dual-branch EfficientNetB0/ResNet50,
  Biomed. Signal Process. Control 2024, doi:10.1016/j.bspc.2024.106224 - DOI UNVERIFIED), and a Sci Rep 2025
  grading paper (doi:10.1038/s41598-025-87171-9). Claims above 0.95 on APTOS usually come from random image-level
  splits; treat 0.90-0.93 as the honest band.
- EyePACS/Messidor-2 anchors: Gulshan et al., JAMA 2016 - AUC 0.991 (EyePACS-1) / 0.990 (Messidor-2) for
  referable DR with multi-grader labels; Voets et al., PLOS ONE 2019 - reproduction on *public* Kaggle labels
  gave AUC 0.951 (Kaggle test) and 0.853 (Messidor-2). The 0.85-0.95 gap is what public single-grader labels cost.

Signals (top 2 from B)
- PPG-BP: Liang et al., Sci Data 2018 (descriptor). Diabetes detection on it: Hettiarachchi & Chitraranjan,
  AIME 2019 (Springer LNAI) AUC 0.79; Costa Prado et al., arXiv 2308.01930 (2023) LR AUC 0.79 +/- 0.15 on
  86 subjects (59 non-diabetic, 27 diabetic) using sex, age, height, weight, HR, BMI + 104 morphology features.
  Field ceiling for PPG-only diabetes: Avram et al., Nature Medicine 2020, AUC 0.766 on 53,870 smartphone users.
  So an AUC of 0.75-0.80 with wide CIs is the realistic target; anything higher on 219 subjects is overfitting.
- Plantar thermograms: Hernandez-Contreras et al., IEEE Access 2019 (descriptor); Cruz-Vega et al., Sensors
  2020 (MobileNetV2/ShuffleNet, DM vs control); Diagnostics 2023 (PMC10453276) custom CNN AUC 0.976 image-level
  after augmentation to 1,000 images - subject-level leakage is likely, so quote it with that caveat.
- ECG-T2DM (B1): only the Frontiers in Physiology 2023 descriptor so far; no published complication-prediction
  benchmark on it, which is a novelty angle for the paper.

---

## E. Recommended combination for the 2-day build

Fundus branch (DR ordinal + DME)
1. Train on DeepDRiD regular-fundus images with GroupKFold by patient id (500 patients; ICDR 0-4). CC BY-SA 4.0.
2. External test on IDRiD (516 images, ICDR 0-4 + DME 0-2, Indian) - this is where the DME head gets its only
   real label, so train the DME head on IDRiD's 413 train images and report on its 103 test images with CIs.
3. Second external test on APTOS 2019 (Indian, 3,662, image-level only) for the DR head. Do not redistribute.
4. Optional pretraining on Kaggle 2015 if GPU time allows (patient-level split by filename prefix).
5. Skip Messidor-2 for this deadline (manual form); mention as future validation.

Non-retinal branches
- PPG branch: PPG-BP (CC0) diabetes flag; subject-level CV; 27-40 positives - report AUC with bootstrap CI.
- HRV/autonomic branch: Mendeley 24-h ECG T2DM (CC BY 4.0) - HRV features -> peripheral-neuropathy flag and
  retinopathy flag within diabetics (n=60, leave-one-subject-out). Optionally add PhysioNet Cerebral Perfusion
  (CC BY 4.0, 70 DM / 70 controls, 24-h ECG, HbA1c, retinopathy level) for a diabetic-vs-control HRV model.
- Thermal branch: Plantar thermogram database (CC BY) DM vs control, as a proxy for the thermal sensor; state
  explicitly that periorbital thermal data do not exist openly and the branch is a feasibility surrogate.
- Tabular/systemic branch: Dryad 6kg1sd7 (413 T2DM, DR/DME + duration + renal) and NHANES.

What is honestly NOT available openly
- No open dataset pairs fundus images with PPG/ECG/thermal signals from the same patients. BRSET/mBRSET are the
  closest (fundus + duration + insulin + self-reported neuropathy) and are credentialed. Therefore the multimodal
  fusion can only be validated as late fusion of independently trained branches, or on simulated co-registered
  cohorts; say so in Limitations.
- No open periorbital thermography, bioimpedance, or open pupillometry with neuropathy labels.
- No open fundus set with HbA1c per image.

Licence / attribution text to put in the manuscript
- IDRiD (CC BY 4.0): "IDRiD images and grades: Porwal et al. (2018), IEEE DataPort doi:10.21227/H25W98, CC BY 4.0."
- SUSTech-SYSU (CC BY 4.0): "Lin et al. (2020), figshare doi:10.6084/m9.figshare.12570770.v1, CC BY 4.0."
- DeepDRiD (CC BY-SA 4.0): "Liu et al. (2022), https://github.com/deepdrdoc/DeepDRiD, CC BY-SA 4.0." Any
  re-released annotation derived from it must also be CC BY-SA 4.0.
- PPG-BP (CC0): no legal obligation; cite Liang et al. (2018) and figshare doi:10.6084/m9.figshare.5459299.v5.
- Mendeley ECG-T2DM (CC BY 4.0): cite doi:10.17632/9c47vwvtss.4 and Front. Physiol. 2023.
- Plantar thermograms (CC BY via IEEE DataPort policy): cite Hernandez-Contreras et al. (2019), doi:10.21227/tm4t-9n15.
- PhysioNet Novak-lab sets (CC BY 4.0): cite the PhysioNet DOI (10.13026/m40k-4758; 10.13026/rbeh-9r20) and
  Goldberger et al. (2000) as PhysioNet requires.
- APTOS 2019 / Kaggle 2015: "used under the Kaggle competition rules for non-commercial academic research;
  images not redistributed." Cite the competition URL and, for EyePACS, Cuadros & Bresnick (2009).
- Messidor-2 (if used later): acknowledge the LaTIM laboratory and the Messidor program partners and cite
  Decenciere et al. (2014) and Abramoff et al. (2013); grades from Krause et al. (2018).

---

## Ranked table (suitability for a Q1 paper, this deadline)

| Rank | Dataset | Type | Licence | n (images / subjects) | Labels | Patient id | Metadata | Access effort | Verification |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DeepDRiD | fundus | CC BY-SA 4.0 | 2,000 / 500 | ICDR 0-4, quality | yes | laterality, field | git clone | VERIFIED |
| 2 | IDRiD | fundus | CC BY 4.0 | 516 / - | ICDR 0-4, DME 0-2, lesions | no | none | Zenodo direct / IEEE login | VERIFIED |
| 3 | SUSTech-SYSU | fundus | CC BY 4.0 | 1,219 / - | DR (3 protocols), exudates | no | laterality | figshare direct | VERIFIED |
| 4 | Mendeley 24-h ECG T2DM | ECG + clinical | CC BY 4.0 | - / 60 | neuropathy, retinopathy flags, HbA1c | yes | full clinical table | direct | VERIFIED |
| 5 | PPG-BP | PPG | CC0 | 657 seg / 219 | diabetes, hypertension | yes | age, sex, BMI, BP, HR | figshare direct | VERIFIED |
| 6 | Dryad Zhuang 2019 | tabular | CC0 | - / 413 | DR stage, DME | yes | duration, eGFR, UACR, LDL | direct | VERIFIED |
| 7 | APTOS 2019 | fundus | Kaggle rules (non-commercial research) | 3,662 / - | ICDR 0-4 | no | none | Kaggle API + rules | PARTIAL |
| 8 | PhysioNet Cerebral Perfusion in T2DM | ECG/TCD/MRI + clinical | CC BY 4.0 | - / 140 | diabetes, retinopathy level, HbA1c | yes | rich | direct, 13 GB | VERIFIED |
| 9 | Plantar Thermogram DB | thermal | CC BY (DataPort policy) | 334 / 167 | diabetes | yes | sex | IEEE login | VERIFIED |
| 10 | Kaggle EyePACS 2015 | fundus | Kaggle rules | 88,702 / ~44k | ICDR 0-4 | yes (filename) | laterality | Kaggle API, 89 GB | PARTIAL |
| 11 | DDR | fundus | MIT | 13,673 / - | ICDR 0-4 + ungradable | no | none | Google Drive | PARTIAL |
| 12 | Mendeley rr4rzzrjfc (Turkey EHR) | tabular | CC BY 4.0 | ? | retinopathy, neuropathy, nephropathy | yes | HbA1c, insulin | direct | VERIFIED (n unknown) |
| 13 | ODIR-5K | fundus | unstated | ~10k / 5,000 | DR present (no grade) + 7 others | yes | age, sex, keywords | Kaggle | licence UNCLEAR |
| 14 | Messidor-2 + google-brain grades | fundus | research-only, no redistribution | 1,748 / 874 exams | adjudicated DR, DME | exam-level | none | form + Kaggle | VERIFIED / PARTIAL |
| 15 | RFMiD | fundus | CC BY 4.0 | 3,200 / - | DR binary among 46 | no | none | IEEE login / HF | VERIFIED |
| 16 | D1NAMO | wearable ECG | CC BY-SA 4.0 | - / 29 (9 T1D) | T1D, glucose | yes | food, activity | Zenodo | VERIFIED |
| 17 | PhysioNet Cerebral Vasoregulation | ECG/BP/TCD | CC BY 4.0 | - / 86 | diabetes, autonomic tests, retinopathy level | yes | labs | direct | VERIFIED |
| 18 | Dryad HRV T2DM (5d8jv) | derived HRV | CC0 | - / 68 | T2DM vs control | yes | none | direct | VERIFIED |
| 19 | UCI Debrecen | tabular features | CC BY 4.0 | 1,151 | DR binary | no | none | direct | VERIFIED |
| 20 | DRTiD / FGADR / Retinal-Lesions | fundus | research-only agreements | 3,100 / 2,842 / 1,593 | DR grades (+lesions) | partial | none | request form | VERIFIED (pages) |
| - | e-ophtha, MMAC, HF "DR-Africa" synthetic cards, HF MIT-relabelled EyePACS copies, pupillometry (subscription) | - | - | - | - | - | - | - | excluded, reasons in A10/A11/B7 |

Source URLs used (all fetched 2026-09-07 unless marked): listed inline above. Kaggle pages return titles only;
their licence text is from the Kaggle rules pages as summarised by third parties and should be re-read in a
browser before submission.
