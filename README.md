# NRDI: a physics-verified periorbital wearable with multimodal AI for joint diabetic retinopathy and neuropathy screening

**Short description (for the GitHub "About" box):** Spectacle-frame wearable design for joint diabetic retinopathy and neuropathy screening, verified by a coupled thermal, optical and electrical digital twin, with fundus, EHR and physiological AI branches trained and tested on open data only.

**Status.** Design and modeling study. Every result in this repository comes from physics simulation or from public datasets. No prototype has been tested on people, no human data were generated, and nothing here is a clinical claim.

---

## 1. What the project does

Diabetic retinopathy (DR) and diabetic neuropathy (DN) are screened separately, in clinics, and rarely together. This project designs a spectacle-frame wearable that reads, at the periorbital region, ocular surface temperature (thermopile), pupil dynamics (camera with a 940 nm illuminator), periorbital bioimpedance (four dry electrodes per eye) and temple photoplethysmography (PPG), and it tests two things before any device is built:

1. **Physics.** A digital twin of the periorbital region couples a Pennes bioheat model, GPU Monte Carlo photon transport, a complex-conductivity bioimpedance model and a Longtin-Milton pupil model on one 14-tissue finite-element anatomy (122,279 nodes, 705,228 tetrahedra). It is verified with manufactured solutions and grid convergence, its uncertainty is quantified with Sobol' indices over 640 solves, and it returns eight design verdicts to the hardware.
2. **AI branches.** A ConvNeXt-Tiny ordinal fundus grader trained on DeepDRiD (China) and tested unchanged on IDRiD and APTOS 2019 (India); a macular edema head; logistic and gradient-boosted models of retinopathy and neuropathy on 77,724 routine health records (Türkiye); heart-rate-variability and PPG models on open PhysioNet and PPG-BP cohorts.

## 2. Headline results (all recomputed live in `results/notebooks/NRDI_results.ipynb`)

| Branch | Result | 95 % CI |
|---|---|---|
| Fundus grader, DeepDRiD patient-grouped 5-fold OOF (1,589 images, 399 patients) | QWK 0.857, referable AUC 0.970, accuracy 68.5 % | 0.835 to 0.878; 0.959 to 0.979 |
| IDRiD external test (516 images), thresholds unchanged | QWK 0.729, referable AUC 0.952 | 0.690 to 0.764; 0.933 to 0.967 |
| APTOS 2019 external test (3,662 images), thresholds unchanged | QWK 0.845, referable AUC 0.979, accuracy 55.2 % (systematic +1 grade shift; re-fitted thresholds give QWK 0.878, reported, not used) | 0.836 to 0.853; 0.976 to 0.983 |
| DME head, IDRiD fixed test (103 images) | QWK 0.838, any-DME AUC 0.945 | 0.724 to 0.921; 0.898 to 0.981 |
| EHR retinopathy, boosted, 5-fold OOF (n = 77,724) | AUC 0.720, calibration slope 1.02 | 0.715 to 0.725 |
| EHR neuropathy, boosted, **treatment proxies excluded** | **AUC 0.710** (0.838 with proxies, sensitivity analysis only) | 0.705 to 0.715 |
| HRV, diabetes vs control (PhysioNet, n = 94) | LOO AUC 0.777 | 0.671 to 0.870 |
| HRV, retinopathy within diabetics (n = 58) | LOO AUC 0.367 (null) | 0.212 to 0.520 |
| PPG-BP, diabetes from fingertip PPG (n = 219) | AUC 0.37 to 0.49 on every route (null); hypertension positive control 0.726 | 0.653 to 0.802 |
| Digital twin, thermal | Halving choroidal perfusion cools the cornea by 0.54 °C; a 90° thermopile registers 28 % of it, a 5° part 14 %; MMS order 1.79; GCI at most 2.0 %; ambient temperature Sobol' S_T 0.74 to 1.01 | |
| Digital twin, optical | 3 mm temple PPG: arterial share of attenuation 0.7 % (660 nm) and 2.1 % (880 nm); IEC 62471 caps the 940 nm illuminator at 42.3 mW, which warms the cornea by 0.21 °C | |
| Digital twin, electrical | Local tetrapolar transfer impedance 427 Ω at 50 kHz; orbit, eye and lid share 15.6 %; +10 % orbital water changes it by 1.39 % vs 0.13 % for a bipolar pair with contact impedance | |
| Digital twin, pupil | A 940 nm illuminator evokes 0.10 mm of constriction vs 2.23 mm at 300 lx; the bridge camera under-reads pupil diameter by 36.9 % without an ellipse correction | |

Every number is read from `results/RESULTS_MASTER.json`, which records the source file and key path of each entry.

## 3. Datasets (all public or open; none is redistributed here)

| Dataset | Link | Licence | Used for |
|---|---|---|---|
| DeepDRiD (Shanghai) | https://github.com/deepdrdoc/DeepDRiD, paper https://doi.org/10.1016/j.patter.2022.100512 | CC BY-SA 4.0 | Fundus grader training, patient-grouped OOF test |
| IDRiD (Nanded) | https://doi.org/10.21227/H25W98 (IEEE DataPort), paper https://doi.org/10.3390/data3030025 | CC BY 4.0 | External DR test; DME head training and fixed test |
| APTOS 2019 (Aravind, India) | https://www.kaggle.com/competitions/aptos2019-blindness-detection (public mirror used: https://www.kaggle.com/datasets/mariaherrerot/aptos2019) | Kaggle competition rules | Second external DR test |
| Istanbul e-Nabız diabetes EHR export | https://doi.org/10.17632/rr4rzzrjfc.2 (Mendeley Data) | CC BY 4.0 | Retinopathy and neuropathy models (n = 77,724) |
| PhysioNet: Cerebral Vasoregulation in Diabetes | https://doi.org/10.13026/m40k-4758 | CC BY 4.0 | HRV branch |
| PhysioNet: Cerebral Perfusion and Cognitive Decline in Type 2 Diabetes | https://doi.org/10.13026/rbeh-9r20 | CC BY 4.0 | HRV branch (retinopathy labels, HbA1c, autonomic symptoms) |
| PPG-BP (Guilin) | https://doi.org/10.6084/m9.figshare.5459299, paper https://doi.org/10.1038/sdata.2018.20 | CC0 | PPG branch |
| PaPaGei-S foundation model weights | https://doi.org/10.48550/arXiv.2410.20542 (weights from the authors' release) | see authors' licence | PPG embedding route |
| BRSET / mBRSET (Brazil) | https://physionet.org/content/brazilian-ophthalmological/1.0.1/ | PhysioNet credentialed | Not used; access pending |

Download instructions, licence notes and the expected folder layout under `results/data/raw/` are in `results/DATA_RETRIEVAL.md` and `results/OPEN_DATASETS.md`. Data are not part of this repository.

## 4. Repository layout

```
results/
  RESULTS_MASTER.json        single source of every reportable number (value, CI, source file, key path)
  src/                       training, evaluation and figure scripts (see section 6)
    harness/                 shared splitting, metrics and reproducibility helpers
  out/                       aggregate outputs: *.json metrics, *.npz per-subject predictions (no images, no patient rows)
  figures/paper_npjdm/       final figure set Fig1..Fig8 and SupplementaryFig1..7 (PDF + 600 dpi PNG) with MAP.md
  figures/                   all figure PNG/PDF pairs and CAPTIONS.md
  notebooks/NRDI_results.ipynb   executed walk-through of every result
  DATA_RETRIEVAL.md, OPEN_DATASETS.md
sim/
  SIMULATION_PLAN.md         plan and credibility table for the digital twin
  cad/                       parametric frame and sensor layout (build123d)
  forward_models/            periorbital mesh, thermal, optical (MCX), electrical, pupil, verification, UQ
  render/                    Blender scripts for the CAD hero shots, exploded assembly and physics overlays
  render/out/                selected renders (PNG) and the exploded-assembly animation (MP4)
  dashboard/                 build.py + template.html -> nrdi_twin.html (self-contained live digital twin, v1)
  out/                       simulation outputs as JSON (meshes, fields and binaries are excluded)
requirements.txt
```

## 5. Setup

```bash
git clone https://github.com/shlokDS16/Diabetic-retinopathy.git
cd Diabetic-retinopathy
python -m venv .venv && .venv\Scripts\activate      # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Notes. The fundus scripts need a CUDA GPU (a laptop RTX 5070 Ti was used; one epoch takes about 33 s from the 512 px cache). `pmcx` (Monte Carlo optics) needs a CUDA GPU and was run in a separate environment with `pmcx 0.7.1`; `pyvista` was used in a separate environment for the thermal slices. `hrv_open.py` needs `neurokit2` and `wfdb`. Place the datasets under `results/data/raw/` as described in `results/DATA_RETRIEVAL.md`. Seeds are fixed (20260828 for every learning experiment, 7 for the Sobol' sampler).

## 6. Reproducing the results

| Result | Command | Output |
|---|---|---|
| Fundus grader (5-fold, DeepDRiD) + IDRiD external test | `python results/src/train_fundus.py --model convnext_tiny.fb_in22k_ft_in1k --img 384 --epochs 10` (`--eval-only` reuses saved fold models) | `results/out/fundus_deepdrid.json`, `fundus_oof.npz`, `fundus_idrid_pred.npz` |
| APTOS 2019 external test | `python results/src/eval_aptos.py` (downloads the competition set with a Kaggle token, else the public mirror) | `results/out/fundus_aptos.json`, `fundus_aptos_pred.npz` |
| DME head (IDRiD) | `python results/src/train_dme.py --epochs 12` | `results/out/fundus_dme_idrid.json`, `dme_idrid_pred.npz` |
| EHR models (Istanbul export) | `python results/src/tabular_ehr.py` | `results/out/tabular_ehr.json`, `ehr_oof.npz` |
| HRV branch (PhysioNet) | `python results/src/hrv_open.py` | `results/out/hrv_open.json`, `hrv_pred.npz` |
| PPG branch (PPG-BP) | `python results/src/ppg_open.py` | `results/out/ppg_open.json`, `ppg_pred.npz` |
| Results master | `python results/src/build_results_master.py` | `results/RESULTS_MASTER.json` |
| Figures | `python results/src/figF01_architecture.py` ... `figF13_physio.py`, `figM03_thermal_credibility.py`, `figM04_optical_electrical_pupil.py` | `results/figures/` |
| Notebook | `jupyter nbconvert --to notebook --execute results/notebooks/NRDI_results.ipynb` | executed notebook |

Fold-model weights (`results/out/fundus_fold{0-4}.pt`, `fundus_dme{0-4}.pt`, about 110 MB each) are not in the repository; they will be deposited with a DOI on publication and are available on request.

## 7. Reproducing the digital twin

| Step | Command | Tool |
|---|---|---|
| Frame and sensor poses | `python sim/cad/sensor_layout.py`, `python sim/cad/frame.py` | build123d |
| Periorbital mesh (14 tissues) | `python sim/forward_models/periorbital_mesh.py` | gmsh |
| Thermal forward model and thermopile readings | `python sim/forward_models/thermal3d.py` (add `--mesh sim/out/periorbital_coarse.msh` for the coarse mesh) | scikit-fem |
| Code and solution verification (MMS, GCI) | `python sim/forward_models/verification.py` | scikit-fem |
| Sobol' uncertainty and surrogate | `python sim/forward_models/uq_thermal.py` | SciPy QMC |
| Voxelisation and Monte Carlo optics | `python sim/forward_models/voxelize.py`, `python sim/forward_models/mcx_run.py --nphoton 3e7` | pmcx (MCX) |
| Illuminator heating | `python sim/forward_models/opto_thermal.py` | scikit-fem |
| Bioimpedance (complete electrode model, lead fields) | `python sim/forward_models/bioimpedance3d.py` | scikit-fem |
| Pupil model and camera projection | `python sim/forward_models/pupil.py` | NumPy |
| Renders and exploded animation | `sim/render/blender_hero.py`, `blender_explode.py`, `blender_physics.py` (Blender 5.2, see `sim/render/DEMO.md`) | Blender |
| Live digital twin dashboard | `python sim/dashboard/build.py` then open `sim/dashboard/nrdi_twin.html` | Three.js (offline) |

Tissue properties and every boundary condition are in `sim/out/tissue_db.json` with their literature sources in `sim/out/tissue_db_sources.md`.

## 8. The eight design verdicts returned by the twin

1. A 90° thermopile field of view reads a 28 mm spot and registers only 28 % of a corneal deficit (a 5° part registers 14 %): infer the deficit through the twin's transfer factor.
2. The 940 nm illuminator evokes no measurable pupil reflex: add a visible stimulus of at least 100 lx.
3. Aim the thermopile at the ocular surface, not the canthus skin (choroidal Sobol' index 0.11 vs below 0.01).
4. Dry-pad contact impedance is 68 % of a two-electrode reading: drive and sense on separate pairs.
5. A 3 mm PPG reads the dermal plexus, not the artery (arterial share 2.1 % at 880 nm).
6. A cross-eye electrode pair does not see the orbit: use four local electrodes around one eye.
7. A continuously lit illuminator warms the cornea by 0.21 °C at the safety limit, 39 % of the deficit sought: pulse it.
8. Ambient temperature dominates every thermal readout (0.32 °C per °C): log it and compensate.

## 9. Integrity notes

* No quantum machine learning contributed to any result. `results/src/concept_quantum.py` is a scaffold that was smoke-tested once and produced no output; it is imported by none of the result scripts.
* The neuropathy model excludes drug classes that are dispensed for neuropathic pain (antiepileptics, ATC A16, other nervous-system drugs) because they follow the diagnosis; the reported AUC is 0.710. Both sensitivity analyses are in `results/out/tabular_ehr.json` under `dn_sensitivity`.
* The APTOS result is a calibration story: ordering transfers, grade boundaries do not.
* The two null results (HRV vs retinopathy, PPG vs diabetes) are reported as findings with positive controls.
* Credentialed PhysioNet data (BRSET, mBRSET) were not used.

## 10. Citation

A manuscript describing this work is in preparation. Until it is published, please cite this repository:

```
Goenka, S. et al. NRDI: physics-verified periorbital wearable with multimodal AI for joint diabetic retinopathy and neuropathy screening. GitHub repository, 2026. https://github.com/shlokDS16/Diabetic-retinopathy
```

Please also cite the dataset papers listed in section 3 when reusing their data.

## 11. Licence and contact

Code is released under the MIT Licence (see `LICENSE`). Figures and simulation outputs are CC BY 4.0. The datasets keep their own licences. A provisional patent application covering the wearable design is being filed by the authors.

Contact: Shlok Goenka, Vellore Institute of Technology, shlok.goenka2023@vitstudent.ac.in
