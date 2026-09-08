# Figure plan — 13 main figures + appendix (drafted 2026-09-07; F1–F13 BUILT the same evening, see CAPTIONS.md)

Style: `results/src/pubstyle.py` (SciencePlots nature + Okabe-Ito, 88/180 mm, 7 pt, bold lowercase
panel labels, PDF master + 600-dpi PNG). Rules from `FIGURE_TOOLING.md`: constrained layout, no
overlapping labels (adjustText where needed), legends outside axes when >3 series, one message per panel.
Every number on a figure is read from a results file, never typed in.

| # | File | Title | Panels | Data (all exist unless flagged) |
|---|---|---|---|---|
| F1 | figF01_architecture | NRDI system and evidence pipeline | single diagram: device sensors → signals → branch models → ordinal fusion → DR/DN outputs; simulation loop feeding design verdicts | schemdraw flow (vector), no data |
| F2 | figF02_anatomy_sensors | Anatomical periorbital model and sensor placement | a hero render, b exploded assembly, c mesh cut with 14 tissue regions + boundary groups, d sensor pose table as annotated view | sim/render/out/*.png, sim/out/periorbital_summary.json, sensor_layout.json |
| F3 | figF03_thermal | Bioheat field and thermopile readout | a face temperature map (face_T.ply→render), b sagittal slice, c BAA 90° vs DCI 5° spot and reading vs choroid perfusion (choroid_sweep), d simulated vs published corneal/canthus ranges (validation) | thermal3d_full.json, fields/slice_T_*.png, face_T.ply |
| F4 | figF04_verification | Verification: MMS convergence, GCI and opto-thermal load | a L2 error vs h with observed order, b GCI per QoI, c LED heating vs optical power with IEC limit | verification.json, opto_thermal.json, thermal_sweep.json |
| F5 | figF05_uq | Uncertainty quantification of thermal readouts | a Sobol S1/ST heat-grid 8 params × 4 outputs, b output distributions (640 Saltelli samples), c surrogate parity + RMSE | surrogate_thermal.json, uq_thermal_samples.npz |
| F6 | figF06_optical | Monte Carlo optics: PPG sampling depth and LED corneal exposure | a banana 660/880 nm slices, b spacing sweep: dermal vs arterial fraction, c LED irradiance vs IEC 62471 limit and absorbed fraction per tissue | optical_results.json, ppg_banana_*.npy, led_absorbed_940.npy |
| F7 | figF07_bioimpedance | Periorbital bioimpedance sensitivity | a lead-field on mesh, b tissue contribution (Geselowitz) tetrapolar vs bipolar, c \|Z\| and phase vs frequency, d edema sensitivity and cross-eye ratio, e reciprocity residual | bioimpedance3d_coarse.json, fields/leadfield.ply |
| F8 | figF08_pupil | Pupillography model and 940 nm stimulus verdict | a traces healthy vs autonomic gain 0.92, b metrics vs gain sweep, c stimulus wavelength/irradiance sweep, d camera vergence correction | pupil_model.json |
| F9 | figF09_fundus_dr | Fundus DR grading: patient-level CV and external tests | a OOF confusion (5×5), b referable ROC with bootstrap bands: DeepDRiD OOF, IDRiD, APTOS, c prediction distribution per true grade (violin), d per-fold validation QWK curves | fundus_deepdrid.json, fundus_oof.npz, fundus_idrid_pred.npz, fundus_aptos.json/npz (PENDING download) |
| F10 | figF10_generalisation | Cross-population generalisation and threshold transfer | a forest: QWK, referable AUC, accuracy per dataset with CIs, b sens/spec at referable operating point per dataset, c DeepDRiD thresholds vs refit thresholds (IDRiD, APTOS), d grade-prevalence shift bars | fundus_*.json (APTOS PENDING) |
| F11 | figF11_dme | Macular-oedema head and DR–DME co-occurrence | a DME confusion (test), b any-DME ROC CV vs test, c DR grade × DME grade heat-grid on IDRiD, d DME score vs DR score scatter | fundus_dme_idrid.json, dme_idrid_pred.npz, fundus_idrid_pred.npz |
| F12 | figF12_ehr | Systemic EHR model of DR and DN (n = 77,724) | a ROC DR/DN logistic vs HGB with bands, b calibration curves (quantile bins), c forest of odds ratios per SD (top 12), d permutation importance, e joint outcome OvR AUC + DR–DN 2×2 with OR | tabular_ehr.json; NEEDS `ehr_oof.npz` (save OOF probabilities: small patch to tabular_ehr.py, rerun) |
| F13 | figF13_physio | Physiological signals: HRV and PPG | a HRV DM vs control ROC, b Cohen d DR+ vs DR− within DM with BH p, c HbA1c Spearman rho per feature, d PPG diabetes AUC by route + HTN control, e PaPaGei age parity (positive control) | hrv_open.json, ppg_open.json; NEEDS `hrv_pred.npz`, `ppg_pred.npz` (save CV predictions: small patches, rerun) |

Appendix (built after the 13): A1 mesh quality histograms and tissue property table; A2 thermal
sweep of ambient/h_skin; A3 per-tissue absorption at 940 nm; A4 fold-wise DME curves; A5 EHR
feature list with leakage exclusions; A6 HRV feature distributions; A7 dataset provenance table.

Prerequisite patches before F12/F13: `tabular_ehr.py` → save `ehr_oof.npz` (y_dr, y_dn, p_lr, p_hgb);
`hrv_open.py` → save LOO scores; `ppg_open.py` → save CV probabilities. Reruns are minutes.

Order of construction: F9/F10 (as soon as APTOS lands) → F12 → F13 → F11 → F5 → F4 → F3 → F6 → F7 → F8 → F2 → F1 → appendix.
Check-off list kept here; each figure gets a one-line caption draft in `CAPTIONS.md` when built.

## Build status (2026-09-07 evening)
Built: F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13 (PDF + 600-dpi PNG in results/figures/).
Pending: re-run F9 and F10 after `python results/src/eval_aptos.py` finishes (APTOS panel/rows appear automatically).
Prerequisite patches done: ehr_oof.npz, hrv_pred.npz, ppg_pred.npz are written by the branch scripts.
Change of record: the neuropathy EHR model now excludes treatment proxies (antiepileptics, other_digestive = ATC A16
alpha-lipoic acid, other_nervous_drugs); DN AUC 0.838 -> 0.710. Proxy-included and wider-exclusion runs are kept as
sensitivity analyses in tabular_ehr.json. Appendix figures A1-A7 not started.
Tooling notes: SciencePlots leaves top/right ticks on -> pubstyle sets xtick.top/ytick.right False. Long titles
collide with panel labels at 45 mm panel width -> use loc="right" and two lines. Legends wider than the axes
overflow to the left; shorten labels instead of moving the legend.
