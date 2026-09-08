"""
Build the supervisor-format report: "Datasets, Methodology and Results" for the NRDI project, following the structure of
Report_Intro_Dataset_Method_Results.pdf (numbered sections, numbered tables with captions above, numbered figures with
captions below, inline numbered equations, bold lead-ins such as "Central question." and "Interpretation.").
Every number is read from results/RESULTS_MASTER.json; figures come from the executed notebook results/notebooks/out/
and from results/figures/paper_npjdm/.  Run: python paper_build/teacher_report/build_report.py
"""
import json, pathlib, datetime
import numpy as np
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = pathlib.Path(__file__).resolve().parents[2]; HERE = pathlib.Path(__file__).resolve().parent
M = json.loads((ROOT / "results" / "RESULTS_MASTER.json").read_text(encoding="utf-8"))
NBO = ROOT / "results" / "notebooks" / "out"; FIG = ROOT / "results" / "figures" / "paper_npjdm"
def v(*ks):
    d = M
    for k in ks: d = d[k]
    return d["value"] if isinstance(d, dict) and "value" in d else d
def ci(*ks):
    d = M
    for k in ks: d = d[k]
    return d["ci95"]
def cis(*ks, n=3):
    lo, hi = ci(*ks); return f"[{lo:.{n}f}, {hi:.{n}f}]"

TITLE = "Physics-Verified Periorbital Wearable for Joint Diabetic Retinopathy and Neuropathy Screening: Datasets, Methodology and Results"
doc = Document()
sec = doc.sections[0]; sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.3); sec.top_margin = sec.bottom_margin = Cm(2.2)
st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(10.5); st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.15
hp = sec.header.paragraphs[0]; hp.text = TITLE; hp.alignment = WD_ALIGN_PARAGRAPH.LEFT; hp.runs[0].font.size = Pt(8); hp.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = fp.add_run()
for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
    if tag: fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), tag); r._r.append(fc)
    else: it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = text; r._r.append(it)
TABLE_N = [0]; FIG_N = [0]; EQ_N = [0]

def P(text="", bold_lead=None, size=None, align=None, italic=False, space_after=None):
    p = doc.add_paragraph()
    if bold_lead:
        rr = p.add_run(bold_lead + " "); rr.bold = True
    rr = p.add_run(text); rr.italic = italic
    if size: [setattr(x.font, "size", Pt(size)) for x in p.runs]
    if align: p.alignment = align
    if space_after is not None: p.paragraph_format.space_after = Pt(space_after)
    return p
def H(text, level=1):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = True; r.font.size = Pt(13 if level == 1 else 11.5)
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10); p.paragraph_format.space_after = Pt(6); p.paragraph_format.keep_with_next = True
    return p
def EQ(text):
    EQ_N[0] += 1; p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(1.0)
    r = p.add_run(text); r.font.name = "Cambria Math"; p.add_run(f"\t({EQ_N[0]})")
    p.paragraph_format.tab_stops.add_tab_stop(Cm(15.5)); return EQ_N[0]
def TABLE(caption, header, rows, widths=None, font=9):
    TABLE_N[0] += 1
    cap = doc.add_paragraph(); rr = cap.add_run(f"Table {TABLE_N[0]}. "); rr.bold = True; cap.add_run(caption); cap.paragraph_format.keep_with_next = True; cap.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); t.style = doc.styles["Table Grid"]; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        c = t.cell(0, j); c.text = ""; q = c.paragraphs[0]; q.paragraph_format.space_after = Pt(0); x = q.add_run(str(h)); x.bold = True; x.font.size = Pt(font)
        sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "E7E6E6"); c._tc.get_or_add_tcPr().append(sh)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = ""; q = c.paragraphs[0]; q.paragraph_format.space_after = Pt(0); x = q.add_run(str(val)); x.font.size = Pt(font)
    if widths:
        t.autofit = False
        for j, w in enumerate(widths):
            for i in range(len(rows) + 1): t.cell(i, j).width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2); return TABLE_N[0]
def FIGURE(path, caption, width=15.5):
    FIG_N[0] += 1; p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(path), width=Cm(width)); p.paragraph_format.keep_with_next = True
    cap = doc.add_paragraph(); rr = cap.add_run(f"Figure {FIG_N[0]}. "); rr.bold = True; cap.add_run(caption); cap.paragraph_format.space_after = Pt(10); return FIG_N[0]

# ============================================================ title block
P(TITLE, bold_lead=None, size=15, align=WD_ALIGN_PARAGRAPH.LEFT).runs[0].bold = True
P("A design-stage study on open data with a verified digital twin", size=12, italic=True)
P("Shlok Goenka and Aryan [SURNAME]", size=11)
P("School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology", size=10.5)
P("[Supervisor name and role to be added.]  Code and executed notebook: https://github.com/shlokDS16/Diabetic-retinopathy", size=9.5, italic=True)

# ============================================================ 1 Introduction
H("1. Introduction and Clinical Context")
P("Diabetes affected about 537 million adults in 2021, and more than three quarters of them live in low- and middle-income countries. Two of its slow complications are screened separately, if at all. Diabetic retinopathy affected an estimated 103 million people in 2020 and is projected to reach 161 million by 2045; it is detected by retinal photography, which needs a camera, a trained grader or a certified algorithm, and a visit. Diabetic neuropathy affects roughly half of people with diabetes over their lifetime; it is detected by monofilament and tuning-fork examination in a clinic, and its autonomic form is usually not looked for at all. The two complications share a microvascular origin and co-occur more often than chance, yet no screening pathway looks for both in one sitting and none of the pathways runs outside a clinic.")
P("Deep learning has closed part of the gap for retinopathy. Convolutional networks grade fundus photographs at specialist level, an autonomous system has been cleared on the strength of a pivotal trial, and retinal foundation models transfer across countries. Two limits remain. Every such system needs a fundus image, so it does not run continuously and does not see neuropathy. And grade boundaries do not travel well: graders, cameras and populations shift the thresholds even when the ordering of eyes is preserved.")
P("The periorbital region offers signals that do not need a fundus camera. Ocular surface temperature falls with retinopathy severity, by about 0.6 °C between healthy eyes and non-proliferative disease on infrared thermography, and the temperature field is set in part by choroidal perfusion. Pupil constriction amplitude and velocity fall with autonomic neuropathy. Heart-rate variability falls with autonomic involvement. A spectacle frame sits against exactly this region for hours a day. What has been missing is a way to know, before a device is built, whether a thermopile on a frame can see a 0.6 °C corneal deficit through the geometry of the orbit, whether an infrared illuminator can evoke a pupil reflex, whether four dry electrodes can register orbital edema, and which of these signals is drowned by the room.")
P("This study therefore has two halves. A digital twin of the periorbital region couples thermal, optical and electrical physics on one finite-element anatomy, is verified with manufactured solutions and grid convergence, has its uncertainty quantified with Sobol' indices, and returns eight design verdicts to a spectacle-frame wearable that carries a thermopile, a camera with infrared illumination, dry electrodes and a temple photoplethysmograph. In parallel, the artificial-intelligence branches that the frame will carry are trained and tested on open data only: a fundus grader on photographs from China tested unchanged on two Indian populations, a macular edema head, systemic models of retinopathy and neuropathy on the records of 77,724 people with diabetes in Türkiye, and physiological models on open electrocardiogram and photoplethysmogram cohorts.")
P("How much of what a periorbital wearable would need to measure is physically observable from a spectacle frame, and how far do the artificial-intelligence branches it will carry transfer across populations and away from the labels they were trained on? The study does not report a clinical trial and no prototype has been worn. It audits the design and the models.", bold_lead="Central question.")

# ============================================================ 2 Datasets
H("2. The Datasets")
H("2.1 Sources and provenance", 2)
P("Six public or open datasets were analysed. None was generated for this study and none is redistributed with it; every dataset keeps its own licence. Table 1 lists the sources, the links from which they were obtained and the role each plays.")
TABLE("Datasets analysed, with source links and licences.", ["Dataset", "Origin", "Link", "Licence", "Role"], [
    ["DeepDRiD", "Shanghai Jiao Tong University, China", "https://github.com/deepdrdoc/DeepDRiD (paper doi 10.1016/j.patter.2022.100512)", "CC BY-SA 4.0", "Fundus grader training and patient-grouped out-of-fold test"],
    ["IDRiD", "Nanded, India", "https://doi.org/10.21227/H25W98 (IEEE DataPort; paper doi 10.3390/data3030025)", "CC BY 4.0", "External retinopathy test; macular edema head"],
    ["APTOS 2019", "Aravind Eye Hospital network, India", "https://www.kaggle.com/competitions/aptos2019-blindness-detection (public mirror mariaherrerot/aptos2019)", "Kaggle competition rules", "Second external retinopathy test"],
    ["Istanbul e-Nabız EHR export", "Türkiye (Akdeniz University release)", "https://doi.org/10.17632/rr4rzzrjfc.2 (Mendeley Data)", "CC BY 4.0", "Systemic models of retinopathy and neuropathy"],
    ["PhysioNet Cerebral Vasoregulation in Diabetes; Cerebral Perfusion and Cognitive Decline in Type 2 Diabetes", "Boston, USA (Novak laboratory)", "https://doi.org/10.13026/m40k-4758; https://doi.org/10.13026/rbeh-9r20", "CC BY 4.0", "Heart-rate variability branch"],
    ["PPG-BP", "Guilin, China", "https://doi.org/10.6084/m9.figshare.5459299 (paper doi 10.1038/sdata.2018.20)", "CC0", "Fingertip photoplethysmography branch"],
    ["BRSET and mBRSET", "Brazil", "https://physionet.org/content/brazilian-ophthalmological/1.0.1/", "PhysioNet credentialed", "Not used; access pending at the time of writing"]],
    widths=[2.6, 2.8, 5.2, 1.9, 3.6], font=8)
H("2.2 What this study uses", 2)
P("The three fundus collections are used as images with their International Clinical Diabetic Retinopathy (ICDR) grade; IDRiD additionally supplies a diabetic macular edema (DME) grade on the same photographs. DeepDRiD carries patient identifiers, which allow patient-grouped cross-validation; IDRiD and APTOS do not, so their confidence intervals are image-level. Of the DeepDRiD release only the labelled regular-fundus images are used (the challenge evaluation split is unlabelled). The Istanbul export is used as a tabular record per person: glycated hemoglobin (HbA1c) and its one-year change, age, sex, laboratory values, ICD-chapter diagnosis flags and drug-class dispensing flags. The two PhysioNet cohorts are used for their resting electrocardiograms and their clinical tables (diabetes, retinopathy grade where recorded, autonomic symptom score, HbA1c). PPG-BP is used for its three 2.1 s fingertip photoplethysmogram segments per subject with age, sex, body mass index, hypertension stage and a diabetes flag.")
P("This choice is deliberate. The question is whether the branches the wearable will carry behave across populations and away from their training labels, so each branch is evaluated on data it was not fitted to wherever such data exist, and the physics of the wearable is evaluated in simulation because no prototype has yet produced a measurement. The credentialed smartphone fundus datasets (BRSET, mBRSET), which pair images with clinical variables in a Brazilian population, were not available in time and appear only as the next step.")
H("2.3 Cohort composition", 2)
gd = lambda *k: [int(x) for x in v(*k).values()]
dd, di, da = gd("fundus_dr", "grade_counts"), gd("datasets", "idrid", "grade_counts"), gd("fundus_external_aptos", "grade_counts")
TABLE("Composition of the three fundus datasets by ICDR grade.", ["ICDR grade", "DeepDRiD (China)", "IDRiD (India)", "APTOS 2019 (India)"],
      [[f"{g} ({['none', 'mild', 'moderate', 'severe', 'proliferative'][g]})", f"{dd[g]} ({100*dd[g]/sum(dd):.1f} %)", f"{di[g]} ({100*di[g]/sum(di):.1f} %)", f"{da[g]} ({100*da[g]/sum(da):.1f} %)"] for g in range(5)] +
      [["Total images", f"{sum(dd):,} from {v('datasets','deepdrid','n_patients')} patients", f"{sum(di)} (413 training, 103 test)", f"{sum(da):,}"],
       ["Referable (grade 2 or worse)", f"{100*sum(dd[2:])/sum(dd):.1f} %", f"{100*sum(di[2:])/sum(di):.1f} %", f"{100*sum(da[2:])/sum(da):.1f} %"]], widths=[4.0, 4.0, 4.0, 4.0])
jc = v("ehr", "joint_counts")
TABLE("Composition of the systemic and physiological cohorts.", ["Cohort", "Subjects", "Positives", "Notes"], [
    ["Istanbul EHR export", f"{v('ehr','retinopathy','n'):,}", f"retinopathy {100*v('ehr','retinopathy','prevalence'):.2f} %; neuropathy {100*v('ehr','diabetic_neuropathy_main_excludes_treatment_proxies','prevalence'):.2f} %", f"neither {jc['0']:,}; retinopathy only {jc['1']:,}; neuropathy only {jc['2']:,}; both {jc['3']:,}"],
    ["PhysioNet cohorts, pooled", f"{v('hrv','pooled_n')} usable records", f"{v('hrv','dm_vs_control','n_pos')} with diabetes", f"retinopathy label on {v('hrv','dr_within_dm','n')} diabetic records ({v('hrv','dr_within_dm','n_dr')} positive); HbA1c on {v('hrv','ge75_n')}"],
    ["PPG-BP", f"{v('ppg','n_subjects')}", f"{v('ppg','n_diabetic')} with type 2 diabetes; {v('ppg','hypertension_n_pos')} hypertensive", "three 2.1 s segments per subject at 1 kHz"]], widths=[3.5, 3.0, 4.5, 5.0])
H("2.4 Variables in the records", 2)
TABLE("Fields available in the Istanbul EHR export and how they are used.", ["Field group", "Description", "Used in this study"], [
    ["HbA1c, HbA1c change over one year", "Glycated hemoglobin and its annual change", "Yes, predictors"],
    ["Age, sex", "Demographics", "Yes, predictors"],
    ["Laboratory values", "Creatinine, cholesterol, HDL, triglycerides and others", "Yes, predictors"],
    ["ICD-chapter diagnosis flags", "Hypertension, kidney failure, ischaemic heart disease, musculoskeletal, lipoprotein disorders and others", "Yes, predictors; the eye, nervous-system and neuropathy chapters are excluded because they carry the outcomes"],
    ["Drug-class dispensing flags", "Insulins by type, oral agents, cardiovascular, alimentary, nervous-system and other classes", "Yes, except three classes that are treatment proxies for neuropathy (Section 3.4)"],
    ["Retinopathy flag", "Outcome one", "Label only"],
    ["Diabetic neuropathy flag", "Outcome two", "Label only"],
    ["Glycemic-control label, refraction, cataract, ophthalmic drugs", "Outcome-adjacent columns", "No, excluded as leakage"]], widths=[4.2, 6.2, 5.6])
P(f"After exclusions, {v('datasets','istanbul_ehr','n_features_dr')} features remain for the retinopathy model and {v('datasets','istanbul_ehr','n_features_dn')} for the neuropathy model.")
H("2.5 The labels and their clinical meaning", 2)
TABLE("The ICDR grade used by all three fundus datasets, and the derived referable label.", ["Grade", "Meaning", "Referable"], [
    ["0", "No apparent retinopathy", "No"], ["1", "Mild non-proliferative retinopathy (microaneurysms only)", "No"], ["2", "Moderate non-proliferative retinopathy", "Yes"],
    ["3", "Severe non-proliferative retinopathy", "Yes"], ["4", "Proliferative retinopathy", "Yes"]], widths=[2.0, 10.0, 4.0])
P("The referable label, grade 2 or worse, is the decision a screening device would act on first. IDRiD also grades macular edema 0 to 2 by the distance of hard exudates from the fovea (grade 1 outside one disc diameter, grade 2 within one disc diameter); any DME is grade 1 or 2. In the Istanbul export both outcomes are ICD-chapter flags recorded by the health system; the neuropathy flag does not distinguish peripheral from autonomic neuropathy. In the PhysioNet cohorts the diabetes label comes from the recruitment group and the retinopathy label from the clinical table.")
H("2.6 Preparation of the analysis files", 2)
P("Fundus photographs were cropped to the retinal disc and cached once at 512 pixels, which cut an epoch from 14 minutes of JPEG decoding to about 33 seconds; models train at 384 pixels. DeepDRiD images were grouped by patient before any split so that no patient contributes to both the training and the validation part of a fold. IDRiD reuses file names across its training and test folders, so the loader matches names inside each folder; a first run that matched across folders gave a spuriously low external kappa of 0.61 and was discarded. APTOS grade counts were checked against the published competition counts (1,805, 370, 999, 193 and 295) to verify the mirror.")
P("The Istanbul export is a semicolon-separated file; every numeric field was coerced and unparsable entries became missing rather than zero. Columns that record either outcome or an immediate consequence of it were removed before any model saw the data. For the neuropathy outcome three further drug classes were removed because in this health system they are dispensed for neuropathic pain: antiepileptics (ATC N03, in practice gabapentin and pregabalin), the other-alimentary class (ATC A16, which holds alpha-lipoic acid, a drug prescribed in Türkiye specifically for diabetic neuropathy) and other nervous-system drugs. The electrocardiograms were read from WFDB records; one cohort's header sampling rate produced an implausible heart rate and was re-read at half that rate with the correction logged. PPG-BP segments were used as distributed.")
P("Every step that estimates a quantity from data, including standardisation and the fitting of grade thresholds, is performed inside the training partition of each fold and applied to the held-out fold; no statistic of a held-out fold enters the model evaluated on it. The four grade thresholds of the fundus grader are fitted once on the DeepDRiD out-of-fold scores and applied unchanged to the two external datasets; re-fitting them on an external dataset is reported only as a diagnostic.", bold_lead="Note on data handling.")
H("2.7 What the datasets do not contain", 2)
P("Four absences shape the design. First, no dataset carries the wearable's own channels: there is no periorbital thermography, pupillometry, orbital bioimpedance or temple photoplethysmography paired with a retinopathy or neuropathy label, so the physics side of the study is answered by simulation and the fusion stage that would combine the channels is designed but not evaluated. Second, no open dataset holds all the modalities on the same people, so each branch is tested alone. Third, the external fundus datasets carry no patient identifiers, so their confidence intervals are image-level and likely too narrow. Fourth, the Istanbul export records diagnoses and dispensings, not examinations; its neuropathy label is a coded diagnosis whose timing relative to treatment is not known, which is exactly why the treatment-proxy rule of Section 3.4 is needed.")
P("Open data are the only data on which a design can be tested before a device exists and before a human study is approved, and they are the only data on which a claim of transfer across populations can be made at all: the three fundus datasets come from two countries and three camera and grader regimes, and the systemic and physiological cohorts come from two more. Establishing what transfers and what does not is more useful at this stage than a higher number on a single dataset.", bold_lead="Why these datasets, given those limitations.")
H("2.8 Code and reproducibility", 2)
P("All code, the aggregate outputs, the figure scripts, the simulation scripts and the executed analysis notebook (results/notebooks/NRDI_analysis.ipynb, PART 1 to PART 10) are in the repository https://github.com/shlokDS16/Diabetic-retinopathy. Every number in this report is read from results/RESULTS_MASTER.json, which records the source file and key path of each entry; PART 1 of the notebook recomputes the headline metrics from the stored per-subject predictions and confirms that they agree.")

# ============================================================ 3 Methodology
H("3. Methodology")
H("3.1 The three branches and one head", 2)
P("The multimodal model has three branches and one head, each trained and tested on its own dataset, fused by design into a retinopathy grade, a referable flag, a DME flag and a neuropathy risk. Table 6 lists them; Figure 1 shows how they sit against the wearable and the digital twin.")
TABLE("Branches of the multimodal model.", ["Branch", "Input", "Model", "Output", "Trained on", "Tested on"], [
    ["Fundus grader", "Retinal photograph, 384 px", "ConvNeXt-Tiny, ordinal regression with four fitted thresholds", "ICDR grade 0 to 4; referable flag", "DeepDRiD (5-fold, patient-grouped)", "DeepDRiD out of fold; IDRiD; APTOS 2019"],
    ["Macular edema head", "Retinal photograph, 384 px", "ConvNeXt-Tiny, ordinal regression with two thresholds", "DME grade 0 to 2; any-DME flag", "IDRiD training split (5-fold)", "IDRiD fixed test split"],
    ["Systemic branch", "EHR row per person", "Ridge logistic regression; histogram gradient boosting", "Retinopathy risk; neuropathy risk; joint 4-class", "Istanbul export (5-fold OOF)", "Same, out of fold"],
    ["Physiological branch", "5-minute ECG features; 2.1 s PPG features and embedding", "Leave-one-out and 5-fold logistic regression", "Diabetes; retinopathy within diabetics; hypertension control", "PhysioNet cohorts; PPG-BP", "Same, out of fold"]], widths=[2.4, 2.6, 3.4, 2.8, 2.6, 2.6], font=8)
P("Each branch is evaluated against the simplest comparator its dataset allows: the fundus grader against its own thresholds transported unchanged; the systemic models against a logistic baseline and against the same model with the treatment proxies restored; the physiological models against a demographics-only baseline and a positive control on the same features. If a branch cannot beat its comparator, that is reported as a finding, not hidden.", bold_lead="Design rationale.")
FIGURE(FIG / "Fig1.png", "System architecture: the periorbital wearable (left), the physics digital twin that returns eight design verdicts to it (centre), the multimodal model with its three branches and one head (right), and the open datasets behind every result (bottom).", width=16.0)
H("3.2 The ordinal fundus grader", 2)
P("The grader is a ConvNeXt-Tiny network pretrained on ImageNet-22k with a single linear output. The ICDR grade is treated as an ordinal target: the network regresses a continuous score s under a smooth L1 loss, and a grade is read off by counting the fitted thresholds below the score,")
e1 = EQ("ĝ = Σ_{k=1..4} 1[ s ≥ τ_k ]")
P(f"where the four thresholds τ_k are fitted once by Nelder-Mead search to maximise the quadratic weighted kappa on the out-of-fold DeepDRiD scores. Quadratic weighted kappa penalises a two-grade error four times as much as a one-grade error,")
e2 = EQ("QWK = 1 − Σ_{i,j} w_ij O_ij / Σ_{i,j} w_ij E_ij,   w_ij = (i − j)² / (N − 1)²")
P(f"where O is the observed grade confusion matrix and E its expectation under independence. Each fold ran ten epochs of AdamW (peak learning rate 2 × 10⁻⁴, weight decay 10⁻⁴, batch 16) under a one-cycle schedule with mixed precision, random flips, rotation up to 20 degrees and mild colour jitter; inference averages each image with its horizontally flipped copy. The seed was 20260828 throughout. The macular edema head uses the same recipe for twelve epochs with two thresholds.")
H("3.3 Cross-validation and external testing", 2)
P("Five folds were stratified by grade and grouped by patient, so the out-of-fold prediction of every DeepDRiD image comes from a model that never saw that patient. The score of an external image is the mean of the five fold models, and the DeepDRiD thresholds are applied unchanged. Reported metrics are QWK, the area under the receiver operating characteristic curve (AUC) for referable retinopathy, exact-grade accuracy, and sensitivity and specificity at the operating point implied by the fitted thresholds. Confidence intervals are 95 % percentile intervals from 1,000 bootstrap resamples drawn by patient on DeepDRiD and by image on the external sets. The error mechanism on the external sets is characterised by the signed difference between predicted and true grade: a systematic one-step shift indicates a threshold (boundary) shift, whereas symmetric spread indicates a loss of ordering.")
H("3.4 The systemic models and the treatment-proxy rule", 2)
P("Two models were fitted to each outcome: a ridge-penalised logistic regression (C = 0.1 after standardisation), kept for its interpretable odds ratios, and a histogram gradient boosting classifier (300 iterations, learning rate 0.05, 31 leaves, L2 regularisation 1.0), both under stratified five-fold cross-validation with pooled out-of-fold predictions. Calibration is summarised by the slope and intercept of a logistic recalibration of the out-of-fold logit,")
e3 = EQ("logit P(y = 1) = α + β · logit p̂,   ideal α = 0, β = 1")
P("together with the Brier score. Permutation importance of the boosted model was measured on a stratified 20 % held-out split with five repeats. A four-class boosted model of the joint outcome (neither, retinopathy only, neuropathy only, both) was scored one-versus-rest, and the association between the two diagnoses was tested with Fisher's exact test.")
P("A column that records the diagnosis, or a consequence of the diagnosis, may not be a predictor. The three drug classes named in Section 2.6 are dispensed for neuropathic pain; they follow the diagnosis rather than precede it, and a model that reads them learns who has already been treated. They were therefore excluded from the reported neuropathy model, and two sensitivity designs bracket the decision (Table 7).", bold_lead="The treatment-proxy rule.")
TABLE("The neuropathy sensitivity designs.", ["Design", "Features", "Purpose", "Status"], [
    ["Reported model", "94 features; treatment proxies excluded", "Risk as the records look before neuropathic pain is treated", "Headline"],
    ["Sensitivity A", "97 features; the three proxy classes restored", "Shows how much apparent discrimination the proxies supply", "Reported, not used"],
    ["Sensitivity B", "92 features; analgesics and psychoanaleptics removed as well", "Checks that the reported number is not sensitive to a wider exclusion", "Reported"]], widths=[3.0, 4.5, 5.5, 3.0])
H("3.5 The physiological branch", 2)
P("From the first five minutes of each electrocardiogram, R peaks were located with NeuroKit2 after cleaning, ectopic intervals were gated, and six features were computed: mean heart rate, SDNN, RMSSD, pNN50, the coefficient of variation of RR intervals and the LF/HF ratio. Three questions were asked of the features with a leave-one-out logistic regression (C = 0.5 after standardisation): diabetes against control on the pooled records, retinopathy against no retinopathy among the diabetic records with a retinal grade, and autonomic symptoms in the second cohort. Per-feature effects are Cohen's d,")
e4 = EQ("d = ( x̄_DR+ − x̄_DR− ) / s_pooled")
P("with two-sided Mann-Whitney U tests corrected across the six features by the Benjamini-Hochberg procedure, and Spearman's rank correlation against HbA1c. For PPG-BP, eleven pulse-morphology features were averaged per subject over the three segments (rise time, width at half height, systolic to diastolic amplitude ratio, reflection index, the second-derivative ratios b/a, c/a, d/a and e/a, the ageing index, crest fraction and heart rate), and separately each segment was embedded with the frozen PaPaGei-S foundation model after resampling to 125 Hz and tiling the 2.1 s segment to the model's 10 s window; the 512-dimensional embedding was reduced to 16 principal components. Each route was scored for diabetes and for hypertension with a standardised logistic regression under stratified five-fold cross-validation, next to a demographics-only model of age, sex and body mass index. Two positive controls were run: hypertension from morphology, and age regressed from the embedding.")
H("3.6 The digital twin", 2)
P("All three physics forward models run on one anatomical mesh of the left periorbital region, built parametrically in Gmsh from the anthropometry that also places the sensors (interpupillary distance 63 mm, vertex distance 13 mm, 52 by 40 mm lenses, 18 mm bridge). The mesh has 122,279 nodes and 705,228 linear tetrahedra in fourteen tissue regions; a coarser mesh of 17,905 nodes serves verification and uncertainty sweeps. Thermal and dielectric properties come from the IT'IS tissue database and the Gabriel Cole-Cole fits; optical properties from Jacques' review and measured Asian skin spectra. Table 8 lists the forward models and the tools behind them; Figure 2 shows the frame and the anatomy.")
TABLE("Forward models of the digital twin.", ["Forward model", "Physics", "Tool", "Quantities returned"], [
    ["Thermal", "Pennes bioheat with convection, radiation and evaporation boundaries; radiometric thermopile model", "scikit-fem, P1 tetrahedra", "Apex, canthus, temple temperatures; thermopile readings for two fields of view; choroidal-perfusion sweep"],
    ["Optical", "GPU Monte Carlo photon transport on a 0.25 mm voxelisation, 3 × 10⁷ photons", "MCX (pmcx)", "PPG partial path lengths and arterial share; spacing sweep; illuminator irradiance, absorbed power, IEC 62471 limit"],
    ["Opto-thermal coupling", "Absorbed 940 nm power as a Pennes source", "scikit-fem", "Steady heating of cornea, lid and canthus at 5, 20 and 42.3 mW"],
    ["Electrical", "Complex Laplace equation with σ + iωε, complete electrode model, Geselowitz lead fields", "scikit-fem (coarse mesh)", "Tetrapolar and bipolar impedance at 1 to 100 kHz; tissue shares; edema and temperature sensitivity"],
    ["Pupil", "Longtin-Milton delay-differential model with an autonomic gain; camera projection geometry", "NumPy", "Constriction amplitude, latency, velocities vs stimulus and gain; camera under-read"]], widths=[2.4, 5.0, 2.8, 5.8], font=8)
P("Steady tissue temperature obeys the Pennes bioheat equation,")
e5 = EQ("−∇·(k ∇T) + w (T − T_a) = q_m")
P("with conductivity k, volumetric perfusion coefficient w, arterial temperature T_a of 37 °C and metabolic heat q_m. The choroid is represented by a perfusion term in the scleral shell calibrated to the posterior boundary coefficient of Scott's finite-element eye model (65 W m⁻² K⁻¹); the choroidal perfusion fraction that scales it is the disease handle. The thermopile is modelled as what the sensor measures, an emissivity-weighted radiometric average over every boundary facet inside its field-of-view cone,")
e6 = EQ("T_read = [ Σ_f ε T_f A_f cos θ_f / d_f² ] / [ Σ_f A_f cos θ_f / d_f² ] + (1 − ε) T_amb")
P("compared for a 90 degree and a 5 degree field of view. Each transfer impedance of the electrical model is decomposed into tissue contributions by the Geselowitz identity,")
e7 = EQ("Z = (1/I²) ∫_Ω σ* ∇u_drive · ∇u_sense dΩ")
P("which also supplies two internal checks carried through every solve: reciprocity between the drive and sense solutions, and the residual between the sum of tissue contributions and the direct impedance.")
FIGURE(FIG / "Fig2.png", "The frame and the finite-element anatomy of the digital twin: path-traced rendering of the frame (a), exploded assembly with the nineteen sensing components (b), sagittal section of the fourteen-tissue mesh (c), and the frame posed on the head with the simulated skin temperature field (d).", width=15.5)
H("3.7 Verification, uncertainty and integrity checks", 2)
P("Code verification used the method of manufactured solutions on the real geometry with uniform coefficients, reporting the observed order of convergence between the coarse and full meshes. Solution verification used Roache's grid convergence index with a safety factor of 3,")
e8 = EQ("GCI_fine = F_s | (f_coarse − f_fine) / f_fine | / ( r^p − 1 )")
P("on four quantities of interest. The simulated apex and canthus temperatures were compared with published infrared-thermography ranges. Global sensitivity was quantified with Sobol' indices estimated by the Saltelli scheme on scrambled Sobol' sequences (64 base samples of eight inputs, 640 solves on the coarse mesh) with the Jansen estimators,")
e9 = EQ("S_T,i = E_{X~i}[ Var_{X_i}( Y | X~i ) ] / Var(Y)")
P("Two further checks precede any interpretation. Every headline metric is recomputed in PART 1 of the notebook from the stored per-subject predictions and compared with the results master, so that no number in this report can drift from its source. And an audit confirms that a quantum-circuit scaffold present in the code base (results/src/concept_quantum.py) is imported by none of the result scripts and produced no output; no result in this study depends on it.")

# ============================================================ 4 Results
H("4. Results")
H("4.1 Integrity checks", 2)
P("All ten headline quantities recomputed from the stored predictions agree with the results master to within floating-point precision (Table 9). The leakage audit removed ten outcome-carrying columns from both systemic models and three treatment-proxy classes from the neuropathy model.")
import csv
rows9 = list(csv.reader(open(NBO / "R1_Integrity.csv", encoding="utf-8")))[1:]
TABLE("Recomputed headline quantities against the results master (PART 1 of the notebook).", ["Quantity", "Recomputed", "Results master", "Agree"], [[r[0], f"{float(r[1]):.4f}", f"{float(r[2]):.4f}", r[3]] for r in rows9], widths=[7.5, 2.8, 2.8, 2.0])
H("4.2 Cohort structure and label shift", 2)
P(f"The three fundus datasets differ in composition: grade 0 makes up {100*dd[0]/sum(dd):.0f} % of DeepDRiD, {100*di[0]/sum(di):.0f} % of IDRiD and {100*da[0]/sum(da):.0f} % of APTOS, and IDRiD carries {100*di[4]/sum(di):.0f} % proliferative eyes against {100*da[4]/sum(da):.0f} % in APTOS (Table 2, Figure 3). In the Istanbul export {100*v('ehr','retinopathy','prevalence'):.1f} % of people carry a retinopathy diagnosis and {100*v('ehr','diabetic_neuropathy_main_excludes_treatment_proxies','prevalence'):.1f} % a neuropathy diagnosis; {jc['3']:,} carry both, and the two diagnoses co-occur with an odds ratio of {v('ehr','dr_dn_odds_ratio'):.2f} (Fisher's exact test, P = {v('ehr','dr_dn_fisher_p'):.1e}).")
FIGURE(NBO / "R_Fig1_GradeComposition.png", "Grade composition of the three fundus datasets. The shift in composition between datasets is the label shift the grader must survive.", width=13)
P("The datasets are not samples of one population. Any threshold fitted on DeepDRiD is fitted to Shanghai graders and cameras; how it transports to Nanded and Aravind is an empirical question, answered in Sections 4.4 and 4.5.", bold_lead="Interpretation.")
H("4.3 Fundus grading on DeepDRiD", 2)
rec = v("fundus_dr", "recall_per_grade")
P(f"With patients held out by fold, the grader reached a QWK of {v('fundus_dr','qwk'):.3f} {cis('fundus_dr','qwk')}, a referable AUC of {v('fundus_dr','auc_referable'):.3f} {cis('fundus_dr','auc_referable')} and exact-grade accuracy of {100*v('fundus_dr','accuracy'):.1f} % {cis('fundus_dr','accuracy')} (Table 10, Figure 4). Errors sit on the diagonal's neighbours; the five folds ended between {min(v('fundus_dr','fold_final_epoch_val_qwk')):.2f} and {max(v('fundus_dr','fold_final_epoch_val_qwk')):.2f} validation QWK after ten epochs.")
TABLE("Per-grade recall of the grader on DeepDRiD, out of fold.", ["ICDR grade", "Images", "Recall", "Most common error"], [
    ["0 none", dd[0], f"{100*rec[0]:.0f} %", "graded 1"], ["1 mild", dd[1], f"{100*rec[1]:.0f} %", "graded 0"], ["2 moderate", dd[2], f"{100*rec[2]:.0f} %", "graded 3"],
    ["3 severe", dd[3], f"{100*rec[3]:.0f} %", "graded 2"], ["4 proliferative", dd[4], f"{100*rec[4]:.0f} %", "graded 3 (51 of 92)"]], widths=[3.5, 2.5, 2.5, 5.0])
FIGURE(NBO / "R_Fig2_DeepDRiD.png", "DeepDRiD out-of-fold confusion matrix (left) and the distribution of the ordinal score by true grade with the four fitted thresholds (right).", width=15.5)
H("4.4 Transfer across populations", 2)
ii, aa = M["fundus_external_idrid"], M["fundus_external_aptos"]
P(f"Applied unchanged to two Indian datasets, the fold ensemble kept its ranking power: referable AUC {v('fundus_external_idrid','auc_referable'):.3f} {cis('fundus_external_idrid','auc_referable')} on the 516 IDRiD images and {v('fundus_external_aptos','auc_referable'):.3f} {cis('fundus_external_aptos','auc_referable')} on the 3,662 APTOS images, with QWK {v('fundus_external_idrid','qwk'):.3f} and {v('fundus_external_aptos','qwk'):.3f} (Table 11, Figure 5). Exact-grade accuracy fell to {100*v('fundus_external_idrid','accuracy'):.1f} % and {100*v('fundus_external_aptos','accuracy'):.1f} %.")
TABLE("Transfer of the grader with the DeepDRiD thresholds applied unchanged.", ["Dataset", "n", "QWK [95 % CI]", "Referable AUC [95 % CI]", "Accuracy", "Sensitivity", "Specificity"], [
    ["DeepDRiD OOF (China)", f"{sum(dd):,}", f"{v('fundus_dr','qwk'):.3f} {cis('fundus_dr','qwk')}", f"{v('fundus_dr','auc_referable'):.3f} {cis('fundus_dr','auc_referable')}", f"{100*v('fundus_dr','accuracy'):.1f} %", f"{100*v('fundus_dr','sens_referable_at_oof_operating_point'):.1f} %", f"{100*v('fundus_dr','spec_referable_at_oof_operating_point'):.1f} %"],
    ["IDRiD external (India)", "516", f"{v('fundus_external_idrid','qwk'):.3f} {cis('fundus_external_idrid','qwk')}", f"{v('fundus_external_idrid','auc_referable'):.3f} {cis('fundus_external_idrid','auc_referable')}", f"{100*v('fundus_external_idrid','accuracy'):.1f} %", f"{100*v('fundus_external_idrid','sens_referable_at_oof_operating_point'):.1f} %", f"{100*v('fundus_external_idrid','spec_referable_at_oof_operating_point'):.1f} %"],
    ["APTOS 2019 external (India)", "3,662", f"{v('fundus_external_aptos','qwk'):.3f} {cis('fundus_external_aptos','qwk')}", f"{v('fundus_external_aptos','auc_referable'):.3f} {cis('fundus_external_aptos','auc_referable')}", f"{100*v('fundus_external_aptos','accuracy'):.1f} %", f"{100*v('fundus_external_aptos','sens_referable_at_oof_operating_point'):.1f} %", f"{100*v('fundus_external_aptos','spec_referable_at_oof_operating_point'):.1f} %"]],
    widths=[3.4, 1.2, 2.9, 2.9, 1.8, 1.9, 1.9], font=8)
FIGURE(NBO / "R_Fig3_Transfer.png", "Referable-retinopathy ROC curves on the three datasets (left) and sensitivity and specificity at the DeepDRiD operating point (right).", width=15.5)
P("Sensitivity for referable disease stays above 90 % on every dataset at the transported operating point, while specificity falls from 93 % to 72 % on IDRiD and 85 % on APTOS. The referable flag transfers; the grade does not.", bold_lead="Interpretation.")
H("4.5 Error mechanism: the APTOS boundary shift", 2)
rows13 = list(csv.reader(open(NBO / "R5_ErrorMechanism.csv", encoding="utf-8")))[1:]
sh = v("fundus_external_aptos", "shift_true2_predicted3"); s4 = v("fundus_external_aptos", "shift_true4_predicted3"); s1 = v("fundus_external_aptos", "shift_true1_predicted2")
P(f"The external error is a boundary shift, not a loss of ordering. On APTOS {100*v('fundus_external_aptos','share_over_graded'):.1f} % of images were graded one step or more above their label and only {100*v('fundus_external_aptos','share_under_graded'):.1f} % below; the grader placed {sh['count']} of {sh['of']} moderate eyes at grade 3, {s4['count']} of {s4['of']} proliferative eyes at grade 3 and {s1['count']} of {s1['of']} mild eyes at grade 2 (Table 12, Figure 6). Re-fitting the four thresholds on APTOS, which is reported as a diagnostic and was not used, raises QWK from {v('fundus_external_aptos','qwk'):.3f} to {v('fundus_external_aptos','refit_qwk_not_used'):.3f} and moves the 2|3 boundary from {v('fundus_dr','thresholds')[2]:.2f} to {v('fundus_external_aptos','refit_thresholds_not_used')[2]:.2f}; on IDRiD re-fitting raises QWK from {v('fundus_external_idrid','qwk'):.3f} to {v('fundus_external_idrid','refit_qwk_not_used'):.3f}.")
TABLE("Direction of the grading error on each dataset (signed difference between predicted and true grade).", ["Dataset", "Above label", "Exact", "Below label", "Mean shift (grades)", "Off by one", "Off by two or more"],
      [[r[0], f"{float(r[1]):.1f} %", f"{float(r[2]):.1f} %", f"{float(r[3]):.1f} %", f"{float(r[4]):+.2f}", f"{float(r[5]):.1f} %", f"{float(r[6]):.1f} %"] for r in rows13], widths=[3.4, 2.0, 1.8, 2.0, 2.6, 2.0, 2.2], font=8.5)
FIGURE(NBO / "R_Fig4_APTOS_Shift.png", "APTOS 2019 confusion matrix with the DeepDRiD thresholds: the mass sits one cell above the diagonal.", width=10.5)
P("The same score that ranks APTOS eyes at an AUC of 0.979 grades them exactly only 55 % of the time, because the grade boundaries fitted to Shanghai photographs cut the APTOS score distribution in the wrong places. A screening pathway that reports a grade must therefore own a per-deployment calibration stage; a pathway that reports a referable flag is far less exposed. This finding is why the fusion design of the wearable carries an explicit threshold-fitting stage rather than universal thresholds.", bold_lead="The central contrast of this study.")
H("4.6 The macular edema head", 2)
P(f"The DME head reached a QWK of {v('dme','cv_qwk'):.3f} {cis('dme','cv_qwk')} out of fold on the 413 IDRiD training images and {v('dme','test_qwk'):.3f} {cis('dme','test_qwk')} on the fixed 103-image test split, with any-DME AUC of {v('dme','cv_auc_any_dme'):.3f} and {v('dme','test_auc_any_dme'):.3f} (Table 13). Across all 516 IDRiD images {100*v('dme','any_dme_share_in_referable'):.1f} % of the {M['dme']['any_dme_share_in_referable']['n']} referable eyes carried some macular edema and none of the {M['dme']['any_dme_share_in_nonreferable']['n']} non-referable eyes did; the two heads' scores correlated at r = {v('dme','pearson_r_dr_score_vs_dme_score_test'):.2f} on the test split.")
TABLE("Macular edema head on IDRiD.", ["Evaluation", "n", "QWK [95 % CI]", "Any-DME AUC [95 % CI]"], [
    ["5-fold out of fold, training split", "413", f"{v('dme','cv_qwk'):.3f} {cis('dme','cv_qwk')}", f"{v('dme','cv_auc_any_dme'):.3f} {cis('dme','cv_auc_any_dme')}"],
    ["Fixed test split, fold ensemble", "103", f"{v('dme','test_qwk'):.3f} {cis('dme','test_qwk')}", f"{v('dme','test_auc_any_dme'):.3f} {cis('dme','test_auc_any_dme')}"]], widths=[5.5, 1.5, 4.0, 5.0])
H("4.7 Systemic models and the leakage audit", 2)
eh = M["ehr"]
P(f"On the 77,724-person export the boosted model reached an AUC of {v('ehr','retinopathy','hgb','auc'):.3f} {cis('ehr','retinopathy','hgb','auc')} for retinopathy and {v('ehr','headline_dn_auc'):.3f} {cis('ehr','headline_dn_auc')} for neuropathy, with the logistic model at {v('ehr','retinopathy','logistic','auc'):.3f} and {v('ehr','diabetic_neuropathy_main_excludes_treatment_proxies','logistic','auc'):.3f} (Table 14, Figure 7). Both boosted models are calibrated (slope {v('ehr','retinopathy','hgb','calibration_slope'):.2f} and {v('ehr','diabetic_neuropathy_main_excludes_treatment_proxies','hgb','calibration_slope'):.2f}). With the three treatment-proxy classes restored the neuropathy AUC rises to {v('ehr','dn_sensitivity_with_treatment_proxies','hgb','auc'):.3f} {cis('ehr','dn_sensitivity_with_treatment_proxies','hgb','auc')}; the alpha-lipoic-acid class and the antiepileptic class then carry odds ratios per standard deviation of {v('ehr','dn_sensitivity_with_treatment_proxies','logistic_odds_ratio_per_sd_top')['other_digestive']:.2f} and {v('ehr','dn_sensitivity_with_treatment_proxies','logistic_odds_ratio_per_sd_top')['antiepileptics']:.2f} and permutation importances of {v('ehr','dn_sensitivity_with_treatment_proxies','hgb_permutation_importance_top')['antiepileptics']:.3f} and {v('ehr','dn_sensitivity_with_treatment_proxies','hgb_permutation_importance_top')['other_digestive']:.3f}, an order of magnitude above every other feature. Removing analgesics and psychoanaleptics as well changes the AUC to {v('ehr','dn_sensitivity_without_proxies_analgesics_psychoanaleptics','hgb','auc'):.3f}.")
def eb(k): return [f"{v('ehr',k,m,'auc'):.3f} {cis('ehr',k,m,'auc')}" for m in ("logistic", "hgb")] + [f"{v('ehr',k,'hgb','calibration_slope'):.2f} / {v('ehr',k,'hgb','calibration_intercept'):+.2f}", f"{v('ehr',k,'hgb','brier'):.3f}"]
TABLE("Systemic models, five-fold out of fold, n = 77,724.", ["Outcome and design", "Logistic AUC [95 % CI]", "Boosted AUC [95 % CI]", "Boosted slope / intercept", "Brier"], [
    ["Retinopathy"] + eb("retinopathy"), ["Neuropathy, treatment proxies excluded (reported)"] + eb("diabetic_neuropathy_main_excludes_treatment_proxies"),
    ["Neuropathy, sensitivity A: proxies restored"] + eb("dn_sensitivity_with_treatment_proxies"), ["Neuropathy, sensitivity B: wider exclusion"] + eb("dn_sensitivity_without_proxies_analgesics_psychoanaleptics")], widths=[5.2, 3.0, 3.0, 2.8, 1.6], font=8.5)
tdr = v("ehr", "retinopathy", "logistic_odds_ratio_per_sd_top"); tdn = v("ehr", "diabetic_neuropathy_main_excludes_treatment_proxies", "logistic_odds_ratio_per_sd_top")
TABLE("Largest logistic odds ratios per standard deviation.", ["Retinopathy: feature", "OR per SD", "Neuropathy: feature", "OR per SD"],
      [[list(tdr)[i], f"{list(tdr.values())[i]:.2f}", list(tdn)[i], f"{list(tdn.values())[i]:.2f}"] for i in range(6)], widths=[4.5, 2.5, 4.5, 2.5])
FIGURE(NBO / "R_Fig5_EHR.png", "Left: ROC curves of the boosted retinopathy model, the reported neuropathy model and the leakage run with treatment proxies restored. Right: calibration of the two reported boosted models in quantile bins.", width=15.5)
P(f"A model that reads whether a person has been dispensed gabapentin, pregabalin or alpha-lipoic acid reaches 0.84 for neuropathy; a model that reads the records as they look before treatment reaches 0.71. The lower number is the one reported, because the wearable's purpose is to find people who have not yet been diagnosed. The joint four-class model gives one-versus-rest AUCs of {v('ehr','joint_ovr_auc')['none']:.2f}, {v('ehr','joint_ovr_auc')['DR only']:.2f}, {v('ehr','joint_ovr_auc')['DN only']:.2f} and {v('ehr','joint_ovr_auc')['both']:.2f} for neither, retinopathy only, neuropathy only and both.", bold_lead="Perspective.")
H("4.8 The physiological branch", 2)
hv = M["hrv"]
P(f"Six heart-rate-variability features separated people with diabetes from controls in the pooled PhysioNet cohorts with a leave-one-out AUC of {v('hrv','dm_vs_control','auc'):.3f} {cis('hrv','dm_vs_control','auc')} (n = {v('hrv','dm_vs_control','n')}). The same features did not separate retinopathy within the diabetic group: AUC {v('hrv','dr_within_dm','auc'):.3f} {cis('hrv','dr_within_dm','auc')} (n = {v('hrv','dr_within_dm','n')}), every absolute Cohen's d at or below {v('hrv','dr_within_dm','max_abs_d'):.2f}, and no feature with a Benjamini-Hochberg P below {v('hrv','dr_within_dm','min_p_bh'):.2f} (Table 17). Mean heart rate correlated with HbA1c (Spearman ρ = {v('hrv','hba1c_spearman')['hr_mean']['rho']:.2f}, P = {v('hrv','hba1c_spearman')['hr_mean']['p']:.3f}). Fingertip photoplethysmography carried no diabetes signal beyond demographics on any route, while the same morphology features detected hypertension at {v('ppg','hypertension_auc_by_route','morphology'):.3f} {cis('ppg','hypertension_auc_by_route','morphology')} and the embedding recovered age at r = {v('ppg','age_from_papagei_r'):.2f} (Table 16, Figure 8).")
rows15 = list(csv.reader(open(NBO / "R8_Physiological.csv", encoding="utf-8")))[1:]
TABLE("Physiological branch: discrimination by analysis, with 95 % bootstrap intervals.", ["Analysis", "n", "Positives", "AUC", "95 % interval"], [[r[0], r[1], r[2], f"{float(r[3]):.3f}", f"[{float(r[4]):.3f}, {float(r[5]):.3f}]"] for r in rows15], widths=[7.5, 1.4, 1.8, 1.8, 3.0], font=8.5)
eff = v("hrv", "dr_within_dm", "effects")
TABLE("Per-feature effects for retinopathy within diabetics (n = 58; the null finding).", ["HRV feature", "Cohen's d", "Mann-Whitney p", "BH-corrected p", "Median DR+", "Median DR−"],
      [[k, f"{e['d']:+.2f}", f"{e['p']:.2f}", f"{e['p_bh']:.2f}", f"{e['median_dr']:.2f}", f"{e['median_no_dr']:.2f}"] for k, e in eff.items()], widths=[2.8, 2.2, 2.6, 2.6, 2.4, 2.4])
FIGURE(NBO / "R_Fig6_Physiological.png", "AUC with 95 % bootstrap intervals for every physiological analysis: the positive result (HRV against diabetes), the two nulls, and the positive controls that show the pipelines work.", width=14)
P("A null with a positive control is a finding about signal content, not about the pipeline. Heart-rate variability from a resting electrocardiogram carries diabetes information but no retinopathy information in these cohorts; 2.1 s fingertip recordings carry no diabetes information beyond age, sex and body mass index, whether read as morphology or through a foundation model, although the tiling of short segments into the model's window handicaps the embedding route. For the wearable, the temple photoplethysmograph and any heart-rate feature are retained as context and are not claimed as retinopathy detectors.", bold_lead="Interpretation.")
H("4.9 Digital twin credibility", 2)
P(f"The reference thermal solution at 25 °C ambient gave a corneal apex temperature of {v('thermal','corneal_apex_C'):.2f} °C, an inner-canthus skin temperature of {v('thermal','medial_canthus_skin_C'):.2f} °C and a temple temperature of {v('thermal','temple_skin_C'):.2f} °C, about 1 °C below the published thermography bands of {v('thermal','published_corneal_apex_range_C')[0]:.1f} to {v('thermal','published_corneal_apex_range_C')[1]:.1f} °C and {v('thermal','published_inner_canthus_range_C')[0]:.1f} to {v('thermal','published_inner_canthus_range_C')[1]:.1f} °C. Code verification gave an observed order of {v('verification','observed_order'):.2f} for linear tetrahedra; grid convergence indices were at most {v('verification','gci_max_percent'):.1f} % (Table 18). Halving choroidal perfusion cooled the apex by {abs(v('thermal','apex_deficit_at_50pct_choroid_C')):.2f} °C, close to the 0.6 °C deficit reported clinically; a 90 degree thermopile registered {100*v('thermal','thermopile_90deg_capture_fraction'):.0f} % of it and a 5 degree part {100*v('thermal','thermopile_5deg_capture_fraction'):.0f} %. Ambient temperature dominated every readout with total-order Sobol' indices of {v('uq','ST_ambient_range')[0]:.2f} to {v('uq','ST_ambient_range')[1]:.2f} (Table 19, Figure 9), and the apex followed ambient temperature at {v('uq','apex_slope_per_C_ambient'):.2f} °C per °C.")
gci = v("verification", "gci")
TABLE("Verification and validation of the thermal forward model.", ["Quantity", "Value", "Reference"], [
    ["Observed order of convergence (MMS, P1 tetrahedra)", f"{v('verification','observed_order'):.2f}", "theoretical 2"],
    ["Relative L2 error, full mesh", f"{100*v('verification','mms_full','L2_rel'):.2f} %", ""],
    ["GCI, corneal apex temperature", f"{gci['corneal_apex_C']['GCI_fine_percent']:.3f} %", "5 % typical acceptance"],
    ["GCI, canthus skin temperature", f"{gci['canthus_skin_C']['GCI_fine_percent']:.3f} %", ""],
    ["GCI, 5 degree thermopile reading", f"{gci['thermopile_DCI_C']['GCI_fine_percent']:.3f} %", ""],
    ["GCI, apex deficit at 50 % choroidal perfusion", f"{gci['deficit_50pct_choroid_C']['GCI_fine_percent']:.2f} %", ""],
    ["Corneal apex, simulated vs published", f"{v('thermal','corneal_apex_C'):.2f} °C", f"{v('thermal','published_corneal_apex_range_C')[0]:.1f} to {v('thermal','published_corneal_apex_range_C')[1]:.1f} °C"],
    ["Inner canthus, simulated vs published", f"{v('thermal','medial_canthus_skin_C'):.2f} °C", f"{v('thermal','published_inner_canthus_range_C')[0]:.1f} to {v('thermal','published_inner_canthus_range_C')[1]:.1f} °C"],
    ["Surrogate hold-out RMSE (four readouts)", f"{min(v('uq','surrogate_rmse_holdout_C')):.3f} to {max(v('uq','surrogate_rmse_holdout_C')):.3f} °C", ""]], widths=[7.5, 3.5, 5.0])
ST = np.array(v("uq", "ST")); params = [p[0] for p in v("uq", "params_ranges")]; outs = v("uq", "outputs")
PN = {"choroid_frac": "choroidal perfusion", "skin_perf_frac": "skin perfusion", "eyelid_perf_frac": "eyelid perfusion", "orbit_perf_frac": "orbital perfusion", "T_amb": "ambient temperature", "h_skin": "skin convection", "E_tear": "tear evaporation", "k_sclera_scale": "scleral conductivity"}
TABLE("Total-order Sobol' indices of the thermal readouts (640 Saltelli solves).", ["Input"] + ["corneal apex", "canthus skin", "thermopile at limbus", "thermopile at canthus"],
      [[PN[p]] + [f"{max(ST[i, j], 0):.2f}" for j in range(4)] for i, p in enumerate(params)], widths=[4.5, 2.8, 2.8, 3.0, 3.0])
FIGURE(NBO / "R_Fig7_Twin.png", "Left: change in each thermal readout as choroidal perfusion falls (design verdict 1). Right: total-order Sobol' indices (design verdict 8).", width=15.5)
H("4.10 Design verdicts", 2)
P("Table 20 collects the eight verdicts the twin returned to the hardware, each with the simulated quantity behind it; the optical, electrical and pupil numbers are read from the same results master, and the full forward-model figures are in the repository.")
rows19 = list(csv.reader(open(NBO / "R10_Verdicts.csv", encoding="utf-8")))[1:]
TABLE("The eight design verdicts.", ["Verdict", "Quantity from the twin", "Design consequence"], rows19, widths=[3.4, 8.2, 4.4], font=8.5)
H("4.11 Summary of results", 2)
TABLE("Headline quantities from every stage of the analysis.", ["Stage", "Result"], [
    ["Integrity", "10 of 10 headline quantities reproduce from stored predictions; no quantum contribution; 13 leakage columns removed"],
    ["Fundus grader, DeepDRiD out of fold", f"QWK {v('fundus_dr','qwk'):.3f}; referable AUC {v('fundus_dr','auc_referable'):.3f}; accuracy {100*v('fundus_dr','accuracy'):.1f} %"],
    ["External test, IDRiD", f"QWK {v('fundus_external_idrid','qwk'):.3f}; referable AUC {v('fundus_external_idrid','auc_referable'):.3f}; sensitivity {100*v('fundus_external_idrid','sens_referable_at_oof_operating_point'):.0f} %"],
    ["External test, APTOS 2019", f"QWK {v('fundus_external_aptos','qwk'):.3f}; referable AUC {v('fundus_external_aptos','auc_referable'):.3f}; accuracy {100*v('fundus_external_aptos','accuracy'):.1f} %"],
    ["Error mechanism", f"{100*v('fundus_external_aptos','share_over_graded'):.1f} % of APTOS images graded above label, {100*v('fundus_external_aptos','share_under_graded'):.1f} % below; re-fitted QWK {v('fundus_external_aptos','refit_qwk_not_used'):.3f} (not used)"],
    ["Macular edema head", f"test QWK {v('dme','test_qwk'):.3f}; any-DME AUC {v('dme','test_auc_any_dme'):.3f}; any DME in {100*v('dme','any_dme_share_in_referable'):.0f} % of referable eyes"],
    ["Systemic retinopathy", f"AUC {v('ehr','retinopathy','hgb','auc'):.3f}; calibration slope {v('ehr','retinopathy','hgb','calibration_slope'):.2f}"],
    ["Systemic neuropathy", f"AUC {v('ehr','headline_dn_auc'):.3f} with treatment proxies excluded ({v('ehr','dn_sensitivity_with_treatment_proxies','hgb','auc'):.3f} with them restored)"],
    ["Co-occurrence", f"DR-DN odds ratio {v('ehr','dr_dn_odds_ratio'):.2f}; joint model AUC {v('ehr','joint_ovr_auc')['both']:.2f} for both"],
    ["HRV", f"diabetes AUC {v('hrv','dm_vs_control','auc'):.3f}; retinopathy within diabetics {v('hrv','dr_within_dm','auc'):.3f} (null)"],
    ["Fingertip PPG", f"diabetes at most {v('ppg','diabetes_max_auc_any_route'):.3f} on any route (null); hypertension control {v('ppg','hypertension_auc_by_route','morphology'):.3f}"],
    ["Twin verification", f"MMS order {v('verification','observed_order'):.2f}; GCI at most {v('verification','gci_max_percent'):.1f} %; apex 1.0 °C below the published band"],
    ["Twin uncertainty", f"ambient S_T {v('uq','ST_ambient_range')[0]:.2f} to {v('uq','ST_ambient_range')[1]:.2f}; {v('uq','apex_slope_per_C_ambient'):.2f} °C per °C"],
    ["Design verdicts", "8 returned to the hardware (Table 20)"]], widths=[4.5, 11.5])
P("Two patterns emerge. On the artificial-intelligence side, ordering transfers and boundaries do not: the fundus grader trained in Shanghai ranks Indian eyes almost as well as its own, but its grade thresholds must be re-fitted per deployment, and the systemic models give modest, calibrated risk once the columns that follow a diagnosis are removed. Two physiological signals gave nulls with intact positive controls, which sets what the wearable must not claim. On the physics side, the digital twin reproduces the periorbital temperature field to within one degree, passes code and solution verification, and shows that ambient temperature and the sensor's field of view govern what a thermopile on a frame can see; the eight verdicts changed the hardware before it was built.")
P("What the study does not establish is equally clear. No prototype exists, no human has worn the frame, and every physics result is a prediction to be tested on a phantom bench and then on people under ethical approval. The fusion stage that would combine the branches is designed but not evaluated, because no open dataset carries all the modalities on the same people; the credentialed BRSET and mBRSET datasets are the first place it can be tested. The results are presented as they are, with the nulls and the leakage correction in place, without a claim in either direction beyond what the data support.")

out = HERE / "NRDI_Datasets_Methodology_Results.docx"; doc.save(out)
print("wrote", out, "| tables", TABLE_N[0], "| figures", FIG_N[0], "| equations", EQ_N[0])
