# Figure tooling for a journal-quality medical-AI paper (Python, 2026-09)

Scope: static figures for IEEE Transactions / Elsevier / Nature-portfolio submission.
Figure types covered: confusion matrix, ROC + PR with bootstrap bands, calibration, forest plot (OR),
bar + bootstrap CI, Sobol bars, radar, multi-panel grid, Bland-Altman, SHAP beeswarm, Cohen-d dot plot,
training curves, one system-architecture diagram.

Environment verified on this machine: Python 3.14.3 (win_amd64). All package versions below were read
from the PyPI JSON API on 2026-09-07, and the full recommended stack was resolved with
`pip install --dry-run` on Python 3.14 (it resolves; wheel names are listed in Section 7).

---

## 1. Matplotlib style packages

| Package (pip name) | Version (PyPI) | Released | mpl 3.10/3.11 | Py 3.14 | LaTeX? |
|---|---|---|---|---|---|
| `matplotlib` | 3.11.1 | 2026 | n/a | yes, cp314 win_amd64 wheel | no |
| `SciencePlots` | 2.2.2 | 2026-06-23 | yes (2.2.2 specifically fixed mpl 3.11 `plt.style.core` refactor; use >=2.2.2) | yes (pure Python, requires >=3.8) | `science` sets `text.usetex=True`; add `'no-latex'` |
| `tueplots` | 0.2.4 | 2026-03-24 | yes | yes (pure Python, requires >=3.10) | ML-conference bundles default `usetex=True`; pass `usetex=False` |
| `seaborn` | 0.13.2 | 2024-01-25 | yes (`matplotlib>=3.4,!=3.6.1`; the `register_cmap` breakage was seaborn <0.13 on mpl 3.9) | yes (pure Python) | no |
| `plotly` + `kaleido` | 7.0.0 / 1.4.0 | 2026-08-25 / 2026-08-31 | n/a | yes (pure Python) | no, but kaleido >=1.0 needs a system Chrome/Chromium |

### SciencePlots (recommended base)
- Repo: https://github.com/garrettj403/SciencePlots ; changelog: https://github.com/garrettj403/SciencePlots/blob/master/CHANGES.md
- `import scienceplots` then `plt.style.use(['science', 'nature', 'no-latex'])` or `['science', 'ieee', 'no-latex']`.
  - `ieee`: serif (Times look-alike), narrow column widths, small fonts.
  - `nature`: sans-serif, matches Nature's Helvetica/Arial requirement.
  - Colour styles shipped (Paul Tol): `bright`, `muted`, `vibrant`, `high-contrast`, `light`, `high-vis`, `retro`, `discrete-rainbow-1 ... -23` (23 added in v2.2.0, 2025-11-20). Source files: https://github.com/garrettj403/SciencePlots/tree/master/src/scienceplots/styles/color
- LaTeX: the `science` style turns on `text.usetex`. Disable either by appending the `no-latex` style
  (`plt.style.use(['science','nature','no-latex'])`) or by `plt.rcParams['text.usetex'] = False` after `style.use`.
  v2.2.0 fixed symbol rendering (degree sign etc.) in `science + no-latex`.
- Caveat: `science` sets very small default figsize/fonts; override with the rcParams block in Section 7.

### tueplots (optional, for exact-width figsize helpers)
- Docs: https://tueplots.readthedocs.io/en/latest/ ; bundles API: https://tueplots.readthedocs.io/en/latest/docs_api/tueplots.bundles.html
- Bundles present in 0.2.4: `aaai2024, aistats2022/2023/2025, colm2026, cvpr2024, eccv2024, iclr2023/2024, icml2022/2024, jmlr2001, neurips2021-2024, probnum2025, tmlr2023, uai2023, tue_ai_thesis, beamer_*`.
  No IEEE / Elsevier / Nature bundle. Useful pieces are `tueplots.figsizes`, `tueplots.fontsizes`, `tueplots.axes` (e.g. `axes.lines()` for thin spines).
- `neurips*/icml*/iclr*/cvpr2024/colm2026` default to `usetex=True`: call `bundles.neurips2024(usetex=False, family='sans-serif')`.
- Usage: `with plt.rc_context({**bundles.neurips2024(usetex=False), **axes.lines()}): ...`
- Verdict: not required for journal figures; SciencePlots + a hand-written rcParams block is simpler.

### seaborn
- Docs: https://seaborn.pydata.org/ ; PyPI: https://pypi.org/project/seaborn/
- Use only for `swarmplot`/`stripplot`/`boxplot` primitives. Do not call `sns.set_theme()` after your rcParams block (it overwrites fonts/sizes); if you want it, call `sns.set_theme(style='ticks', context='paper', rc={})` first, then apply your rcParams.

### plotly + kaleido (not recommended for this paper)
- https://plotly.com/python/static-image-export/ ; https://github.com/plotly/Kaleido
- kaleido >=1.0 no longer bundles Chromium; it searches PATH for Chrome/Chromium, or run `plotly_get_chrome` / `plotly.io.get_chrome()`. Requires plotly >=6.1.1.
- Font embedding in plotly PDF/SVG output is less controllable than matplotlib; mixing plotly and matplotlib figures in one paper produces visibly different typography. Skip unless a figure already exists in plotly.

### Excluded
- `mplcyberpunk` (glow effects; not journal style), `latexplotlib` (requires LaTeX toolchain by design).

### Built-in matplotlib extras worth knowing
- `plt.style.use('petroff10')`: colour-vision-deficiency-tested 10-colour cycle from Petroff (2021), shipped in matplotlib's stylelib (https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/mpl-data/stylelib/petroff10.mplstyle). Use Okabe-Ito or Tol instead when you need <=8 categories; `petroff10` when you need 9-10.

---

## 2. Colour

### Okabe-Ito (8 colours, CVD-safe; use for lines/markers, up to 7 series + black)
Source table: Wilke, *Fundamentals of Data Visualization*, ch. 19 (https://clauswilke.com/dataviz/color-pitfalls.html); original: Okabe & Ito 2008 (https://jfly.uni-koeln.de/color/).

| Name | Hex | RGB |
|---|---|---|
| Orange | `#E69F00` | 230,159,0 |
| Sky blue | `#56B4E9` | 86,180,233 |
| Bluish green | `#009E73` | 0,158,115 |
| Yellow | `#F0E442` | 240,228,66 |
| Blue | `#0072B2` | 0,114,178 |
| Vermilion | `#D55E00` | 213,94,0 |
| Reddish purple | `#CC79A7` | 204,121,167 |
| Black | `#000000` | 0,0,0 |

Practical order for a medical paper (dark first, yellow last): `#0072B2, #D55E00, #009E73, #CC79A7, #E69F00, #56B4E9, #F0E442, #000000`.

### Paul Tol qualitative schemes
Primary source (site moved): https://sronpersonalpages.nl/~pault/ (the old `personal.sron.nl` host no longer resolves). Hex codes cross-checked against SciencePlots' `bright.mplstyle` / `muted.mplstyle`.

- **Bright** (7, for lines and labels): `#4477AA` blue, `#EE6677` red, `#228833` green, `#CCBB44` yellow, `#66CCEE` cyan, `#AA3377` purple, `#BBBBBB` grey.
- **Muted** (9 + pale grey): `#CC6677` rose, `#332288` indigo, `#DDCC77` sand, `#117733` green, `#88CCEE` cyan, `#882255` wine, `#44AA99` teal, `#999933` olive, `#AA4499` purple, `#DDDDDD` pale grey (reserve for missing/excluded data).
- **High-contrast** (3 + black/white, survives greyscale printing): `#004488` blue, `#BB5566` red, `#DDAA33` yellow, `#000000`, `#FFFFFF`.
- **Vibrant** (7): `#EE7733` orange, `#0077BB` blue, `#33BBEE` cyan, `#EE3377` magenta, `#CC3311` red, `#009988` teal, `#BBBBBB` grey.

### Continuous / diverging colormaps: `cmcrameri` 1.10 (released 2026-08-04)
- https://github.com/callumrollo/cmcrameri ; bundles Crameri Scientific Colour Maps 8.0.
- `import cmcrameri.cm as cmc` then `cmap=cmc.batlow` (sequential; confusion matrix, heatmaps), `cmc.vik` or `cmc.roma` (diverging; SHAP feature-value colouring, Cohen-d sign), `cmc.batlowS` (categorical), `_r` suffix reverses.
- Avoid `jet`/`rainbow`. Matplotlib's `viridis`/`cividis` are also acceptable; `cividis` is CVD-optimised.

### Rules
- Colour must never be the only channel: pair with marker shape / line style (IEEE explicitly asks for this: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/).
- Submit RGB, not CMYK (Nature: RGB recommended, converted at print).
- Confidence bands: same hue as the line at `alpha=0.2`, `linewidth=0`.

---

## 3. Typography and sizing by publisher

| | Nature portfolio | IEEE Transactions | Elsevier |
|---|---|---|---|
| Single column | 89 mm | 3.5 in = 88.9 mm = 21 pica | 90 mm |
| 1.5 column | 120 mm or 136 mm | (no formal 1.5 col; use 1 or 2) | 140 mm |
| Double column | 183 mm | 7.16 in = 182 mm = 43 pica | 190 mm |
| Max height | 247 mm page depth (figure guide: keep <=170 mm to leave legend room) | not stated | not stated |
| Text size | 5-7 pt (min 5, max 7 for non-panel text) | approx. 9-10 pt at final size, consistent across figures | 7 pt normal text; >=6 pt sub/superscript |
| Panel labels | **8 pt bold, upright, lowercase a, b, c** | (a), (b), (c) lowercase in parentheses, placed as sub-captions (IEEEtran convention) | Not mandated publisher-wide; check the journal's Guide for Authors (uppercase A, B common in biomedical titles) |
| Fonts | Sans-serif: Helvetica or Arial; Courier for sequences | Helvetica, Times New Roman, Arial, Cambria, Symbol | Arial/Helvetica, Courier, Symbol, Times/Times New Roman (Type 1 or TrueType) |
| Vector formats | AI, EPS, PDF preferred; SVG acceptable; **do not outline text**; embed fonts | PS, EPS, PDF; embed fonts or convert to outlines | EPS preferred; PDF accepted |
| Raster | Photos >=300 dpi; JPEG/TIFF/PNG **not accepted** for main vector figures | >300 dpi colour/greyscale, >600 dpi line art; PNG/TIFF accepted | TIFF: 300 dpi halftone, 500 dpi combination, 1000 dpi line art |
| Colour space | RGB (auto-converted to CMYK for print) | not stated | RGB accepted |
| File naming | Fig1.pdf etc. | first 5 letters of surname + number, e.g. `goenk1.pdf` | per journal |

Sources:
- Nature final artwork guide (PDF): https://www.nature.com/documents/nature-final-artwork.pdf (89/120-136/183 mm; Helvetica/Arial; 5-7 pt; 8 pt bold upright a, b, c; RGB; no outlined text)
- Nature research figure guide: https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/ (89/183 mm, <=170 mm height, 5-7 pt, vector AI/EPS/PDF, no JPEG/TIFF/PNG for main figures)
- Nature formatting guide (page requires cookies; states 90/180 mm and "lower-case type, first letter capitalised, no full stop" for figure lettering): https://www.nature.com/nature/for-authors/formatting-guide
- IEEE resolution and size: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/
- IEEE file formatting (fonts, 9-10 pt, embed/outline, naming): https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/file-formatting/
- Elsevier artwork sizing (90/140/190 mm; 7 pt; 300/500/1000 dpi): https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing
- Elsevier artwork overview (formats, fonts): https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-overview

Practical decision: design every figure at **88 mm** (single) or **180 mm** (double) width with **7 pt** body text and **8 pt bold** panel labels. That satisfies Nature exactly, is within 1-2 mm of IEEE/Elsevier, and only IEEE's "9-10 pt" note is softer (IEEE accepts smaller in practice; bump `font.size` to 8 for an IEEE submission). Export **PDF (fonttype 42)** as master plus **600 dpi TIFF/PNG** fallbacks.

---

## 4. Libraries per figure type

| Figure | Library / API | Notes |
|---|---|---|
| Confusion matrix | `sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y, yhat, normalize='true', cmap=cmc.batlow, colorbar=False, ax=ax)` | Annotate counts and row-% together via `values_format`; use `layout='compressed'` for grids of square panels. |
| ROC / PR curves | `sklearn.metrics.RocCurveDisplay.from_predictions(..., plot_chance_level=True, ax=ax)`; `RocCurveDisplay.from_cv_results(cv_results, X, y, curve_kwargs=...)` (new in scikit-learn 1.7); `PrecisionRecallDisplay.from_predictions` | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html |
| Bootstrap CI band on ROC | hand-rolled (see snippet) or `pauc` 0.2.2 (DeLong + bootstrap CI, plotting; https://pypi.org/project/pauc/), `MLstatkit` 0.1.91 (DeLong test between two AUCs, bootstrap CIs for AUC/F1/PR-AUC; https://pypi.org/project/MLstatkit/), `confidenceinterval` 1.0.5 (analytic + bootstrap CIs for classification metrics; https://pypi.org/project/confidenceinterval/) | Hand-rolled is 12 lines and keeps full styling control; use `MLstatkit` for the DeLong p-value in the text. |
| Calibration | `sklearn.calibration.CalibrationDisplay.from_predictions(y, p, n_bins=10, strategy='quantile', ref_line=True, ax=ax)` | https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html ; add a histogram of `p` beneath as a second axes; report Brier score / ECE in legend. |
| Forest plot (OR) | `forestplot` 0.4.1: `fp.forestplot(df, estimate='or', ll='ci_low', hl='ci_high', varlabel='label', groupvar='group', annote=[...], xlabel='Odds ratio', logscale=True)`; `fp.mforestplot` for multi-model | https://github.com/LSYS/forestplot ; https://forestplot.readthedocs.io/ . **Caveat**: last release 2024-07-28 and it pins `matplotlib-inline<=0.1.3`, which downgrades IPython's `matplotlib-inline`. Install with `pip install --no-deps forestplot` (it only needs pandas/numpy/matplotlib). For full typographic control a hand-rolled `ax.errorbar(x, y, xerr=[lo, hi], fmt='s', capsize=2)` on a log x-axis with `ax.axvline(1)` is often cleaner. |
| Bar chart + bootstrap CI | matplotlib `ax.bar(..., yerr=ci, capsize=2, error_kw={'lw':0.6})`; CIs from `scipy.stats.bootstrap` (BCa, `n_resamples=10000`) | Prefer dot-and-CI over bars when the baseline is not zero. |
| Sobol sensitivity | `SALib` 1.5.2: `Si = sobol.analyze(problem, Y)`; bar `Si['S1']`, `Si['ST']` with `yerr=Si['S1_conf']`, `Si['ST_conf']` | https://pypi.org/project/SALib/ ; horizontal bars sorted by ST. |
| Radar / spider | matplotlib gallery `radar_factory` (custom `RadarAxes` projection, `frame='polygon'`) | https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html ; keep <=6 axes and <=4 series; fill `alpha=0.15`; always add a table with exact values in supplement (radar area is not a valid metric). |
| Multi-panel grid | `fig, axs = plt.subplot_mosaic([['a','b'],['c','d']], layout='constrained', figsize=(180/25.4, 120/25.4))`; panel labels via `ScaledTranslation` | https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html |
| Bland-Altman | `statsmodels.graphics.agreement.mean_diff_plot(m1, m2, sd_limit=1.96, ax=ax, scatter_kwds=..., mean_line_kwds=..., limit_lines_kwds=...)` (statsmodels 0.15.0) | https://www.statsmodels.org/stable/generated/statsmodels.graphics.agreement.mean_diff_plot.html ; add CI shading on bias and LoA lines yourself. |
| SHAP beeswarm | `shap` 0.52.0: `shap.plots.beeswarm(expl, max_display=12, ax=ax, show=False, color=cmc.vik, color_bar=True)` | https://shap.readthedocs.io/en/latest/generated/shap.plots.beeswarm.html ; `ax` and `plot_size` are mutually exclusive (pass `ax`, omit `plot_size`). Windows/Py3.14 uses the `cp312-abi3-win_amd64` wheel (stable ABI), confirmed by pip dry-run. |
| Swarm (small n) | `seaborn.swarmplot(..., size=2.5, ax=ax)` or `stripplot(jitter=0.2)` | swarm fails to place points when n is large; switch to `stripplot` or a violin. |
| Cohen-d dot plot | matplotlib `ax.errorbar(d, y, xerr=[d-lo, hi-d], fmt='o', ms=3, capsize=2)`; CI from `scipy.stats.bootstrap` or Hedges' formula; vertical reference lines at 0.2/0.5/0.8 | `pingouin.compute_effsize` also available but not required. |
| Training curves | matplotlib; plot mean +/- SD across seeds as line + band; log-scale y for loss when appropriate | Thin lines (0.8 pt), no markers, label the final value directly at the line end instead of a legend. |

### Bootstrap ROC band (drop-in)
```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
def roc_band(y, p, n_boot=2000, seed=0, grid=np.linspace(0, 1, 101)):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    tprs, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max(): continue
        fpr, tpr, _ = roc_curve(y[idx], p[idx])
        tprs.append(np.interp(grid, fpr, tpr)); aucs.append(roc_auc_score(y[idx], p[idx]))
    tprs = np.array(tprs)
    return grid, np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0), np.percentile(aucs, [2.5, 97.5])
# ax.fill_between(grid, lo, hi, alpha=0.2, lw=0, color=c); ax.plot(fpr, tpr, color=c, label=f"AUC {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

---

## 5. Architecture / system diagram (vector + editable)

| Tool | Vector | Editable later | Auto-layout | Windows install | Verdict |
|---|---|---|---|---|---|
| **draw.io desktop** (CLI export) | PDF/SVG | yes (`.drawio` source; `--embed-diagram` keeps XML inside the PDF/SVG) | manual (has "Arrange > Layout") | `winget install --id JGraph.Draw` | **Recommended** |
| Graphviz (`dot`) + `graphviz` 0.21 py | PDF/SVG | yes (text DSL) | yes | `winget install -e --id Graphviz.Graphviz` ; `pip install graphviz` | Best for auto-laid-out DAGs; boxes look generic |
| schemdraw 0.23 (`schemdraw.flow`) | SVG/PDF via matplotlib | yes (Python) | no | `pip install schemdraw[matplotlib]` | Good if you want fonts identical to the plots; tedious for >15 boxes. https://schemdraw.readthedocs.io/en/stable/elements/flow.html |
| `diagrams` 0.25.1 | PNG-first (SVG/PDF possible) | yes (Python) | yes (Graphviz) | needs Graphviz | Cloud-icon oriented; wrong idiom for hardware + model pipelines |
| PlantUML | SVG | yes (text) | yes | needs Java | UML look; hard to make it look like a Nature schematic |
| TikZ | PDF | yes (LaTeX) | no | LaTeX toolchain | Perfect font match if the paper is LaTeX; slowest iteration |

### Recommendation: draw.io desktop, exported headlessly from the command line
Install:
```
winget install --id JGraph.Draw --exact --accept-package-agreements --accept-source-agreements
```
Export (PDF master, cropped to content; SVG with the editable diagram embedded):
```
"C:\Program Files\draw.io\draw.io.exe" -x -f pdf --crop -o fig_architecture.pdf fig_architecture.drawio
"C:\Program Files\draw.io\draw.io.exe" -x -f svg --embed-diagram -o fig_architecture.svg fig_architecture.drawio
"C:\Program Files\draw.io\draw.io.exe" -x -f png -s 6 -o fig_architecture_600dpi.png fig_architecture.drawio
```
Flags (`-x/--export`, `-f/--format pdf|png|jpg|svg|xml|html`, `-o/--output`, `--crop`, `-t/--transparent`, `-b/--border`, `-s/--scale`, `--width/--height`, `-p/--page-index` (1-based), `-a/--all-pages`, `--embed-svg-images`, `-e/--embed-diagram`, `-u/--uncompressed`): https://linuxcommandlibrary.com/man/drawio ; repo https://github.com/jgraph/drawio-desktop.
- On Windows the desktop app exports from the shell directly; on Linux CI it needs `xvfb-run` (Electron). https://github.com/jgraph/drawio-desktop/issues/127
- In draw.io set page size to the target width (88 or 180 mm), font Helvetica/Arial 7 pt, line width 0.75 pt, and use Okabe-Ito fills at 20 % tint so it matches the plots. Use "Edit > Edit Style" to set `fontFamily=Helvetica` globally; PDF export embeds fonts.
- draw.io PDF is native vector; text stays editable in Illustrator/Inkscape (a Nature requirement).

Fallback if a pure-Python pipeline is mandatory: Graphviz `dot -Tpdf` with `node [shape=box, fontname="Helvetica", fontsize=7]`, `rankdir=LR`, and `splines=ortho`.

---

## 6. Anti-clutter rules

### Overlapping labels: `adjustText` 1.4.0 (released 2026-06-08)
- https://github.com/Phlya/adjustText ; docs https://adjusttext.readthedocs.io/
- `from adjustText import adjust_text`; collect `texts = [ax.text(x, y, s) for ...]` then `adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='0.5', lw=0.4), expand=(1.2, 1.4))`.
- Call it **after** the final `ax.set_xlim/ylim` and after layout is resolved (`fig.canvas.draw()` first), otherwise positions shift on save.
- Use for: labelled scatter points (per-fold or per-model markers), Cohen-d labels, ROC operating-point annotations. Do not use for tick labels (rotate or use horizontal bars instead).

### constrained_layout vs tight_layout
- Use `layout='constrained'` at figure creation (`plt.subplots(..., layout='constrained')` or rcParam `figure.constrained_layout.use = True`). It is a live layout engine: re-solved on every draw, handles colourbars, suptitles and legends placed outside axes. Guide: https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html
- `tight_layout()` is one-shot, ignores artists outside axes (legends via `bbox_to_anchor` get clipped), and **calling it turns constrained layout off**. Never mix the two.
- `layout='compressed'` for grids of fixed-aspect panels (confusion matrices, images) to remove the gaps constrained layout would leave.
- Tune spacing with `fig.get_layout_engine().set(w_pad=2/72, h_pad=2/72, wspace=0.03, hspace=0.03)`.
- Limitation: constrained layout requires every `plt.subplot()` call to use the same grid; use `subplot_mosaic`/`subplots`/`GridSpec` instead.

### Legends outside axes without overlap
- Figure-level, reserved space (matplotlib >=3.7, needs constrained layout):
  `fig.legend(handles, labels, loc='outside upper center', ncols=4, frameon=False)` -- the `outside` prefix makes the layout engine reserve room. Legend guide: https://matplotlib.org/stable/users/explain/axes/legend_guide.html
- Axes-level: `ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))` -- constrained layout will shrink the axes to fit it. If you do not want the axes shrunk: `leg.set_in_layout(False)` and save with `bbox_inches='tight'`.
- Prefer direct labelling (text at the end of each line) over a legend when there are <=4 series; prefer one shared legend per multi-panel figure rather than one per panel.
- Deduplicate handles across panels: `h, l = axs['a'].get_legend_handles_labels()` then one `fig.legend`.

### Other rules that keep figures clean at 88 mm
- Max ~5 series per axes; more goes into panels.
- `ax.spines[['top','right']].set_visible(False)`; ticks outward, 0.5 pt lines; no gridlines except faint horizontal in bar charts (`alpha=0.3, lw=0.4`).
- Tick labels: `ax.xaxis.set_major_locator(MaxNLocator(5))`; never rotate more than 45 deg -- switch to horizontal bars.
- Axes titles only when the panel letter + caption cannot carry the meaning; keep titles at `font.size`, not larger.
- Panel labels: physical offset, not axes fraction, so they align across panels of different sizes:
  ```python
  from matplotlib.transforms import ScaledTranslation
  for lab, ax in axs.items():
      ax.text(0, 1, lab, transform=ax.transAxes + ScaledTranslation(-20/72, 4/72, fig.dpi_scale_trans),
              fontsize=8, fontweight='bold', va='bottom', ha='left')
  ```
- Save: `fig.savefig('Fig1.pdf')` (vector master), plus `fig.savefig('Fig1.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})` and `dpi=600` PNG. Never `bbox_inches='tight'` on the final export (changes the physical width you designed); constrained layout makes it unnecessary.
- Fonts: set `pdf.fonttype = 42` and `ps.fonttype = 42` (TrueType, editable text -- required by Nature "do not outline text"); `svg.fonttype = 'none'`.

---

## 7. Recommended stack

Pip line (all versions live on PyPI as of 2026-09-07; resolved with `pip install --dry-run` on Python 3.14.3 / win_amd64):
```
pip install "matplotlib==3.11.1" "seaborn==0.13.2" "scikit-learn==1.9.0" "statsmodels==0.15.0" "scipy" "numpy" "SciencePlots==2.2.2" "cmcrameri==1.10" "adjustText==1.4.0" "shap==0.52.0" "SALib==1.5.2" && pip install --no-deps "forestplot==0.4.1"
```
Wheels that will be used on this machine: `matplotlib-3.11.1-cp314-cp314-win_amd64.whl`, `scikit_learn-1.9.0-cp314-cp314-win_amd64.whl`, `shap-0.52.0-cp312-abi3-win_amd64.whl`; SciencePlots, cmcrameri, adjustText, forestplot, SALib, seaborn are pure-Python wheels.
Optional: `pip install "tueplots==0.2.4"` (figsize helpers), `pip install "schemdraw[matplotlib]==0.23"` (Python-drawn flow diagram), `winget install --id JGraph.Draw` (architecture figure).

Not needed: LaTeX, plotly/kaleido, Chrome.

### rcParams block (paste after `import scienceplots; plt.style.use(['science', 'nature', 'no-latex'])`, or use standalone)
```python
import matplotlib as mpl
from cycler import cycler
MM = 1 / 25.4
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']
mpl.rcParams.update({
    'figure.figsize': (88 * MM, 66 * MM),        # single column; use (180*MM, h) for double
    'figure.dpi': 150, 'savefig.dpi': 600,
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.w_pad': 2 / 72, 'figure.constrained_layout.h_pad': 2 / 72,
    'savefig.bbox': 'standard', 'savefig.pad_inches': 0.0, 'savefig.transparent': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',  # editable TrueType text
    'text.usetex': False, 'mathtext.fontset': 'dejavusans',
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6, 'legend.title_fontsize': 6,
    'axes.prop_cycle': cycler(color=OKABE_ITO),
    'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'axes.axisbelow': True, 'axes.labelpad': 2, 'axes.titlepad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.pad': 2, 'ytick.major.pad': 2, 'xtick.minor.visible': False, 'ytick.minor.visible': False,
    'lines.linewidth': 1.0, 'lines.markersize': 3, 'lines.markeredgewidth': 0.5,
    'errorbar.capsize': 2, 'patch.linewidth': 0.5, 'hatch.linewidth': 0.4,
    'legend.frameon': False, 'legend.handlelength': 1.5, 'legend.handletextpad': 0.5,
    'legend.columnspacing': 1.0, 'legend.borderaxespad': 0.3,
    'image.cmap': 'cividis', 'image.interpolation': 'nearest',
})
# For an IEEE submission: mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
#   'ytick.labelsize': 7, 'legend.fontsize': 7, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman']})
```

### Export helper
```python
def save(fig, stem, formats=('pdf', 'png', 'tif')):
    for ext in formats:
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tif' else {}
        fig.savefig(f'{stem}.{ext}', dpi=600, **kw)
```
