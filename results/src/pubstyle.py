"""
Publication figure style for the NRDI paper figure set (F1-F13 + appendix).
Basis: results/figures/FIGURE_TOOLING.md (SciencePlots 'science'+'nature'+'no-latex', Okabe-Ito,
88 / 180 mm widths, 7 pt body, 8 pt bold lowercase panel labels, TrueType-embedded PDF master).

    import pubstyle as ps
    ps.apply()
    fig, axs = ps.figure(cols=2, rows=1, width="double", height_mm=70)
    ps.label(axs[0], "a"); ...; ps.save(fig, "figF09_fundus_dr")
"""
import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from cycler import cycler
import scienceplots  # noqa: F401  (registers the styles)

MM = 1 / 25.4
SINGLE_MM, MID_MM, DOUBLE_MM = 88, 120, 180
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']
TOL_MUTED = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499', '#DDDDDD']
# semantic colours used across the whole set (never re-assign)
C = {"fundus": '#0072B2', "external": '#D55E00', "external2": '#009E73', "dme": '#CC79A7', "ehr": '#332288',
     "hrv": '#117733', "ppg": '#882255', "thermal": '#D55E00', "optical": '#E69F00', "electrical": '#0072B2',
     "pupil": '#009E73', "ref": '#7f7f7f', "ink": '#1e1b17', "good": '#117733', "warn": '#CC6677'}
INK, MUTED, RULE = '#1e1b17', '#6e675e', '#c9c2b6'
SEED = 20260828
FIGDIR = pathlib.Path(__file__).resolve().parents[1] / "figures"


def apply(journal="nature"):
    plt.style.use(['science', 'nature', 'no-latex'])
    mpl.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 600,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.w_pad': 2 / 72, 'figure.constrained_layout.h_pad': 2 / 72,
        'savefig.bbox': 'standard', 'savefig.pad_inches': 0.0, 'savefig.transparent': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
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
        'xtick.top': False, 'ytick.right': False, 'xtick.minor.top': False, 'ytick.minor.right': False,
        'lines.linewidth': 1.0, 'lines.markersize': 3, 'lines.markeredgewidth': 0.5,
        'errorbar.capsize': 2, 'patch.linewidth': 0.5, 'hatch.linewidth': 0.4,
        'legend.frameon': False, 'legend.handlelength': 1.5, 'legend.handletextpad': 0.5,
        'legend.columnspacing': 1.0, 'legend.borderaxespad': 0.3,
        'image.cmap': 'cividis', 'image.interpolation': 'nearest',
    })
    if journal == "ieee":
        mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                             'legend.fontsize': 7, 'font.family': 'serif',
                             'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif']})


def figure(cols=1, rows=1, width="single", height_mm=66, layout="constrained", **kw):
    w = {"single": SINGLE_MM, "mid": MID_MM, "double": DOUBLE_MM}.get(width, width)
    fig, axs = plt.subplots(rows, cols, figsize=(w * MM, height_mm * MM), layout=layout, **kw)
    return fig, axs


def label(ax, letter, dx_mm=-4.0, dy_mm=1.5):
    """8 pt bold lowercase panel label at a fixed physical offset from the axes' top-left corner."""
    off = ScaledTranslation(dx_mm * MM, dy_mm * MM, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes + off, fontsize=8, fontweight='bold', va='bottom', ha='left')


def save(fig, stem, formats=('pdf', 'png'), outdir=None):
    outdir = pathlib.Path(outdir or FIGDIR)
    outdir.mkdir(exist_ok=True)
    for ext in formats:
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tif' else {}
        fig.savefig(outdir / f'{stem}.{ext}', dpi=600, **kw)
    print("wrote", outdir / f"{stem}.pdf")


# ------------------------------------------------------------------ helpers ---
def roc_band(y, score, n_boot=1000, seed=SEED, groups=None):
    """Bootstrap ROC: grid fpr, mean tpr, 2.5/97.5 % tpr bands and the AUC CI. Group-level if `groups` given."""
    from sklearn.metrics import roc_curve, roc_auc_score
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(score)
    grid = np.linspace(0, 1, 201); tprs, aucs = [], []
    if groups is not None:
        g = np.asarray(groups); ug = np.unique(g); idx_by = {k: np.where(g == k)[0] for k in ug}
    for _ in range(n_boot):
        idx = (np.concatenate([idx_by[k] for k in rng.choice(ug, len(ug), replace=True)]) if groups is not None
               else rng.integers(0, len(y), len(y)))
        if len(np.unique(y[idx])) < 2:
            continue
        f, t, _ = roc_curve(y[idx], s[idx]); tprs.append(np.interp(grid, f, t)); aucs.append(roc_auc_score(y[idx], s[idx]))
    tprs = np.array(tprs)
    return (grid, tprs.mean(0), np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0),
            (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))))


def plot_roc(ax, y, score, color, name, groups=None, n_boot=1000, band=True):
    from sklearn.metrics import roc_curve, roc_auc_score
    f, t, _ = roc_curve(y, score); a = roc_auc_score(y, score)
    g, m, lo, hi, ci = roc_band(y, score, n_boot=n_boot, groups=groups)
    if band:
        ax.fill_between(g, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(f, t, color=color, label=f"{name}: AUC {a:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    return a, ci


def confusion(ax, cm, labels, cmap="Blues", normalise="row", fmt="{:d}"):
    cm = np.asarray(cm, float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalise == "row" else cm / cm.sum()
    im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fmt.format(int(cm[i, j])), ha="center", va="center", fontsize=5.5,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.spines[['top', 'right']].set_visible(True)
    ax.tick_params(length=0)
    return im


def refline(ax, kind="diag"):
    if kind == "diag":
        ax.plot([0, 1], [0, 1], color=RULE, lw=0.6, ls=(0, (3, 2)), zorder=0)
    elif kind == "zero":
        ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    elif kind == "one":
        ax.axvline(1, color=RULE, lw=0.6, zorder=0)


def finish_legend(ax, loc="lower right", **kw):
    leg = ax.legend(loc=loc, **kw)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg
