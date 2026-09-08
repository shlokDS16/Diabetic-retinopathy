"""
Shared figure style for ALL results figures.

Import this before pyplot in every results figure script:

    import figstyle as fs
    fig = plt.figure(figsize=fs.DOUBLE)

Design intent: later models are added to existing figures by registering a
colour in MODEL_COLORS once — never by restyling a figure. The accent orange
is reserved for the headline fusion model and must not be used earlier.
Rationale and sources: results/RESEARCH_NOTES.md.
"""
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType, never Type 3
    "legend.frameon": False,
})

# IEEE column widths (182 mm / 88.9 mm), heights chosen per figure
DOUBLE = (7.16, 2.6)
SINGLE = (3.5, 2.8)

INK    = "#1e1b17"
MUTED  = "#6e675e"
RULE   = "#bfb7aa"
GOOD   = "#2e6a4c"
WARN   = "#8a6009"

# ---- model colour registry -------------------------------------------------
# Add one line per new model; figures pick colours up from here automatically.
MODEL_COLORS = {
    "clinical_lr":   "#4a5568",   # baseline: neutral slate
    "single_best":   "#2b5578",   # best single-modality branch (when it exists)
    "fusion":        "#b04a20",   # RESERVED: headline multimodal fusion model
    "fusion_q":      "#7a3018",   # fusion + quantum layer
}
MODEL_LABELS = {
    "clinical_lr": "Clinical risk factors (LR)",
    "single_best": "Best single modality",
    "fusion":      "Multimodal fusion",
    "fusion_q":    "Multimodal fusion + Q",
}

SEED = 20260828


def panel_title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.5)


def save(fig, path_stem):
    """600 dpi PNG for review + vector PDF for submission."""
    fig.savefig(f"{path_stem}.png")
    fig.savefig(f"{path_stem}.pdf")
