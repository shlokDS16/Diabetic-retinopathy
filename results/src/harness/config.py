"""
Experiment configuration. THE single source of every tunable in the results
pipeline — nothing downstream may hardcode a value that lives here.

Design rules
  - plain dataclasses, JSON round-trippable (save_config/load_config)
  - ablation arms are DATA: named override dicts in ABLATION_ARMS,
    applied onto the base config by `apply_overrides`
  - paths are derived from PROJECT_ROOT at runtime, never absolute literals
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field, asdict
import json
import pathlib
from typing import Any

# repo root = .../<project>/results ; this file sits in results/src/harness
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
@dataclass
class SplitConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    scheme: str = "stratified_group"      # patient-level, class-stratified
    group_key: str = "patient_id"
    stratify_key: str = "dr_grade"


@dataclass
class TrainConfig:
    seeds: tuple = (0, 1, 2, 3, 4)        # 5 seeds per arm (PROGRESS §A10)
    max_epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 12
    early_stop_metric: str = "val_auc"    # monitored on INNER validation only
    # loss weights (build manifest §3.5)
    w_dn: float = 0.5
    w_concept: float = 0.5
    w_align: float = 0.1
    w_domain: float = 0.1
    w_rank: float = 0.25
    device: str = "auto"                  # auto -> cuda if available


@dataclass
class ModelConfig:
    mixer: str = "quantum"                # quantum | classical | none
    d_token: int = 64
    n_bottleneck: int = 4
    trunk_layers: int = 2
    modality_dropout: float = 0.3
    min_keep: int = 2
    n_qubits: int = 10
    q_depth: int = 4
    gate_init: float = 0.1
    circuit_variant: str = "ring"         # ring | no_entangle
    freeze_quantum: bool = False          # q_frozen ablation: random fixed VQC
    dr_grades: int = 5
    dn_grades: int = 3
    # which modality branches participate (ablation hook — leave-one-out arms)
    modalities: tuple = ("pupil", "cgm", "ppg", "thermal", "bis", "hrv", "static")


@dataclass
class EvalConfig:
    bootstrap_iters: int = 2000
    bootstrap_level: str = "patient"
    ci: float = 0.95
    permutation_iters: int = 1000
    calibration_loess_frac: float = 0.6
    dca_thresholds: tuple = (0.05, 0.80)


@dataclass
class ExperimentConfig:
    name: str = "base"
    master_seed: int = 20260828           # governs split generation only
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    out_dir: str = "runs"                 # relative to results/

    def resolve_out(self) -> pathlib.Path:
        return PROJECT_ROOT / self.out_dir / self.name


# ---------------------------------------------------------------------------
def save_config(cfg: ExperimentConfig, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2, default=list),
                    encoding="utf-8")


def _build_config(d: dict) -> ExperimentConfig:
    def build(cls, dd):
        kw = {}
        for f in dataclasses.fields(cls):
            v = dd.get(f.name, dataclasses.MISSING)
            if v is dataclasses.MISSING:
                continue
            if dataclasses.is_dataclass(f.type) or f.name in (
                    "split", "train", "model", "eval"):
                sub = {"split": SplitConfig, "train": TrainConfig,
                       "model": ModelConfig, "eval": EvalConfig}[f.name]
                kw[f.name] = build(sub, v)
            elif isinstance(v, list):
                kw[f.name] = tuple(v)
            else:
                kw[f.name] = v
        return cls(**kw)

    return build(ExperimentConfig, d)


def load_config(path: pathlib.Path) -> ExperimentConfig:
    return _build_config(json.loads(path.read_text(encoding="utf-8")))


def apply_overrides(cfg: ExperimentConfig, overrides: dict) -> ExperimentConfig:
    """Dotted-path overrides, e.g. {"model.mixer": "classical"}."""
    cfg = load_config_from_dict(asdict(cfg))  # deep copy via round-trip
    for key, val in overrides.items():
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise KeyError(f"unknown config key: {key}")
        cur = getattr(obj, parts[-1])
        if isinstance(cur, tuple) and isinstance(val, list):
            val = tuple(val)
        object.__setattr__(obj, parts[-1], val)
    return cfg


def load_config_from_dict(d: dict) -> ExperimentConfig:
    # deep copy through JSON so tuples/lists normalise identically to disk I/O
    return _build_config(json.loads(json.dumps(d, default=list)))


# ---------------------------------------------------------------------------
# THE ABLATION REGISTRY — arms are data. Adding an arm = adding an entry.
# Names match PROGRESS.md §A10 / build manifest §4.2.
ABLATION_ARMS: dict[str, dict[str, Any]] = {
    # the four decisive arms, run first
    "fusion_q":        {},                                    # headline model
    "fusion_c80":      {"model.mixer": "classical"},          # param-matched twin
    "no_entangle":     {"model.circuit_variant": "no_entangle"},  # CZs removed
    "q_frozen":        {"model.freeze_quantum": True},        # random frozen VQC
    # structure ablations
    "no_mixer":        {"model.mixer": "none"},
    "single_task_dr":  {"train.w_dn": 0.0},
    "no_concepts":     {"train.w_concept": 0.0},
    "no_shaspec":      {"train.w_align": 0.0, "train.w_domain": 0.0},
    "no_mod_dropout":  {"model.modality_dropout": 0.0},
    # qubit / depth sweeps
    **{f"qubits_{n}": {"model.n_qubits": n} for n in (4, 6, 8, 12)},
    **{f"depth_{d}": {"model.q_depth": d} for d in (1, 2, 6)},
    # modality leave-one-out
    **{f"drop_{m}": {"model.modalities":
                     [x for x in ("pupil", "cgm", "ppg", "thermal",
                                  "bis", "hrv", "static") if x != m]}
       for m in ("pupil", "cgm", "ppg", "thermal", "bis", "hrv", "static")},
}


if __name__ == "__main__":
    import tempfile
    cfg = ExperimentConfig()
    p = pathlib.Path(tempfile.gettempdir()) / "cfg_roundtrip.json"
    save_config(cfg, p)
    cfg2 = load_config(p)
    assert asdict(cfg) == asdict(cfg2), "round-trip mismatch"
    arm = apply_overrides(cfg, ABLATION_ARMS["fusion_c80"])
    assert arm.model.mixer == "classical" and cfg.model.mixer == "quantum"
    arm2 = apply_overrides(cfg, ABLATION_ARMS["drop_bis"])
    assert "bis" not in arm2.model.modalities and len(arm2.model.modalities) == 6
    try:
        apply_overrides(cfg, {"model.nonexistent": 1})
        raise SystemExit("override validation FAILED")
    except KeyError:
        pass
    for name, ov in ABLATION_ARMS.items():
        apply_overrides(cfg, ov)          # every arm must apply cleanly
    print(f"config round-trip OK; {len(ABLATION_ARMS)} ablation arms registered:")
    print("  " + ", ".join(ABLATION_ARMS))
