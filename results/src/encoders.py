"""
Per-modality encoders (step A6). Each branch maps one modality's native
representation to a small set of d=64 tokens; the fusion trunk (A7) consumes
the concatenated token sequence.

Design constraints (see PROGRESS.md §3 and the build manifest §3.1):
  - capacity lives here, not in the fusion trunk; ~150k trainable params total
  - every encoder is standalone-trainable so per-modality AUCs can be
    reported before fusion
  - PPG uses frozen PaPaGei-S (BSD-3-Clause, Nokia Bell Labs) + a trainable
    adapter; input contract is (B, 1, 1250) at 125 Hz, 10-s segments

Smoke test:  python results/src/encoders.py
"""
import pathlib
import sys

import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
D_TOKEN = 64


# ---------------------------------------------------------------- TCN core --
class TCNBlock(nn.Module):
    """Dilated causal residual block: conv-GELU-dropout x2 + skip."""

    def __init__(self, c_in, c_out, k=5, dilation=1, p_drop=0.1):
        super().__init__()
        pad = (k - 1) * dilation          # causal left-pad
        self.pad = pad
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(c_in, c_out, k, dilation=dilation))
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(c_out, c_out, k, dilation=dilation))
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        z = nn.functional.pad(x, (self.pad, 0))
        z = self.drop(self.act(self.conv1(z)))
        z = nn.functional.pad(z, (self.pad, 0))
        z = self.drop(self.act(self.conv2(z)))
        return self.act(z + self.skip(x))


class TCN(nn.Module):
    def __init__(self, c_in, channels=(32, 32, 64, 64), k=5, p_drop=0.1):
        super().__init__()
        blocks, c = [], c_in
        for i, ch in enumerate(channels):
            blocks.append(TCNBlock(c, ch, k=k, dilation=2 ** i, p_drop=p_drop))
            c = ch
        self.net = nn.Sequential(*blocks)
        self.c_out = c

    def forward(self, x):                  # (B, C, T) -> (B, c_out, T)
        return self.net(x)


class AttnPool(nn.Module):
    """Pool a (B, C, T) sequence into n_tokens learned-query tokens of dim d."""

    def __init__(self, c_in, n_tokens, d=D_TOKEN):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.key = nn.Linear(c_in, d)
        self.val = nn.Linear(c_in, d)

    def forward(self, x):                  # (B, C, T) -> (B, n_tokens, d)
        h = x.transpose(1, 2)              # (B, T, C)
        k, v = self.key(h), self.val(h)    # (B, T, d)
        att = torch.einsum("nd,btd->bnt", self.query, k) / (k.shape[-1] ** 0.5)
        att = att.softmax(dim=-1)
        return torch.einsum("bnt,btd->bnd", att, v)


class Time2Vec(nn.Module):
    """Kazemi et al. 2019: one linear + (d-1) periodic time features."""

    def __init__(self, d=8):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, t):                  # (B, T) -> (B, d, T)
        z = t.unsqueeze(-1) * self.w + self.b          # (B, T, d)
        z = torch.cat([z[..., :1], torch.sin(z[..., 1:])], dim=-1)
        return z.transpose(1, 2)


def _mlp(d_in, d_hidden, d_out, p_drop=0.1):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Dropout(p_drop), nn.Linear(d_hidden, d_out))


# ------------------------------------------------------------- the branches --
class PupilEncoder(nn.Module):
    """Stimulus-locked pupil trace, 30 Hz x 8 s = 240 samples -> 4 tokens."""

    N_TOKENS = 4

    def __init__(self):
        super().__init__()
        self.tcn = TCN(c_in=1)
        self.pool = AttnPool(self.tcn.c_out, self.N_TOKENS)

    def forward(self, x):                  # (B, 1, 240)
        return self.pool(self.tcn(x))


class CGMEncoder(nn.Module):
    """14 trailing days at 5-min sampling (B, 1, 4032) + 8 glycaemic-
    variability metrics -> 2 tokens (1 waveform, 1 GV)."""

    N_TOKENS = 2

    def __init__(self, n_gv=8):
        super().__init__()
        self.down = nn.Conv1d(1, 16, 15, stride=6)     # 4032 -> ~670
        self.tcn = TCN(c_in=16, channels=(16, 32, 32))
        self.pool = AttnPool(self.tcn.c_out, 1)
        self.gv = _mlp(n_gv, 32, D_TOKEN)

    def forward(self, x, gv):              # (B,1,4032), (B,8)
        tok_wave = self.pool(self.tcn(self.down(x)))
        return torch.cat([tok_wave, self.gv(gv).unsqueeze(1)], dim=1)


class PPGEncoder(nn.Module):
    """Frozen PaPaGei-S (512-d) + trainable adapter -> 2 tokens.

    Contract: input (B, 1, 1250), z-scored per segment, 125 Hz.
    Weights are lazy-loaded so the scaffold works without the 22 MB file.
    """

    N_TOKENS = 2
    EMB = 512

    def __init__(self, weights=None):
        super().__init__()
        self.backbone = None
        self.weights_path = weights
        self.adapter = nn.Sequential(
            nn.Linear(self.EMB, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, self.N_TOKENS * D_TOKEN))

    def load_backbone(self):
        vend = HERE.parent / "vendor" / "papagei"
        sys.path.insert(0, str(vend))
        from models.resnet import ResNet1DMoE
        m = ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3,
                        stride=2, groups=1, n_block=18, n_classes=self.EMB,
                        n_experts=3)
        sd = torch.load(self.weights_path, map_location="cpu",
                        weights_only=True)
        sd = { (k[7:] if k.startswith("module.") else k): v
               for k, v in sd.items() }
        m.load_state_dict(sd)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.backbone = m

    def forward(self, x=None, emb=None):   # (B,1,1250) or precomputed (B,512)
        if emb is None:
            if self.backbone is None:
                self.load_backbone()
            with torch.no_grad():
                emb = self.backbone(x)[0]          # first element = embeddings
        B = emb.shape[0]
        return self.adapter(emb).reshape(B, self.N_TOKENS, D_TOKEN)


class ThermalEncoder(nn.Module):
    """2 canthal channels at ~1/60 Hz over a session, plus Time2Vec of clock
    time -> 2 tokens."""

    N_TOKENS = 2

    def __init__(self, d_time=8):
        super().__init__()
        self.t2v = Time2Vec(d_time)
        self.conv = nn.Sequential(
            nn.Conv1d(2 + d_time, 32, 15, stride=5), nn.GELU(),
            nn.Conv1d(32, 32, 7, stride=2), nn.GELU())
        self.pool = AttnPool(32, self.N_TOKENS)

    def forward(self, x, t):               # (B,2,T), (B,T) clock time
        z = torch.cat([x, self.t2v(t)], dim=1)
        return self.pool(self.conv(z))


class BISEncoder(nn.Module):
    """Bioimpedance: 32 log-spaced frequencies x (Re, Im) flattened to 64,
    plus 4 fitted Cole-Cole params -> 1 token."""

    N_TOKENS = 1

    def __init__(self, n_spec=64, n_cole=4):
        super().__init__()
        self.mlp = _mlp(n_spec + n_cole, 128, D_TOKEN)

    def forward(self, spec, cole):         # (B,64), (B,4)
        return self.mlp(torch.cat([spec, cole], dim=-1)).unsqueeze(1)


class HRVEncoder(nn.Module):
    """16 classical HRV features -> 1 token."""

    N_TOKENS = 1

    def __init__(self, n_feat=16):
        super().__init__()
        self.mlp = _mlp(n_feat, 32, D_TOKEN)

    def forward(self, x):                  # (B,16)
        return self.mlp(x).unsqueeze(1)


class StaticEncoder(nn.Module):
    """Clinical covariates (age, sex, duration, insulin, ...) -> 1 token."""

    N_TOKENS = 1

    def __init__(self, n_feat=6):
        super().__init__()
        self.mlp = _mlp(n_feat, 32, D_TOKEN)

    def forward(self, x):                  # (B,6)
        return self.mlp(x).unsqueeze(1)


# ------------------------------------------------------------------ smoke ---
def n_params(m, trainable_only=True):
    return sum(p.numel() for p in m.parameters()
               if p.requires_grad or not trainable_only)


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    rows = []

    enc = PupilEncoder()
    out = enc(torch.randn(B, 1, 240))
    rows.append(("pupil", out.shape, n_params(enc)))

    enc = CGMEncoder()
    out = enc(torch.randn(B, 1, 4032), torch.randn(B, 8))
    rows.append(("cgm", out.shape, n_params(enc)))

    enc = PPGEncoder()                      # adapter only; backbone frozen+lazy
    out = enc(emb=torch.randn(B, 512))
    rows.append(("ppg-adapter", out.shape, n_params(enc)))

    enc = ThermalEncoder()
    out = enc(torch.randn(B, 2, 600), torch.rand(B, 600))
    rows.append(("thermal", out.shape, n_params(enc)))

    enc = BISEncoder()
    out = enc(torch.randn(B, 64), torch.randn(B, 4))
    rows.append(("bis", out.shape, n_params(enc)))

    enc = HRVEncoder()
    out = enc(torch.randn(B, 16))
    rows.append(("hrv", out.shape, n_params(enc)))

    enc = StaticEncoder()
    out = enc(torch.randn(B, 6))
    rows.append(("static", out.shape, n_params(enc)))

    total_tokens, total_params = 0, 0
    print(f"{'branch':12s} {'output':22s} {'trainable params':>16s}")
    print("-" * 54)
    for name, shape, np_ in rows:
        print(f"{name:12s} {str(tuple(shape)):22s} {np_:16,d}")
        total_tokens += shape[1]
        total_params += np_
    print("-" * 54)
    print(f"{'TOTAL':12s} {total_tokens} tokens x {D_TOKEN}d {total_params:>13,d}")
    assert all(s[2] == D_TOKEN for _, s, _ in rows), "token dim mismatch"
    print("\nall branches forward cleanly; token dim uniform at d=64")
