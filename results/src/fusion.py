"""
Fusion trunk (step A7).

Consumes the 13 modality tokens from encoders.py and produces a pooled
fusion vector for the concept bottleneck (A8).

Design (build manifest section 3.2; rationale in PROGRESS.md):
  - MBT-style bottleneck: 4 learnable fusion tokens. Modality tokens may
    attend to the bottleneck tokens and to their own modality's tokens ONLY;
    cross-modal exchange is forced through the bottleneck. This is the
    capacity cap that keeps the trunk honest at n in the tens-to-hundreds.
  - Shared/specific decomposition (ShaSpec-flavoured): each modality's
    pooled representation splits into a shared half (alignment loss pulls
    modalities together) and a specific half (a modality classifier pushes
    them apart). Both are auxiliary losses returned to the trainer.
  - Reliability gate: a scalar r_m per modality from its own tokens,
    to be supervised by measured signal quality; token contributions are
    scaled by r_m before attention.
  - Modality dropout: at train time each modality is dropped with
    p=MODALITY_DROP (at least MIN_KEEP survive) and replaced by a learned
    absent-embedding, so inference with missing sensors degrades gracefully.

Smoke test:  python results/src/fusion.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64
N_BOTTLENECK = 4
MODALITY_DROP = 0.3
MIN_KEEP = 2

# token layout must match encoders.py output order
MODALITY_SLOTS = {
    "pupil": 4, "cgm": 2, "ppg": 2, "thermal": 2,
    "bis": 1, "hrv": 1, "static": 1,
}
N_TOKENS = sum(MODALITY_SLOTS.values())          # 13
MODALITIES = list(MODALITY_SLOTS)


def _slot_ranges():
    out, i = {}, 0
    for m, k in MODALITY_SLOTS.items():
        out[m] = (i, i + k)
        i += k
    return out


SLOTS = _slot_ranges()


def build_attn_mask(device=None):
    """(L, L) boolean mask, True = BLOCKED. Layout: [bottleneck | modality]."""
    L = N_BOTTLENECK + N_TOKENS
    m = torch.ones(L, L, dtype=torch.bool, device=device)
    m[:N_BOTTLENECK, :] = False                       # bottleneck sees all
    for a, b in SLOTS.values():
        rows = slice(N_BOTTLENECK + a, N_BOTTLENECK + b)
        m[rows, :N_BOTTLENECK] = False                # modality -> bottleneck
        m[rows, rows] = False                         # modality -> itself
    return m


class TrunkLayer(nn.Module):
    """Pre-norm transformer layer with a fixed attention mask."""

    def __init__(self, d=D, heads=4, ffn=128, p_drop=0.2):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=p_drop,
                                         batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(),
                                 nn.Dropout(p_drop), nn.Linear(ffn, d))

    def forward(self, x, mask):
        h = self.n1(x)
        a, _ = self.att(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.n2(x))


class FusionTrunk(nn.Module):
    def __init__(self, d=D, n_layers=2):
        super().__init__()
        self.bottleneck = nn.Parameter(torch.randn(N_BOTTLENECK, d) * 0.02)
        self.absent = nn.ParameterDict({
            m: nn.Parameter(torch.randn(k, d) * 0.02)
            for m, k in MODALITY_SLOTS.items()})
        self.reliability = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, 1))
            for m in MODALITIES})
        self.layers = nn.ModuleList(TrunkLayer(d) for _ in range(n_layers))
        # shared/specific split heads (d -> d/2 each)
        self.to_shared = nn.Linear(d, d // 2)
        self.to_specific = nn.Linear(d, d // 2)
        self.domain_head = nn.Linear(d // 2, len(MODALITIES))
        self.register_buffer("mask", build_attn_mask(), persistent=False)

    # -- modality dropout -----------------------------------------------------
    def _drop_modalities(self, tokens, present):
        """Randomly hide modalities at train time; always respect `present`
        (a (B, n_mod) bool tensor of what the device actually delivered)."""
        B = tokens.shape[0]
        if self.training:
            rnd = torch.rand(B, len(MODALITIES), device=tokens.device) > MODALITY_DROP
            # guarantee MIN_KEEP survivors per row
            for i in range(B):
                if rnd[i].sum() < MIN_KEEP:
                    keep = torch.randperm(len(MODALITIES))[:MIN_KEEP]
                    rnd[i, keep] = True
            present = present & rnd
        out = tokens.clone()
        for j, m in enumerate(MODALITIES):
            a, b = SLOTS[m]
            gone = ~present[:, j]
            if gone.any():
                out[gone, a:b] = self.absent[m].unsqueeze(0).expand(
                    int(gone.sum()), -1, -1).to(out.dtype)
        return out, present

    def forward(self, tokens, present=None):
        """
        tokens : (B, 13, D) from the encoders, in MODALITY_SLOTS order
        present: (B, 7) bool, which modalities the device delivered
                 (None = all present)
        returns dict with fused vector, per-modality reliability,
        shared/specific pieces and the aux losses.
        """
        B = tokens.shape[0]
        if present is None:
            present = torch.ones(B, len(MODALITIES), dtype=torch.bool,
                                 device=tokens.device)
        tokens, present = self._drop_modalities(tokens, present)

        # reliability gates from each modality's own (post-dropout) tokens
        rel = []
        for m in MODALITIES:
            a, b = SLOTS[m]
            rel.append(torch.sigmoid(
                self.reliability[m](tokens[:, a:b].mean(1))))
        rel = torch.cat(rel, dim=1)                       # (B, 7) in (0,1)

        scaled = tokens.clone()
        for j, m in enumerate(MODALITIES):
            a, b = SLOTS[m]
            scaled[:, a:b] = tokens[:, a:b] * rel[:, j].view(B, 1, 1)

        x = torch.cat([self.bottleneck.unsqueeze(0).expand(B, -1, -1),
                       scaled], dim=1)
        for layer in self.layers:
            x = layer(x, self.mask)

        fused = x[:, :N_BOTTLENECK].mean(1)               # (B, D)

        # shared/specific decomposition on per-modality pooled vectors
        pooled = torch.stack([x[:, N_BOTTLENECK + a:N_BOTTLENECK + b].mean(1)
                              for a, b in SLOTS.values()], dim=1)  # (B,7,D)
        shared = self.to_shared(pooled)                   # (B,7,D/2)
        specific = self.to_specific(pooled)

        # alignment: shared halves of PRESENT modalities should agree
        w = present.float().unsqueeze(-1)
        centroid = (shared * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)
        loss_align = (((shared - centroid) ** 2).mean(-1) * present.float()).sum() \
                     / present.float().sum().clamp(min=1)

        # domain: specific halves should identify their modality
        logits = self.domain_head(specific)               # (B,7,7)
        tgt = torch.arange(len(MODALITIES), device=tokens.device)
        loss_domain = F.cross_entropy(
            logits.reshape(-1, len(MODALITIES)),
            tgt.repeat(B), reduction="none").reshape(B, -1)
        loss_domain = (loss_domain * present.float()).sum() \
                      / present.float().sum().clamp(min=1)

        return {"fused": fused, "reliability": rel, "present": present,
                "shared": shared, "specific": specific,
                "loss_align": loss_align, "loss_domain": loss_domain}


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8
    trunk = FusionTrunk()
    n = sum(p.numel() for p in trunk.parameters())
    print(f"trunk trainable params: {n:,}")

    # --- mask correctness ----------------------------------------------------
    m = build_attn_mask()
    a0, b0 = SLOTS["pupil"]; a1, b1 = SLOTS["hrv"]
    assert m[N_BOTTLENECK + a0, N_BOTTLENECK + a1], "pupil must NOT see hrv"
    assert not m[N_BOTTLENECK + a0, 0], "modality must see bottleneck"
    assert not m[0, N_BOTTLENECK + a1], "bottleneck must see modalities"
    print("attention mask: cross-modal blocked, bottleneck routes  OK")

    # --- forward, train mode (dropout active) --------------------------------
    trunk.train()
    out = trunk(torch.randn(B, N_TOKENS, D))
    print(f"fused {tuple(out['fused'].shape)}  reliability {tuple(out['reliability'].shape)}")
    print(f"aux losses: align {out['loss_align']:.4f}  domain {out['loss_domain']:.4f}")
    kept = out["present"].float().mean(0)
    print("modality keep-rate this batch:",
          {m: f"{kept[j]:.2f}" for j, m in enumerate(MODALITIES)})

    # --- eval mode with two modalities missing -------------------------------
    trunk.eval()
    present = torch.ones(B, len(MODALITIES), dtype=torch.bool)
    present[:, MODALITIES.index("bis")] = False
    present[:, MODALITIES.index("pupil")] = False
    out2 = trunk(torch.randn(B, N_TOKENS, D), present=present)
    print(f"eval w/o bis+pupil: fused {tuple(out2['fused'].shape)}, finite "
          f"{bool(torch.isfinite(out2['fused']).all())}")

    # --- gradients flow ------------------------------------------------------
    trunk.train()
    out3 = trunk(torch.randn(B, N_TOKENS, D, requires_grad=True))
    (out3["fused"].sum() + out3["loss_align"] + out3["loss_domain"]).backward()
    g = sum(p.grad.abs().sum().item() for p in trunk.parameters()
            if p.grad is not None)
    print(f"grad mass {g:.2f}  -> {'FLOWS' if g > 0 else 'DEAD'}")
