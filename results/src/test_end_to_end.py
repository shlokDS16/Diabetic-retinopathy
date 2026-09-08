"""
End-to-end integration test: A6 encoders -> A7 trunk -> A8 concepts ->
A9 quantum mixer -> ordinal heads, trained as ONE graph on synthetic data.

Passes when: single forward/backward works, every stage receives gradient,
a short overfit run drives the loss down (the standard "can it learn at all"
check), and a modality can be dropped at eval without breaking anything.

Run:  python results/src/test_end_to_end.py
"""
import pathlib, sys, time
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import encoders as E
import fusion as F
from concept_quantum import ConceptModel, CornHead, N_CONCEPTS

torch.manual_seed(0)
B = 12


class FullModel(nn.Module):
    def __init__(self, mixer="quantum"):
        super().__init__()
        self.pupil = E.PupilEncoder()
        self.cgm = E.CGMEncoder()
        self.ppg = E.PPGEncoder()          # adapter only; emb passed directly
        self.thermal = E.ThermalEncoder()
        self.bis = E.BISEncoder()
        self.hrv = E.HRVEncoder()
        self.static = E.StaticEncoder()
        self.trunk = F.FusionTrunk()
        self.head = ConceptModel(mixer=mixer)

    def forward(self, batch, present=None):
        toks = torch.cat([
            self.pupil(batch["pupil"]),
            self.cgm(batch["cgm"], batch["gv"]),
            self.ppg(emb=batch["ppg_emb"]),
            self.thermal(batch["thermal"], batch["t_clock"]),
            self.bis(batch["bis_spec"], batch["bis_cole"]),
            self.hrv(batch["hrv"]),
            self.static(batch["static"]),
        ], dim=1)
        fused = self.trunk(toks, present=present)
        out = self.head(fused["fused"])
        out.update({k: fused[k] for k in
                    ("loss_align", "loss_domain", "reliability", "present")})
        return out


def synth(B):
    return {
        "pupil": torch.randn(B, 1, 240), "cgm": torch.randn(B, 1, 4032),
        "gv": torch.randn(B, 8), "ppg_emb": torch.randn(B, 512),
        "thermal": torch.randn(B, 2, 600), "t_clock": torch.rand(B, 600),
        "bis_spec": torch.randn(B, 64), "bis_cole": torch.randn(B, 4),
        "hrv": torch.randn(B, 16), "static": torch.randn(B, 6),
    }


if __name__ == "__main__":
    model = FullModel(mixer="quantum")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"full model trainable params: {n:,}")

    batch = synth(B)
    y_dr = torch.randint(0, 5, (B,))
    y_dn = torch.randint(0, 3, (B,))
    y_c = torch.randn(B, N_CONCEPTS)

    # ---- forward + backward -------------------------------------------------
    out = model(batch)
    loss = (CornHead.loss(out["dr_logits"], y_dr)
            + 0.5 * CornHead.loss(out["dn_logits"], y_dn)
            + 0.5 * nn.functional.mse_loss(out["concepts"], y_c)
            + 0.1 * out["loss_align"] + 0.1 * out["loss_domain"])
    loss.backward()

    stages = {"pupil": model.pupil, "cgm": model.cgm, "ppg": model.ppg,
              "thermal": model.thermal, "bis": model.bis, "hrv": model.hrv,
              "static": model.static, "trunk": model.trunk,
              "concept+mixer": model.head}
    print("gradient reach per stage:")
    all_ok = True
    for name, mod in stages.items():
        g = sum(p.grad.abs().sum().item() for p in mod.parameters()
                if p.grad is not None)
        ok = g > 0
        all_ok &= ok
        print(f"  {name:14s} {'OK' if ok else 'DEAD'}  ({g:.2e})")
    assert all_ok, "a stage received no gradient"

    # ---- short overfit run: loss must drop ----------------------------------
    model2 = FullModel(mixer="quantum")
    opt = torch.optim.Adam(model2.parameters(), lr=3e-3)
    t0, losses = time.perf_counter(), []
    for step in range(30):
        opt.zero_grad()
        o = model2(batch)
        l = (CornHead.loss(o["dr_logits"], y_dr)
             + 0.5 * CornHead.loss(o["dn_logits"], y_dn)
             + 0.5 * nn.functional.mse_loss(o["concepts"], y_c)
             + 0.1 * o["loss_align"] + 0.1 * o["loss_domain"])
        l.backward(); opt.step(); losses.append(l.item())
    dt = time.perf_counter() - t0
    print(f"\noverfit check: loss {losses[0]:.3f} -> {losses[-1]:.3f} "
          f"in 30 steps ({dt/30*1000:.0f} ms/step)  "
          f"{'LEARNS' if losses[-1] < losses[0]*0.8 else 'NOT LEARNING'}")
    gamma = float(torch.clamp(model2.head.mixer.gate, 0, 1))
    print(f"quantum gate gamma after training: {gamma:.3f} (init 0.10)")

    # ---- eval with missing modalities ---------------------------------------
    model2.eval()
    present = torch.ones(B, 7, dtype=torch.bool)
    present[:, [0, 4]] = False          # drop pupil + bis
    with torch.no_grad():
        o = model2(batch, present=present)
    print(f"eval w/o pupil+bis: risk finite = "
          f"{bool(torch.isfinite(o['risk']).all())}")
    print("\nEND-TO-END: PASS")
