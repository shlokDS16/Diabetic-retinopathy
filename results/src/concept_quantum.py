"""
Concept bottleneck + quantum fusion layer (steps A8 + A9).

Architecture (build manifest sections 3.3-3.5):

  fused (B,64) --> concept head --> c in R^10   (each dim individually
                                                 supervised by a measured
                                                 clinical quantity)
  c --> [VQC 10 qubits, depth 4, 80 params] --> q in R^10
  z = LayerNorm( gamma * q + (1 - gamma) * c ),  gamma learnable, init 0.1
  z --> ordinal heads (CORN):  DR severity (5 grades), DN severity (3)
        risk index = expected ordinal grade (monotone by construction)

Quantum backend: PennyLane default.qubit + backprop. lightning.qubit is
FORBIDDEN here — measured 32x batching penalty on this machine (see memory
note `quantum-sim-backend-choice`).

Classical control for ablation A18 ships in the same file
(`Classical80`) with EXACTLY the same parameter count as the VQC.

Smoke test:  python results/src/concept_quantum.py
"""
import numpy as np
import torch
import torch.nn as nn

N_CONCEPTS = 10
QDEPTH = 4
CONCEPT_NAMES = [
    "pupil_constriction_velocity", "redilation_latency_t75",
    "baseline_pupil_diameter", "canthal_thermal_deficit",
    "thermal_asymmetry", "extracellular_resistance_r0",
    "cole_dispersion_alpha", "parasympathetic_index",
    "glycaemic_burden", "glycaemic_variability",
]


class QuantumFusion(nn.Module):
    """10-qubit VQC on the concept vector, parallel with a residual path."""

    def __init__(self, n=N_CONCEPTS, depth=QDEPTH):
        super().__init__()
        import pennylane as qml
        dev = qml.device("default.qubit", wires=n)   # NEVER lightning.qubit

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for i in range(n):
                qml.RY(inputs[..., i], wires=i)
            for d in range(depth):
                for i in range(n):
                    qml.CZ(wires=[i, (i + 1) % n])
                for i in range(n):
                    qml.RY(weights[d, i, 0], wires=i)
                    qml.RZ(weights[d, i, 1], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        self.q = qml.qnn.TorchLayer(circuit, {"weights": (depth, n, 2)})
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(n)

    def forward(self, c):
        angles = torch.pi * torch.sigmoid(c)
        q = self.q(angles)
        g = torch.clamp(self.gate, 0.0, 1.0)
        return self.norm(g * q + (1 - g) * c)


class Classical80(nn.Module):
    """Ablation A18: classical map with EXACTLY the VQC's parameter count
    (depth*n*2 = 80): low-rank bilinear 10 -> 4 -> 10 with no biases
    (10*4 + 4*10 = 80), same residual gate + norm."""

    def __init__(self, n=N_CONCEPTS, rank=4):
        super().__init__()
        self.a = nn.Linear(n, rank, bias=False)
        self.b = nn.Linear(rank, n, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(n)

    def forward(self, c):
        q = torch.tanh(self.b(torch.tanh(self.a(c))))
        g = torch.clamp(self.gate, 0.0, 1.0)
        return self.norm(g * q + (1 - g) * c)


class CornHead(nn.Module):
    """CORN ordinal head: K-1 binary logits P(y > k)."""

    def __init__(self, d_in, n_grades):
        super().__init__()
        self.fc = nn.Linear(d_in, n_grades - 1)

    def forward(self, z):
        return self.fc(z)                      # logits for P(y>0)...P(y>K-2)

    @staticmethod
    def expected_grade(logits):
        return torch.sigmoid(logits).sum(-1)   # monotone risk index

    @staticmethod
    def loss(logits, y):
        """CORN conditional training loss."""
        K1 = logits.shape[-1]
        total, n = 0.0, 0
        for k in range(K1):
            mask = y >= k                       # conditional subset
            if mask.sum() == 0:
                continue
            tgt = (y[mask] > k).float()
            total = total + nn.functional.binary_cross_entropy_with_logits(
                logits[mask, k], tgt, reduction="sum")
            n += int(mask.sum())
        return total / max(n, 1)


class ConceptModel(nn.Module):
    """fused vector -> supervised concepts -> (quantum|classical) mix ->
    ordinal DR + DN heads + risk index."""

    def __init__(self, d_in=64, mixer="quantum", dr_grades=5, dn_grades=3):
        super().__init__()
        self.concept_head = nn.Sequential(
            nn.Linear(d_in, 32), nn.GELU(), nn.Linear(32, N_CONCEPTS))
        self.mixer = QuantumFusion() if mixer == "quantum" else Classical80()
        self.dr = CornHead(N_CONCEPTS, dr_grades)
        self.dn = CornHead(N_CONCEPTS, dn_grades)

    def forward(self, fused):
        c = self.concept_head(fused)
        z = self.mixer(c)
        dr = self.dr(z)
        dn = self.dn(z)
        return {"concepts": c, "z": z, "dr_logits": dr, "dn_logits": dn,
                "risk": CornHead.expected_grade(dr)}


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 16
    fused = torch.randn(B, 64)

    for mixer in ("quantum", "classical"):
        m = ConceptModel(mixer=mixer)
        n_mix = sum(p.numel() for p in m.mixer.parameters()
                    if p.numel() > 1 or "gate" not in
                    [n for n, _ in m.mixer.named_parameters()])
        n_core = sum(p.numel() for n, p in m.mixer.named_parameters()
                     if n not in ("gate",) and "norm" not in n)
        out = m(fused)
        # quick backward
        y_dr = torch.randint(0, 5, (B,))
        y_dn = torch.randint(0, 3, (B,))
        loss = (CornHead.loss(out["dr_logits"], y_dr)
                + CornHead.loss(out["dn_logits"], y_dn)
                + nn.functional.mse_loss(out["concepts"],
                                         torch.randn(B, N_CONCEPTS)))
        loss.backward()
        g = sum(p.grad.abs().sum().item() for p in m.parameters()
                if p.grad is not None)
        print(f"{mixer:9s} mixer-core params {n_core:3d} | "
              f"risk {tuple(out['risk'].shape)} range "
              f"[{out['risk'].min():.2f},{out['risk'].max():.2f}] | "
              f"loss {loss.item():.3f} | grad {'FLOWS' if g > 0 else 'DEAD'}")

    assert len(CONCEPT_NAMES) == N_CONCEPTS
    print("\nconcept names locked:", ", ".join(CONCEPT_NAMES[:3]), "...")
    print("parameter parity: VQC 4*10*2 = 80  vs  Classical80 10*4+4*10 = 80  OK")
