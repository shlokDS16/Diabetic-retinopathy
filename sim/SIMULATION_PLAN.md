# NRDI SIMULATION PLAN — "One Geometry, One Data Contract, Three Surfaces"

Written 2026-09-03 from three parallel research sweeps (rendering pipelines, exploded-view
techniques, Claude-integrated tools). Demo deadline: **2026-09-09**. Status: PLAN, awaiting go.

---

## 1. What the panel will see (the demo spine, ~8 minutes)

1. **Hero shot (film, 20 s).** Photoreal glasses on a face. Camera pushes in. The frame turns to
   glass, the skin becomes translucent, the orbit appears as a capped cutaway with the temperature
   field glowing inside. One continuous shot.
2. **Exploded view (film + live, 40 s).** All 15 components separate along designed vectors with
   thin-line callouts, then re-assemble onto the face. As each sensor lands, it "switches on" and
   shows what it sees: thermopile FOV cone on the medial canthus, PPG photon banana at the temple,
   electrode current streamlines through the eyelid and orbital fat, camera frustum on the pupil.
3. **Physics layers (live).** Toggle thermal / optical / electrical fields on the same anatomy.
   Every field is from our own solver on our own mesh, not an illustration.
4. **Digital twin (live).** Sliders for the patient vector (choroidal perfusion, periorbital
   ECW fraction, skin perfusion, autonomic gain, arterial stiffness). Sensor readouts update in
   milliseconds with uncertainty bands. Inverse mode: feed noisy readings, recover the patient
   vector. The "cold room vs true progression" scenario from PROJECT_MASTER §6, now derived.
5. **Design verdicts.** Two findings with physics behind them: (a) MLX90614-BAA 90° FOV cannot
   work, DCI can; (b) 940 nm-only illumination cannot evoke a pupil reflex — a stimulus LED is
   required. Panels respect a simulation that changes the BOM.
6. **Credibility table (ASME V&V 40).** Code verification (MMS), mesh convergence (GCI),
   validation vs Sodi 2009 / Chandrasekar 2021 / pupillometry table, Sobol indices.

---

## 2. Architecture

### 2.1 The single geometry
- **Glasses**: build123d (existing `sim/cad/frame.py` + `sensor_layout.py`). Every body gets a
  unique ASCII `.label` (no spaces/dots — three.js sanitises names). Vendor STEPs imported with
  `import_step`, wrapped in one Compound, exported once with `export_gltf(binary=True)` →
  `sim/out/glasses.glb`. Post-pass: `gltf-transform dedup/weld/draco` (never `join`).
- **Anatomy (physics)**: parametric 3D hemi-periorbital gmsh mesh from the same anthropometry
  (cornea, aqueous, lens, vitreous, retina-choroid-sclera shell, orbital fat, eyelids, orbital
  bone, nasal bridge, temple skin/fat; angular + superficial temporal arteries as line elements).
- **Anatomy (hero)**: Z-Anatomy / BodyParts3D (CC BY-SA 4.0) outer skin + orbit; BlendSwap
  CC0 eye shader. Physics surfaces drive colour; hero mesh only supplies skin/eye realism.
  MIDA swaps in behind the same interface when the licence arrives (scripts only, no meshes).

### 2.2 The data contract (everything reads these, nothing else)
| File | Producer | Consumers |
|---|---|---|
| `sim/out/glasses.glb` | build123d | Blender, Three.js |
| `sim/out/parts.json` | sensor_layout.py | Blender explode keyframes, GSAP timeline, callout text |
| `sim/out/tissue_db.json` | hand-curated, cited, with uncertainty ranges | all solvers, UQ |
| `sim/out/fields/*.ply` (vertex attrs: T, phase, \|J\|) | scikit-fem via meshio | Blender (Sequence Loader / PLY), Three.js vertex colours, PyVista figures |
| `sim/out/fields/*.vdb` (fluence, T grid) | numpy → openvdb (Blender's bundled module) | Blender Principled Volume, Volume-to-Mesh isosurfaces |
| `sim/out/fields/photons.npz` | MCX `debuglevel='M'` trajectories | Blender emissive curves, Three.js line segments |
| `sim/out/surrogate.json` | PCE/Sobol over patient vector | Three.js twin (in-page, ms latency), figures |
| `sim/out/palette.json` | one file: cmcrameri colormaps, HDRI name, material base colours | Blender, Three.js, matplotlib figstyle |

### 2.3 The three surfaces
1. **Hero film** — Blender 5.2 LTS, Cycles + **OptiX** (CUDA has an sm_120 bug; OptiX is the
   workaround and the right choice anyway), headless `blender -b -P script.py`. Skin = Random
   Walk (Skin) SSS; Poly Haven studio HDRI (CC0); shadow catcher; capped cutaway via collection
   Boolean; isosurfaces via Volume-to-Mesh; photon paths as emissive curves + compositor glow;
   camera Empty with keyframed orbit/dolly + DoF. Output PNG → H.264 MP4 + WebM.
2. **Live twin dashboard** — Three.js (cdnjs), react-three-fiber + drei + GSAP. `<Stage>`,
   `<Environment>` with the same HDRI, `<ContactShadows>`, `MeshTransmissionMaterial` glass,
   `<Html occlude>` callouts + `Line2` leaders, NRRD/3D-texture volume shader for fields,
   explode timeline from `parts.json`, surrogate evaluated in-page. Hosted as a Claude Artifact
   (shareable) AND served locally from the laptop (venue-internet-proof).
3. **Journal figures** — matplotlib via `results/src/figstyle.py` + PyVista (PBR/SSAO/EDL) for
   the V&V numbers. Same `palette.json`.

---

## 3. Tool selection (Claude integration first)

| Role | Tool | Claude integration | Status here |
|---|---|---|---|
| Orchestrator | Claude Code | native | running |
| Dashboard host | Claude Code Artifacts | native | available |
| Dashboard layout pass | `/design` canvas skill | native | available |
| Visual QA | Claude in Chrome | native | available |
| Photoreal render + explode look-dev | **Blender MCP (ahujasid)** — basis of Anthropic's official Blender connector; update past June-2026 CVE patch | MCP | to install (`winget install BlenderFoundation.Blender`, `uvx blender-mcp`) |
| Final renders (reproducible) | headless `bpy` scripts | scripted | with Blender |
| Geometry | build123d (existing); `build123d-mcp` optional | MCP available | installed |
| Physics | scikit-fem, gmsh, pmcx (existing venvs) | scripted | installed |
| System diagram | Excalidraw MCP / Mermaid MCP | MCP | connected |
| Slide deck | Gamma MCP | MCP | connected |
| Explainer stitching (optional) | Remotion skill (in ui-design-master) | skill | installed |
| Illustrative b-roll only, labelled | Higgsfield / OpenArt MCP | MCP | connected |

**Rejected**: Fusion 360 (Animation workspace has no API), Onshape (cannot export exploded
states via API), KeyShot (headless is paid Pro), Omniverse (launcher deprecated, heavy),
Unity/Unreal (overkill in 7 days), Ansys Student (publication ban), OASiS/Foam-Agent (Linux),
Meshy/Tripo/Hunyuan (decorative, not physical; only for props if ever).

**Component models & licences**: ESP32-S3 from Espressif KiCad library (CC BY-SA + exception);
MLX90614, MAX30102, AD5933, OV7670 from SnapEDA (CC BY-SA 4.0 + Design Exception); LEDs from
KiCad packages3D; LiPo/electrodes/LRA procedural in build123d. One credit line in the dashboard
footer covers all.

---

## 4. Day plan (today 09-03 → demo 09-09)

| Day | Physics track | Visual track | Must-have artefact |
|---|---|---|---|
| **D1 09-03** | 3D periorbital gmsh mesh + `tissue_db.json` + patient vector; port Pennes to 3D; Scott 1988 property check | Install Blender 5.2, verify OptiX on sm_120; label all bodies, `export_gltf`, `parts.json`, vendor STEPs; EEVEE test explode | `glasses.glb` + first explode test render |
| **D2 09-04** | Optical: voxelize, MCX 660/880/940, photon trajectories, LED safety (IEC 62471), opto-thermal source | Cycles materials, HDRI, shadow catcher; exploded animation rendered; PLY/VDB bridges | Exploded-view MP4 v1 |
| **D3 09-05** | Electrical: CEM complex Laplace, 1–100 kHz, lead-field map, Hanai edema, σ(T) | Thermal cutaway render; photon-path render; current streamlines | Three physics renders |
| **D4 09-06** | Pupil DDE + camera projection; HRV chain; MMS + GCI; Sobol + PCE surrogate | `surrogate.json`; field layers as vertex colours / 3D textures | V&V table + surrogate |
| **D5 09-07** | Validation metrics vs clinical | Three.js twin: explode + layers + sliders + inverse mode; Artifact + local build | Live dashboard |
| **D6 09-08** | Figures | Hero film composite; Gamma deck; rehearsal; fallbacks (MP4 + PNG stills, offline HTML) | Demo package |
| **D7 09-09** | — | Demo | — |

Stretch items dropped first if time runs short: Bayesian inverse mode (keep forward sliders),
transient thermal (keep steady), HRV chain, Remotion stitching.

**Data-gate rule stands**: the moment mBRSET/BRSET are on disk, stop wherever we are and run
A5 → training → 23-arm ablation. Every visual artefact already rendered remains usable.

---

## 5. Known constraints (from research, verified 2026-09-02)
- `bpy` PyPI wheel needs Python **3.13 exactly** (we have nrdi13). Whether it bundles OptiX
  kernels is unverified → plan on full Blender install + `blender -b`.
- Artifacts cannot reach local MCP servers or WebSockets at view time → "live" = in-page
  surrogate. Serve locally as backup.
- SciBlend / Sequence Loader / Microscopy Nodes state Blender 4.2+; if they fail on 5.2, fall
  back to Blender 4.5 LTS (Blackwell-capable since 4.4). BVtkNodes is pinned to 4.2 — skip.
- openvdb `copyFromArray` has an axis-shuffle report → test on an asymmetric array first.
- RTX 5070 Ti laptop = 12 GB: keep VDB grids ≤ 256³ float.
- Bioimpedance validation data is whole-body (Hwang 2023) → validate direction + tissue-level
  conductivities, state this openly.
