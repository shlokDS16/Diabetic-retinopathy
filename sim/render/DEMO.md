# NRDI Blender live-demo kit

Everything below opens in **Blender 5.2 LTS** (`C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`).
The `.blend` files are self-contained (HDRI and textures are packed), so they can be copied to
another machine with nothing else. Renders use Cycles on the GPU (OptiX); the viewport works on any GPU.

Fastest start: double-click `sim/render/open_blender_demo.bat` (opens `explode.blend`).

## The five scenes

| File (`sim/render/out/`) | What it shows | Built by |
|---|---|---|
| `explode.blend` | 31-part exploded assembly, 170-frame timeline, leader lines from every sensor to its mount ring on the frame, camera-facing callouts (callout + part number) | `blender_explode.py` |
| `hero_product.blend` | product still, 3/4 macro on the cyclorama | `blender_hero.py --shot product` |
| `hero_face.blend` | glasses on the photogrammetry head (Lee Perry-Smith, CC BY 3.0), pupils on the optical axes | `blender_hero.py --shot face` |
| `physics_thermography.blend` | "image 2": the 3D Pennes skin temperature on the real face, inferno 32.5-35.8 C, MLX90614 DCI (5 deg) and BAA (90 deg) cones, colour bar | `blender_physics.py --shot thermography` |
| `physics_thermal.blend` | sagittal cutaway through the globe with the temperature slice and the thermopile FOV cone | `blender_physics.py --shot thermal` |

## Opening a scene

1. `File > Open` (Ctrl+O) and pick the `.blend`, or double-click the file in Explorer.
2. If a dialog asks about *Load UI*, leave it ticked; the file opens on the camera view.
3. If the view is not through the camera: hover the 3D viewport and press `Numpad 0`
   (`View > Cameras > Active Camera`).
4. Render settings (resolution, samples) are stored in the file. The GPU device is a user
   preference, not stored in the .blend: once per machine check
   `Edit > Preferences > System > Cycles Render Devices` is on **OptiX** with the RTX ticked.

## Playing the explode timeline (`explode.blend`)

- The Timeline editor is at the bottom (frames 0-170, 30 fps). Press **Space** with the mouse over
  the 3D viewport to play/pause, or click the play button in the timeline header.
- Drag the blue frame cursor to scrub: 0-20 assembled, 20-90 explode (lenses, then frame, then
  sensors, then electronics), 90-110 exploded hold, 110-170 reassemble.
- Leader lines, mount rings and labels are keyed to fade in with each part and are hidden while
  assembled. They live in the `annotations` collection (Outliner, top right): untick its eye icon
  for the bare explode, or tick/untick individual `leader_*`, `mount_*`, `label_*` objects.
- `Shift+Left` jumps to frame 0, `Shift+Right` to frame 170. `Ctrl+Space` maximises the viewport.
- Orbit: middle-mouse drag (or Alt+left drag with *Emulate 3 Button Mouse* enabled); wheel to zoom.
  Leaving the camera view is fine; `Numpad 0` returns to it.

## Rendered viewport shading (live Cycles / OptiX)

1. In the 3D viewport header, top right, are four shading circles: Wireframe, Solid,
   **Material Preview**, **Rendered**. Click the last one (or press `Z` and pick Rendered).
2. *Material Preview* (EEVEE) is instant and is the safe choice while scrubbing the timeline;
   *Rendered* runs Cycles progressively on the GPU: noisy for about a second, clean after a few.
3. To limit viewport cost: Properties editor > Render tab > Sampling > Viewport > Max Samples 64,
   Denoise ticked.
4. Full-quality still: `F12` (Render > Render Image). Animation: `Ctrl+F12` writes PNGs to the path
   in Output properties; encode with `python sim/render/encode_mp4.py <dir> <name.mp4>`.

## What is physics output and what is a render (the pipeline for the panel)

```
sim/cad/*.py  ->  sim/out/parts_stl/*.stl + parts.json           geometry + explode manifest (CAD)
                          |
sim/forward_models/periorbital_mesh.py -> sim/out/periorbital.msh                    tetra mesh of the periorbital anatomy
sim/forward_models/thermal3d.py        -> sim/out/thermal3d_full.json, thermal3d_T_full.npy   Pennes solve (solver output)
sim/forward_models/mcx_run.py          -> sim/out/ppg_banana_880.npy, optical_results.json    Monte-Carlo photon transport
sim/forward_models/bioimpedance3d.py   -> sim/out/bioimpedance3d_coarse.json, bioimp_leadfield_coarse.npy
                          |
sim/render/fields_export.py      -> sim/out/fields/skin_T.ply, slice_T_*.png, banana_880.npy/.vdb, leadfield.ply
sim/render/thermography_face.py  -> sim/out/fields/face_T.ply, colorbar_inferno.png   (T projected on the head scan)
                          |
sim/render/blender_*.py (headless Blender 5.2, Cycles/OptiX) -> sim/render/out/*.png, *.blend, explode_seq_cycles/
sim/render/encode_mp4.py                                     -> sim/render/out/explode_cycles.mp4
```

| Kind | Files |
|---|---|
| **Physics outputs** (solver results; the numbers the paper quotes) | `sim/out/thermal3d_full.json`, `thermal3d_T_full.npy`, `thermal3d_full.vtu`, `optical_results.json`, `ppg_banana_*.npy`, `led_*_940.npy`, `bioimpedance3d_coarse.json`, `bioimp_leadfield_coarse.npy`, `opto_thermal.json`, `pupil_model.json`, `verification.json`, `surrogate_thermal.json`, `uq_thermal_samples.npz` |
| **Field bridges** (physics re-sampled onto render surfaces; the colours are the data) | `sim/out/fields/skin_T.ply`, `face_T.ply`, `slice_T_z0.png`, `slice_T_x31.png`, `banana_880.npy/.vdb`, `leadfield.ply`, `palette.json` |
| **Renders** (pictures of the above) | `sim/render/out/hero_*.png`, `physics_*.png`, `explode_cycles_f*.png`, `explode_seq_cycles/*.png`, `explode_cycles.mp4` |
| **Blender scenes** (open live) | `sim/render/out/*.blend` |

Note on `face_T.ply`: the periorbital mesh covers the eye and temple region; outside it (forehead,
chin, neck) the face colour fades to a 33.2 C baseline that is *not* a solver output. On-patch
colours are the `thermal3d_full` solution, nearest node, mirrored across x = 0.

Open any `.ply` field directly (`File > Import > Stanford PLY`, scale 0.001) to show the panel that
the colours come from the solver files, not from a paint job. `thermal3d_full.vtu` opens in ParaView.

## Live demo WITHOUT the Claude artifact (2026-09-04)

Everything the panel needs to see runs in real engineering tools on this laptop. One click:
`sim\DEMO_RUN.bat` opens all four windows below. Order of showing that reads as a pipeline:

| Step | Tool (real, installed) | What to say / do |
|---|---|---|
| 1 | **gmsh GUI** — `sim/out/periorbital.msh` | "This is the finite-element mesh: 705k tetrahedra, 14 tissue regions, built from the same anthropometry as the CAD." Rotate; Tools ▸ Visibility to isolate regions. |
| 2 | **PyVista / VTK** — `python sim/render/view_fields.py` | Left: anatomy by region with the glasses and sensor poses in the same frame. Right: the Pennes temperature field; **drag the clip-plane arrow** through the globe live. This is the solver output (`thermal3d_full.vtu`), not an illustration. |
| 3 | **Blender 5.2** — `explode.blend` | Space to play the exploded assembly (leader lines to every mount point). Switch viewport to Rendered (OptiX) to show materials update live. `physics_thermography.blend` shows the same temperature field mapped onto the face. |
| 4 | **Browser (offline)** — `sim/dashboard/nrdi_twin.html` | The digital twin: move the patient sliders, readouts update through the surrogate fitted to 640 FE solves. Works with Wi-Fi off; no Claude branding. |
| 5 | **Files** — `sim/out/*.json`, `sim/out/tissue_db_sources.md` | If asked "where do the numbers come from": open `bioimpedance3d_coarse.json` or `verification.json`; every dashboard number is in these files, and every tissue property has a citation line. |

Fallbacks if a machine misbehaves: `explode_cycles.mp4` and the PNG stills in `sim/render/out/`.
