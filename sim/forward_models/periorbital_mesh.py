"""
3D periorbital anatomical mesh -- the ONE shared substrate for the thermal,
optical and electrical models.

Built parametrically with gmsh/OCC from the same anthropometry as
sim/cad/sensor_layout.py, so every sensor pose lands on this mesh by
construction. Coordinates are the project frame (mm):
    origin = midpoint between pupils at the corneal plane
    +X wearer's left, +Y anterior, +Z superior
Only the wearer's LEFT half (x >= 0) is meshed; x = 0 is a symmetry plane.

Regions (physical volumes) -- names are the keys of tissue_db.json:
    skin, fat_subcutaneous, muscle, bone_cortical, brain_grey (deep core),
    eyelid, orbital_fat, cornea, aqueous, lens, vitreous, sclera,
    artery_angular, artery_temporal
Boundary surfaces (physical surfaces):
    skin_exposed      air-facing skin (head + eyelids)
    cornea_exposed    cornea inside the palpebral fissure (tear film)
    conj_exposed      exposed sclera / fornix inside the fissure
    symmetry          x = 0 plane
    cut               artificial domain-clip planes (adiabatic, >= 30 mm from any sensor)
    artery_angular_wall, artery_temporal_wall   vessel lumen walls (Dirichlet T_art)

Geometry approach: only the outer envelope, the palpebral fissure, the globe
spheres, the lens and the artery pipes are CAD entities (conformal surfaces).
Tissue layers, orbit cavity and lids are assigned per element by an analytic
classifier after meshing -- robust, and exactly reproducible on any refinement.

    Head        ellipsoid centred (0, -83, -15), semi-axes (72, 88, 105).
                Chosen so the skin at the medial canthus sits at y ~ +2
                (sensor_layout's assumption) and the superficial temporal
                artery site lies on the surface at x ~ 69, y = -78.
    Layers      nested ellipsoids: skin 1.5, fat 3.0, muscle 4.5, bone 6.0 mm,
                everything deeper = perfused core (brain_grey).
    Orbit       cone from the rim (y = +8, r = 26) to the apex (y = -45, r = 4).
    Lid pad     periorbital soft-tissue bulge = cone ∩ (ellipsoid grown 5 mm
                in y) ∩ {x >= 20}; puts the lids at the corneal plane.
    Brow ridge  (anatomy v2, 2026-09-04) ellipsoid centred (0, -18, 22),
                semi-axes (62, 30, 12): the supraorbital skin the brow-bumper
                electrodes touch (skin 1.5 / fat to 6 mm / bone_cortical).
    Nose        ellipsoid centred (0, 0, -12), semi-axes (11, 18, 24): the
                nasal sidewall the nose-pad electrodes touch (skin 1.5 / fat).
                Envelope = fuse(head, pad, brow, nose) ∩ box.
    Fissure     elliptic cylinder 28 x 11 mm along y, removed from the lids.
    Globe       Scott (1988) schematic eye in 3D (same numbers as eye_mesh.py).
    Arteries    pipes r = 0.75 (angular) and r = 1.0 (temporal) swept along
                splines seated 3.0 mm below the local skin surface.

Run:  python sim/forward_models/periorbital_mesh.py [--coarse]
"""
import sys, json, pathlib, time
import numpy as np
import gmsh

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cad"))
import sensor_layout as L                                  # noqa: E402

OUT = HERE.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- head -----
HEAD_C  = (0.0, -83.0, -15.0)
HEAD_A  = (72.0, 88.0, 105.0)          # semi-axes x, y, z
T_SKIN, T_FAT, T_MUSCLE, T_BONE = 1.5, 3.0, 4.5, 6.0
PAD_DY  = 5.0                          # lid pad = ellipsoid grown in y
PAD_XMIN = 20.0                        # pad starts lateral of the medial canthus

# ------------------------------------------------- bulges (anatomy v2) -----
BROW_C, BROW_A = (0.0, -18.0, 22.0), (62.0, 30.0, 12.0)
NOSE_C, NOSE_A = (0.0, 0.0, -12.0), (11.0, 18.0, 24.0)
BULGE_T_SKIN, BULGE_T_FAT = 1.5, 6.0   # outside the head ellipsoid: skin to 1.5, fat to 6 mm
# electrode contact points that must lie on the skin (sensor_layout, left eye)
CONTACTS = {"brow_med": (21.5, 10.0, 21.0), "brow_lat": (41.5, 4.1, 21.0),
            "nose_sup": (7.5, 10.8, -2.0), "nose_inf": (8.5, 11.3, -10.0)}

# ------------------------------------------------------------- domain ------
BOX = ((0.0, 95.0), (-115.0, 30.0), (-55.0, 55.0))

# -------------------------------------------------------------- orbit ------
PX = L.IPD / 2.0                       # pupil x
ORBIT_FRONT_Y, ORBIT_FRONT_R = 8.0, 26.0
ORBIT_APEX_Y,  ORBIT_APEX_R  = -45.0, 4.0
LID_Y = -7.5                           # lids occupy y >= LID_Y inside the orbit
FISSURE_RX, FISSURE_RZ = 14.0, 5.5     # palpebral fissure semi-axes

# -------------------------------------------------------------- globe ------
# Scott (1988) schematic eye, posterior = -y here (eye_mesh.py used +z).
GLOBE_R, GLOBE_CY   = 12.0, -12.4
CORNEA_R, CORNEA_CY = 7.8, -7.8
SHELL_T, CORNEA_T   = 1.0, 0.52
LENS_CY, LENS_RR, LENS_RY = -7.3, 4.6, 2.0
ACD = 3.0                              # limbus plane at y = -ACD

# ----------------------------------------------------------- arteries ------
ANG_R, TMP_R = 0.75, 1.0
ANG_X = 14.0
TMP_Y = L.TEMPLE_ARTERY_Y              # -78
VESSEL_DEPTH = 3.0                     # centreline depth; r <= 1 keeps it inside the fat layer (1.5-4.5 mm)

REGIONS = ["artery_angular", "artery_temporal",
           "cornea", "sclera", "lens", "aqueous", "vitreous",
           "eyelid", "orbital_fat",
           "skin", "fat_subcutaneous", "muscle", "bone_cortical", "brain_grey"]


def ellipsoid(occ, c, a):
    t = occ.addSphere(*c, 1.0)
    occ.dilate([(3, t)], *c, *a)
    return (3, t)


def head_shell(occ, t_out, t_in):
    """Ellipsoid shell between depth t_out and t_in below the head surface."""
    outer = ellipsoid(occ, HEAD_C, tuple(a - t_out for a in HEAD_A))
    if t_in is None:
        return [outer]
    inner = ellipsoid(occ, HEAD_C, tuple(a - t_in for a in HEAD_A))
    out, _ = occ.cut([outer], [inner], removeObject=True, removeTool=True)
    return out


def surface_y(x, z):
    """Head-ellipsoid surface y at (x, z) -- used to seat the vessels."""
    cx, cy, cz = HEAD_C
    a, b, c = HEAD_A
    s = 1.0 - ((x - cx) / a) ** 2 - ((z - cz) / c) ** 2
    return cy + b * np.sqrt(max(s, 0.0))


def surface_x(y, z):
    cx, cy, cz = HEAD_C
    a, b, c = HEAD_A
    s = 1.0 - ((y - cy) / b) ** 2 - ((z - cz) / c) ** 2
    return cx + a * np.sqrt(max(s, 0.0))


def build(coarse=False, write=True):
    t0 = time.time()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("periorbital")
    occ = gmsh.model.occ

    box = (3, occ.addBox(BOX[0][0], BOX[1][0], BOX[2][0],
                         BOX[0][1] - BOX[0][0], BOX[1][1] - BOX[1][0],
                         BOX[2][1] - BOX[2][0]))

    # ---- outer soft-tissue envelope: head ellipsoid + periorbital lid pad ----
    cone = (3, occ.addCone(PX, ORBIT_FRONT_Y, 0.0, 0.0, ORBIT_APEX_Y - ORBIT_FRONT_Y, 0.0,
                           ORBIT_FRONT_R, ORBIT_APEX_R))
    pad_ell = ellipsoid(occ, HEAD_C, (HEAD_A[0], HEAD_A[1] + PAD_DY, HEAD_A[2]))
    pad_half = (3, occ.addBox(PAD_XMIN, -200, -200, 400, 400, 400))
    pad, _ = occ.intersect([cone], [pad_ell], removeObject=True, removeTool=True)
    pad, _ = occ.intersect(pad, [pad_half], removeObject=True, removeTool=True)
    head_out = ellipsoid(occ, HEAD_C, HEAD_A)
    envelope, _ = occ.fuse([head_out], pad, removeObject=True, removeTool=True)
    # anatomy v2: brow ridge + nose, fused one at a time (dilated spheres, like the head)
    for c, a in ((BROW_C, BROW_A), (NOSE_C, NOSE_A)):
        envelope, _ = occ.fuse(envelope, [ellipsoid(occ, c, a)], removeObject=True, removeTool=True)
    envelope, _ = occ.intersect(envelope, [box], removeObject=True, removeTool=True)

    # ---- globe (Scott schematic eye) ----------------------------------------
    def sph(cy, r):
        return (3, occ.addSphere(PX, cy, 0.0, r))

    outer_g, _ = occ.fuse([sph(GLOBE_CY, GLOBE_R)], [sph(CORNEA_CY, CORNEA_R)],
                          removeObject=True, removeTool=True)
    inner_g, _ = occ.fuse([sph(GLOBE_CY, GLOBE_R - SHELL_T)],
                          [sph(CORNEA_CY - CORNEA_T, CORNEA_R - CORNEA_T)],
                          removeObject=True, removeTool=True)
    lens = (3, occ.addSphere(PX, LENS_CY, 0.0, 1.0))
    occ.dilate([lens], PX, LENS_CY, 0.0, LENS_RR, LENS_RY, LENS_RR)

    # ---- palpebral fissure: air column in front of the globe -----------------
    fissure = (3, occ.addCylinder(PX, LID_Y, 0.0, 0.0, 40.0, 0.0, 1.0))
    occ.dilate([fissure], PX, LID_Y, 0.0, FISSURE_RX, 1.0, FISSURE_RZ)

    # ---- arteries: pipes seated 3 mm below the local skin surface -------------
    def seated_pipe(points, r):
        """Vessel as ONE straight cylinder between the two seated end points.
        Swept pipes (periodic seam) and fused segment chains (sliver arcs at
        the joints) both break the surface mesher; a single cylinder is clean.
        Depth therefore varies along the vessel (see summary 'checks')."""
        p0, p1 = np.asarray(points[0]), np.asarray(points[-1])
        return [(3, occ.addCylinder(*p0, *(p1 - p0), r))]

    ang_pts = [(ANG_X, surface_y(ANG_X, z) - VESSEL_DEPTH, z) for z in (-15.0, 15.0)]
    # temporal artery: 35 mm span centred on the PPG site (z = 12); the chord
    # sits <= 1.2 mm deeper than the surface-offset curve at mid-span
    tmp_pts = [(surface_x(TMP_Y, z) - VESSEL_DEPTH, TMP_Y, z) for z in (-5.0, 30.0)]
    ang = seated_pipe(ang_pts, ANG_R)
    tmp = seated_pipe(tmp_pts, TMP_R)

    # ---- one fragment: conformal globe shell, lens, vessel walls, fissure ----
    # Tissue layers, orbit and lids are NOT CAD entities: they are assigned per
    # element by the analytic classifier below (interfaces resolved at mesh
    # resolution, which the mesh-size fields keep <= 1.4 mm around the orbit).
    occ.synchronize()
    tools = [fissure] + outer_g + inner_g + [lens] + ang + tmp
    _, fmap = occ.fragment(envelope, tools)
    occ.synchronize()
    n_env = len(envelope)
    env_set = {t for outs in fmap[:n_env] for d, t in outs if d == 3}
    fiss_set = {t for d, t in fmap[n_env] if d == 3}
    globe_set = {t for outs in fmap[n_env + 1:n_env + 1 + len(outer_g)] for d, t in outs if d == 3}
    air = fiss_set - globe_set
    keep = (env_set | globe_set) - air
    drop = [(3, t) for d, t in gmsh.model.getEntities(3) if t not in keep]
    occ.remove(drop, recursive=True)
    occ.synchronize()

    # ---- mesh size fields ---------------------------------------------------
    f = gmsh.model.mesh.field
    scale = 2.0 if coarse else 1.0

    def ball(cx, cy, cz, r, vin, vout, thick):
        b = f.add("Ball")
        f.setNumber(b, "XCenter", cx); f.setNumber(b, "YCenter", cy); f.setNumber(b, "ZCenter", cz)
        f.setNumber(b, "Radius", r); f.setNumber(b, "VIn", vin * scale)
        f.setNumber(b, "VOut", vout * scale); f.setNumber(b, "Thickness", thick)
        return b

    fields = [
        ball(PX, GLOBE_CY, 0.0, 16.0, 0.7, 3.0, 10.0),                 # globe
        ball(PX, -12.0, 0.0, 40.0, 1.4, 3.0, 15.0),                    # orbit + lids
        ball(PX - L.MEDIAL_CANTHUS_DX, 2.0, -2.0, 14.0, 0.9, 3.0, 8.0),  # canthus / angular artery
        ball(66.0, TMP_Y, 12.0, 28.0, 1.2, 3.0, 12.0),                 # temple / PPG site
        ball(31.5, 7.0, 21.0, 16.0, 1.0, 3.0, 8.0),                    # brow electrodes
        ball(8.0, 11.0, -6.0, 10.0, 0.9, 3.0, 6.0),                    # nose electrodes
    ]
    fmin = f.add("Min")
    f.setNumbers(fmin, "FieldsList", fields)
    f.setAsBackgroundMesh(fmin)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0 * scale)
    gmsh.option.setNumber("Mesh.Algorithm", 6)        # Frontal-Delaunay 2D
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)     # HXT
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

    t1 = time.time()
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:                      # HXT is strict; Delaunay is tolerant
        print(f"  HXT failed ({str(e)[:60]}...); retrying with Delaunay")
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)

    # ---- classify elements analytically (robust to entity re-tagging) --------
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    P = coords.reshape(-1, 3)
    idx = {int(t): i for i, t in enumerate(ntags)}
    remap = np.vectorize(idx.get)

    et, etags, enodes = gmsh.model.mesh.getElements(3)
    tets = np.concatenate([remap(np.asarray(n).reshape(-1, 4)) for n in enodes]).astype(np.int64)
    ct = P[tets].mean(axis=1)

    st, stags, snodes = gmsh.model.mesh.getElements(2)
    tris = np.concatenate([remap(np.asarray(n).reshape(-1, 3)) for n in snodes]).astype(np.int64)

    region = classify_volume(ct, ang_pts, tmp_pts)

    # boundary triangles = faces shared by exactly one tet; artery walls = faces
    # between artery and tissue tets (interior, but needed for Dirichlet)
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                                    tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    owner_tet = np.tile(np.arange(len(tets)), 4)
    fkeys = {}
    for f, o in zip(map(tuple, faces), owner_tet):
        fkeys.setdefault(f, []).append(int(o))
    key = np.sort(tris, axis=1)
    tri_region = np.zeros(len(tris), dtype=np.int64)
    tri_keep = np.zeros(len(tris), dtype=bool)
    surf_id = {"skin_exposed": 101, "cornea_exposed": 102, "conj_exposed": 103,
               "symmetry": 104, "cut": 105, "artery_angular_wall": 106,
               "artery_temporal_wall": 107}
    cs = P[tris].mean(axis=1)
    rid_ang = REGIONS.index("artery_angular") + 1
    rid_tmp = REGIONS.index("artery_temporal") + 1
    for i, k in enumerate(map(tuple, key)):
        owners = fkeys.get(k, [])
        regs = {int(region[o]) for o in owners}
        if len(owners) == 2 and rid_ang in regs and len(regs) == 2:
            tri_region[i] = surf_id["artery_angular_wall"]; tri_keep[i] = True
        elif len(owners) == 2 and rid_tmp in regs and len(regs) == 2:
            tri_region[i] = surf_id["artery_temporal_wall"]; tri_keep[i] = True
        elif len(owners) == 1:
            tri_region[i] = surf_id[classify_boundary(cs[i], region[owners[0]])]
            tri_keep[i] = True
    tris, tri_region = tris[tri_keep], tri_region[tri_keep]

    # ---- write with meshio (physical names carried as field_data) -----------
    import meshio
    field_data = {nm: [i + 1, 3] for i, nm in enumerate(REGIONS)}
    field_data.update({nm: [tg, 2] for nm, tg in surf_id.items()})
    mesh = meshio.Mesh(P, [("tetra", tets), ("triangle", tris)],
                       cell_data={"gmsh:physical": [region, tri_region],
                                  "gmsh:geometrical": [region, tri_region]},
                       field_data=field_data)
    t2 = time.time()

    # ---- summary -------------------------------------------------------------
    vol = np.abs(np.einsum("ij,ij->i", np.cross(P[tets[:, 1]] - P[tets[:, 0]],
                                                 P[tets[:, 2]] - P[tets[:, 0]]),
                           P[tets[:, 3]] - P[tets[:, 0]])) / 6.0
    summary = {
        "mesh_file": str(OUT / ("periorbital_coarse.msh" if coarse else "periorbital.msh")),
        "units": "mm", "frame": "sensor_layout (origin between pupils, +Y anterior, +Z up)",
        "nodes": int(len(P)), "tets": int(len(tets)), "boundary_tris": int(len(tris)),
        "regions": {nm: {"physical_id": i + 1,
                         "volume_mm3": float(vol[region == i + 1].sum()),
                         "n_tets": int((region == i + 1).sum())}
                    for i, nm in enumerate(REGIONS)},
        "surfaces": {nm: {"physical_id": tg, "n_faces": int((tri_region == tg).sum())}
                     for nm, tg in surf_id.items()},
        "head": {"centre": HEAD_C, "semi_axes": HEAD_A,
                 "layers_mm": {"skin": T_SKIN, "fat": T_FAT, "muscle": T_MUSCLE, "bone": T_BONE}},
        "bulges": {"brow": {"centre": BROW_C, "semi_axes": BROW_A},
                   "nose": {"centre": NOSE_C, "semi_axes": NOSE_A},
                   "layers_mm": {"skin": BULGE_T_SKIN, "fat": BULGE_T_FAT}},
        "checks": {
            "canthus_surface_y_mm": float(surface_y(PX - L.MEDIAL_CANTHUS_DX, L.MEDIAL_CANTHUS_DZ)),
            "temple_surface_x_mm": float(surface_x(TMP_Y, L.TEMPLE_ARTERY_Z)),
            "unclassified_tets": int((region == 0).sum()),
            "contact_to_skin_mm": contact_distances(cs[tri_keep][tri_region == surf_id["skin_exposed"]]),
        },
        "timing_s": {"geometry": t1 - t0, "meshing": t2 - t1},
    }
    if write:
        meshio.write(summary["mesh_file"], mesh, file_format="gmsh22", binary=True)   # binary loads ~10x faster
        (OUT / ("periorbital_coarse_summary.json" if coarse else "periorbital_summary.json")
         ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gmsh.finalize()
    return summary


def contact_distances(skin_centroids):
    """Distance from each electrode contact point to the nearest skin_exposed facet centroid."""
    from scipy.spatial import cKDTree
    tree = cKDTree(skin_centroids)
    return {k: float(tree.query(np.asarray(v))[0]) for k, v in CONTACTS.items()}


def _bulge(p, t):
    """(inside brow, inside nose) shrunk by t mm on every semi-axis."""
    return (_ell(p, BROW_C, tuple(a - t for a in BROW_A)) <= 1.0,
            _ell(p, NOSE_C, tuple(a - t for a in NOSE_A)) <= 1.0)


def _ell(p, c, a):
    return (((p[:, 0] - c[0]) / a[0]) ** 2 + ((p[:, 1] - c[1]) / a[1]) ** 2
            + ((p[:, 2] - c[2]) / a[2]) ** 2)


def _pipe_dist(p, pts):
    """Distance from points to a vessel centreline parametrised by z."""
    pts = np.asarray(pts)
    x = np.interp(p[:, 2], pts[:, 2], pts[:, 0])
    y = np.interp(p[:, 2], pts[:, 2], pts[:, 1])
    inside_z = (p[:, 2] >= pts[0, 2] - 1e-6) & (p[:, 2] <= pts[-1, 2] + 1e-6)
    d = np.hypot(p[:, 0] - x, p[:, 1] - y)
    return np.where(inside_z, d, np.inf)


def _cone_r(y):
    return ORBIT_APEX_R + (ORBIT_FRONT_R - ORBIT_APEX_R) * (y - ORBIT_APEX_Y) / (ORBIT_FRONT_Y - ORBIT_APEX_Y)


def classify_volume(p, ang_pts, tmp_pts):
    """Region id (1-based index into REGIONS) for each point, by analytic geometry."""
    rid = {nm: i + 1 for i, nm in enumerate(REGIONS)}
    out = np.zeros(len(p), dtype=np.int64)
    x, y, z = p.T

    # arteries first (highest priority)
    out[_pipe_dist(p, ang_pts) < ANG_R] = rid["artery_angular"]
    out[_pipe_dist(p, tmp_pts) < TMP_R] = rid["artery_temporal"]

    # globe
    dg = np.sqrt((x - PX) ** 2 + (y - GLOBE_CY) ** 2 + z ** 2)
    dc = np.sqrt((x - PX) ** 2 + (y - CORNEA_CY) ** 2 + z ** 2)
    outer = (dg <= GLOBE_R) | (dc <= CORNEA_R)
    inner = (dg <= GLOBE_R - SHELL_T) | (dc <= CORNEA_R - CORNEA_T)
    shell = outer & ~inner
    lens = (((x - PX) / LENS_RR) ** 2 + ((y - LENS_CY) / LENS_RY) ** 2 + (z / LENS_RR) ** 2) <= 1.0
    g = out == 0
    out[g & shell & (y >= -ACD)] = rid["cornea"]
    out[g & shell & (y < -ACD)] = rid["sclera"]
    out[g & inner & lens] = rid["lens"]
    out[g & inner & ~lens & (y >= LENS_CY)] = rid["aqueous"]
    out[g & inner & ~lens & (y < LENS_CY)] = rid["vitreous"]

    # bulges (anatomy v2): points OUTSIDE the head ellipsoid but inside the brow
    # ridge / nose. Layers by the semi-axes-minus-t test on the bulge itself:
    # skin to 1.5 mm, fat to 6 mm, then bone (brow) / fat (nose). Takes priority
    # over the orbit cone so the brow caps the upper-lid pad.
    g = out == 0
    in_head = _ell(p, HEAD_C, HEAD_A) <= 1.0
    b0, n0 = _bulge(p, 0.0)
    b1, n1 = _bulge(p, BULGE_T_SKIN)
    b2, n2 = _bulge(p, BULGE_T_FAT)
    ext = g & ~in_head & (b0 | n0)
    out[ext & ~(b1 | n1)] = rid["skin"]
    out[ext & (b1 | n1) & ~(b2 | n2)] = rid["fat_subcutaneous"]
    out[ext & b2] = rid["bone_cortical"]
    out[ext & n2 & ~b2] = rid["fat_subcutaneous"]

    # orbit (cone) -> lids / orbital fat
    in_cone = (y <= ORBIT_FRONT_Y) & (y >= ORBIT_APEX_Y) & (np.hypot(x - PX, z) <= _cone_r(y))
    g = out == 0
    out[g & in_cone & (y >= LID_Y)] = rid["eyelid"]
    out[g & in_cone & (y < LID_Y)] = rid["orbital_fat"]

    # head layers by depth below the OUTER surface: a point is deeper than t if
    # it is inside the head OR a bulge shrunk by t (so no skin layer is left
    # buried under the brow / nose where the head surface is no longer exposed)
    g = out == 0
    d1 = T_SKIN
    d2 = d1 + T_FAT
    d3 = d2 + T_MUSCLE
    d4 = d3 + T_BONE

    def inside(t):
        b, n = _bulge(p, t)
        return (_ell(p, HEAD_C, tuple(a - t for a in HEAD_A)) <= 1.0) | b | n

    i1, i2, i3, i4 = inside(d1), inside(d2), inside(d3), inside(d4)
    out[g & ~i1] = rid["skin"]
    out[g & i1 & ~i2] = rid["fat_subcutaneous"]
    out[g & i2 & ~i3] = rid["muscle"]
    out[g & i3 & ~i4] = rid["bone_cortical"]
    out[g & i4] = rid["brain_grey"]
    return out


def classify_boundary(c, region_id):
    """Name of the boundary surface a boundary-triangle centroid lies on."""
    x, y, z = c
    eps = 0.3
    if abs(x - BOX[0][0]) < eps:
        return "symmetry"
    if (abs(x - BOX[0][1]) < eps or abs(y - BOX[1][0]) < eps or abs(y - BOX[1][1]) < eps
            or abs(z - BOX[2][0]) < eps or abs(z - BOX[2][1]) < eps):
        return "cut"
    name = REGIONS[region_id - 1]
    if name == "cornea":
        return "cornea_exposed"
    if name in ("sclera", "orbital_fat", "aqueous", "vitreous", "lens"):
        return "conj_exposed"
    return "skin_exposed"


if __name__ == "__main__":
    coarse = "--coarse" in sys.argv
    s = build(coarse=coarse)
    print(f"wrote {s['mesh_file']}")
    print(f"nodes {s['nodes']:,}   tets {s['tets']:,}   boundary tris {s['boundary_tris']:,}   "
          f"geometry {s['timing_s']['geometry']:.1f}s  meshing+classify {s['timing_s']['meshing']:.1f}s")
    print(f"\n{'region':18s} {'vol (mm3)':>12s} {'#tets':>8s}")
    for nm in REGIONS:
        r = s["regions"][nm]
        print(f"{nm:18s} {r['volume_mm3']:12.1f} {r['n_tets']:8d}{'   <-- EMPTY' if r['n_tets'] == 0 else ''}")
    print(f"\n{'surface':22s} {'#faces':>6s}")
    for nm, r in s["surfaces"].items():
        print(f"{nm:22s} {r['n_faces']:6d}")
    print(f"\nchecks: {s['checks']}")
    print("contact -> nearest skin_exposed facet centroid (mm): "
          + ", ".join(f"{k} {v:.2f}" for k, v in s["checks"]["contact_to_skin_mm"].items()))
