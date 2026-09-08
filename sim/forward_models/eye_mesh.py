"""
2D axisymmetric human eye mesh for the Pennes bioheat model.

Geometry follows the Scott (1988) schematic eye, the reference almost every
subsequent ocular thermal model reuses. Coordinates are (r, z): r >= 0 is
radial distance from the optical axis, z increases posteriorly from the
corneal apex at z = 0.

    PROVENANCE: layer dimensions are standard schematic-eye values and are
    marked "to verify against Scott (1988)" until the paper cites the source
    directly.

Regions are built by explicit boolean cuts -- not by fragment-and-classify,
which produced spurious regions where construction helpers extended past the
globe.

    1 cornea    2 aqueous    3 lens    4 vitreous    5 sclera/retina

Run:  python sim/forward_models/eye_mesh.py
"""
import pathlib
import numpy as np
import gmsh

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# --- schematic eye dimensions, millimetres ----------------------------------
GLOBE_R   = 12.0
GLOBE_CZ  = 12.4
CORNEA_R  = 7.8
CORNEA_CZ = 7.8
SHELL_T   = 1.0      # scleral thickness
CORNEA_T  = 0.52     # central corneal thickness
LENS_CZ   = 7.3
LENS_RR   = 4.6
LENS_RZ   = 2.0
ACD       = 3.0      # anterior chamber depth

MESH_MIN, MESH_MAX = 0.10, 0.45
REGION_NAMES = {1: "cornea", 2: "aqueous", 3: "lens", 4: "vitreous", 5: "sclera"}


def _disk(cz, r):
    return (2, gmsh.model.occ.addDisk(0, cz, 0, r, r))


def build(write=True):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("eye_axisym")
    occ = gmsh.model.occ

    # Full eye outline and the inner (non-fibrous) volume
    outer, _ = occ.fuse([_disk(GLOBE_CZ, GLOBE_R)], [_disk(CORNEA_CZ, CORNEA_R)])
    inner, _ = occ.fuse([_disk(GLOBE_CZ, GLOBE_R - SHELL_T)],
                        [_disk(CORNEA_CZ + CORNEA_T, CORNEA_R - CORNEA_T)])

    # Fibrous shell = outer minus inner. Keep `inner` for further cutting.
    shell, _ = occ.cut(outer, occ.copy(inner), removeObject=True, removeTool=True)

    # Split the shell at the limbus plane into cornea (anterior) and sclera
    ant_half = (2, occ.addRectangle(-GLOBE_R - 2, -GLOBE_R - 2, 0,
                                    2 * GLOBE_R + 4, GLOBE_R + 2 + ACD))
    cornea, _ = occ.intersect(occ.copy(shell), occ.copy([ant_half]),
                              removeObject=True, removeTool=True)
    sclera, _ = occ.cut(shell, occ.copy([ant_half]),
                        removeObject=True, removeTool=True)

    # Lens, then the remaining humours
    lens, _ = occ.intersect([_disk(LENS_CZ, 1.0)], occ.copy(inner),
                            removeObject=True, removeTool=True)
    occ.remove(lens, recursive=True)                       # placeholder removed
    lens_e = (2, occ.addDisk(0, LENS_CZ, 0, LENS_RR, LENS_RZ))
    lens, _ = occ.intersect([lens_e], occ.copy(inner),
                            removeObject=True, removeTool=True)

    humours, _ = occ.cut(inner, occ.copy(lens), removeObject=True, removeTool=True)

    # Aqueous is anterior to the lens equator; vitreous is the rest
    eq_half = (2, occ.addRectangle(-GLOBE_R - 2, -GLOBE_R - 2, 0,
                                   2 * GLOBE_R + 4, GLOBE_R + 2 + LENS_CZ))
    aqueous, _ = occ.intersect(occ.copy(humours), occ.copy([eq_half]),
                               removeObject=True, removeTool=True)
    vitreous, _ = occ.cut(humours, occ.copy([eq_half]),
                          removeObject=True, removeTool=True)
    occ.remove([ant_half, eq_half], recursive=True)
    occ.synchronize()

    # Restrict to the axisymmetric half-plane r >= 0
    half = (2, occ.addRectangle(0, -GLOBE_R - 4, 0,
                                GLOBE_R + 4, 2 * GLOBE_R + GLOBE_CZ + 8))
    occ.synchronize()
    regions = {}
    for gid, shape in ((1, cornea), (2, aqueous), (3, lens),
                       (4, vitreous), (5, sclera)):
        cut, _ = occ.intersect(shape, occ.copy([half]),
                               removeObject=True, removeTool=True)
        regions[gid] = cut
    occ.remove([half], recursive=True)
    occ.synchronize()

    # Make interfaces conformal, tracking which region each piece came from
    flat = [dt for tags in regions.values() for dt in tags]
    order = [gid for gid, tags in regions.items() for _ in tags]
    frags, fmap = occ.fragment(flat, [])
    occ.synchronize()

    owner = {}
    for src_idx, outs in enumerate(fmap):
        gid = order[src_idx]
        for dim, tag in outs:
            if dim == 2:
                owner.setdefault(tag, gid)

    grouped = {}
    for tag, gid in owner.items():
        grouped.setdefault(gid, []).append(tag)
    for gid, tags in sorted(grouped.items()):
        gmsh.model.addPhysicalGroup(2, tags, gid)
        gmsh.model.setPhysicalName(2, gid, REGION_NAMES[gid])

    # Boundary curves: anterior (air-exposed) and posterior (blood-backed)
    ant, post = [], []
    for dim, tag in gmsh.model.getBoundary(
            [(2, t) for t in owner], oriented=False, combined=True):
        if dim != 1:
            continue
        r, z, _ = occ.getCenterOfMass(1, tag)
        if r < 1e-6:
            continue                                    # symmetry axis
        if abs(np.hypot(r, z - CORNEA_CZ) - CORNEA_R) < 0.3 and z < ACD + 1.0:
            ant.append(tag)
        elif abs(np.hypot(r, z - GLOBE_CZ) - GLOBE_R) < 0.3:
            post.append(tag)
    for tags, gid, nm in ((ant, 10, "anterior"), (post, 11, "posterior")):
        if tags:
            gmsh.model.addPhysicalGroup(1, tags, gid)
            gmsh.model.setPhysicalName(1, gid, nm)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_MIN)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_MAX)
    gmsh.model.mesh.generate(2)

    areas = {REGION_NAMES[g]: sum(occ.getMass(2, t) for t in tags)
             for g, tags in grouped.items()}
    nn = len(gmsh.model.mesh.getNodes()[0])
    path = OUT / "eye_axisym.msh"
    if write:
        gmsh.write(str(path))
    gmsh.finalize()
    return path, nn, areas, len(ant), len(post)


if __name__ == "__main__":
    path, nn, areas, na, npst = build()
    print(f"wrote {path}")
    print(f"nodes: {nn}")
    print(f"boundary curves: {na} anterior, {npst} posterior\n")
    print(f"{'region':10s} {'area (mm2)':>12s}")
    for nm in ("cornea", "aqueous", "lens", "vitreous", "sclera"):
        a = areas.get(nm)
        flag = "" if a else "   <-- EMPTY"
        print(f"{nm:10s} {(a or 0):12.2f}{flag}")
    total = sum(areas.values())
    print(f"{'TOTAL':10s} {total:12.2f}")
