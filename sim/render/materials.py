"""Shared Blender material set for every NRDI render (hero, explode, physics).
Extracted from blender_hero.py so the product and physics shots match.
"""
import bpy

def _mat(name):
    m = bpy.data.materials.new(name)
    if not m.node_tree:
        m.use_nodes = True
    return m, m.node_tree.nodes["Principled BSDF"], m.node_tree

def mat_acetate(name, rgb=(0.006, 0.006, 0.007)):
    """Black acetate: near-black base, coat roughness high enough that the
    studio HDRI does not wash the frame out to light grey."""
    m, b, nt = _mat(name)
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = 0.22
    b.inputs["Coat Weight"].default_value = 0.4
    b.inputs["Coat Roughness"].default_value = 0.12
    b.inputs["Subsurface Weight"].default_value = 0.05
    b.inputs["Subsurface Radius"].default_value = (0.002, 0.002, 0.002)
    return m

def mat_glass(name):
    m, b, nt = _mat(name)
    b.inputs["Base Color"].default_value = (0.97, 0.99, 1.0, 1)
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.0
    b.inputs["IOR"].default_value = 1.498          # CR-39
    b.inputs["Coat Weight"].default_value = 0.4    # AR-coat sheen
    b.inputs["Coat Tint"].default_value = (0.80, 0.88, 1.0, 1)
    m.surface_render_method = "DITHERED"
    try:
        m.use_raytrace_refraction = True
    except AttributeError:
        pass
    return m

def mat_metal(name, rgb=(0.75, 0.76, 0.78), rough=0.28):
    m, b, nt = _mat(name)
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = rough
    return m

def mat_pcb(name):
    """Soldermask green with faint procedural trace pattern."""
    m, b, nt = _mat(name)
    tex = nt.nodes.new("ShaderNodeTexVoronoi"); tex.inputs["Scale"].default_value = 900.0
    tex.feature = "DISTANCE_TO_EDGE"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0; ramp.color_ramp.elements[0].color = (0.02, 0.16, 0.06, 1)
    ramp.color_ramp.elements[1].position = 0.08; ramp.color_ramp.elements[1].color = (0.01, 0.10, 0.04, 1)
    nt.links.new(tex.outputs["Distance"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.35
    b.inputs["Coat Weight"].default_value = 0.5
    return m

def mat_ic(name):
    m, b, nt = _mat(name)
    b.inputs["Base Color"].default_value = (0.012, 0.012, 0.012, 1)
    b.inputs["Roughness"].default_value = 0.45
    b.inputs["Specular IOR Level"].default_value = 0.4
    return m

def mat_window(name, rgb=(0.02, 0.02, 0.03)):
    m, b, nt = _mat(name)
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = 0.05
    b.inputs["Coat Weight"].default_value = 1.0
    return m

def mat_led(name):
    m, b, nt = _mat(name)
    b.inputs["Base Color"].default_value = (0.35, 0.05, 0.05, 1)
    b.inputs["Transmission Weight"].default_value = 0.6
    b.inputs["Roughness"].default_value = 0.1
    b.inputs["Emission Color"].default_value = (0.6, 0.05, 0.02, 1)
    b.inputs["Emission Strength"].default_value = 0.4
    return m

def mat_gel(name):
    """Dry Ag/AgCl-coated electrode disc: reads as a contact, not a pad --
    a touch of metallic sheen over a dark base."""
    m, b, nt = _mat(name)
    b.inputs["Base Color"].default_value = (0.02, 0.02, 0.022, 1)
    b.inputs["Roughness"].default_value = 0.35
    b.inputs["Metallic"].default_value = 0.3
    b.inputs["Coat Weight"].default_value = 0.2
    return m

def mat_silicone(name, rgb=(0.03, 0.03, 0.035)):
    """Soft matte medical silicone (brow bumper, nose pads)."""
    m, b, nt = _mat(name)
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = 0.7
    b.inputs["Subsurface Weight"].default_value = 0.2
    b.inputs["Subsurface Radius"].default_value = (0.001, 0.001, 0.001)
    b.inputs["Specular IOR Level"].default_value = 0.35
    return m

def mat_flex(name):
    """Polyimide flex / FPC: amber, slightly translucent, glossy coverlay."""
    m, b, nt = _mat(name)
    b.inputs["Base Color"].default_value = (0.45, 0.18, 0.03, 1)
    b.inputs["Roughness"].default_value = 0.3
    b.inputs["Transmission Weight"].default_value = 0.15
    b.inputs["Coat Weight"].default_value = 0.6
    return m

def build_mats():
    return {
    "frame": mat_acetate("acetate"), "lens": mat_glass("cr39"),
    "pcb": mat_pcb("soldermask"), "chip": mat_ic("ic"), "window": mat_window("window"),
    "led": mat_led("ir_led"), "metal": mat_metal("stainless"), "gel": mat_gel("gel"),
    "silicone": mat_silicone("silicone"), "flex": mat_flex("polyimide"),
    "gold": mat_metal("gold", rgb=(0.95, 0.72, 0.30), rough=0.25),
    }

MATS = None
def material_for(p):
    global MATS
    if MATS is None:
        MATS = build_mats()
    lab, cat = p["label"], p["category"]
    if lab.startswith("hinge") or lab.startswith("battery") or lab.startswith("haptic"):
        return MATS["metal"]
    if lab.startswith(("browpad", "nosepad")):
        return MATS["silicone"]
    if lab.startswith("elec"):
        return MATS["gel"]
    if lab.startswith("thermo_pins"):
        return MATS["gold"]
    if lab.startswith(("ppg_flex", "cam_fpc")):
        return MATS["flex"]
    if lab.startswith(("irled_pkg", "ppg")):
        return MATS["chip"]
    if lab.startswith("irled"):
        return MATS["led"]
    if lab.startswith(("cam", "thermo")) and not lab.startswith(("camboard", "cam_fpc")):
        return MATS["window"]
    if cat == "frame":
        return MATS["frame"]
    if cat == "lens":
        return MATS["lens"]
    if lab.startswith("pcb"):
        return MATS["pcb"]
    return MATS["chip"]

