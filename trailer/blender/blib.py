"""Shared procedural building blocks for the trailer's Blender shots (Blender 4.2, Cycles CPU).

Each shot script imports this, builds its scene, then calls render_range().
Frames are 1920x804 (2.39:1), rendered on twos where the camera moves and
interpolated to 24 fps afterwards with optical flow.
"""
import bpy, math, random, sys, os
from mathutils import Vector

W, H = 1920, 804


def reset(seed=1):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    random.seed(seed)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.fps = 24
    sc.render.engine = 'CYCLES'
    c = sc.cycles
    c.device = 'CPU'; c.samples = 24; c.use_denoising = True; c.denoiser = 'OPENIMAGEDENOISE'
    c.max_bounces = 4; c.diffuse_bounces = 2; c.glossy_bounces = 2; c.transmission_bounces = 2
    c.volume_bounces = 0; c.volume_step_rate = 4.0; c.use_adaptive_sampling = True; c.adaptive_threshold = 0.03
    c.caustics_reflective = False; c.caustics_refractive = False; c.blur_glossy = 1.0
    sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_depth = '16'
    vl = sc.view_layers[0]; vl.use_pass_mist = True
    return sc


def mat(name, color=(0.5, 0.5, 0.5), rough=0.6, metal=0.0, emit=None, emit_strength=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Roughness'].default_value = rough; p.inputs['Metallic'].default_value = metal
    if emit:
        p.inputs['Emission Color'].default_value = (*emit, 1); p.inputs['Emission Strength'].default_value = emit_strength
    return m


def window_mat(name, base=(0.02, 0.022, 0.025), lit=(1.0, 0.62, 0.32), density=0.18, strength=6.0, scale=(0.42, 0.48)):
    """Facade: dark concrete with a grid of randomly lit windows.

    World-space position is scaled to window cells (scale = cells per metre, so ~4 x 3.5 m windows
    regardless of tower size); u = x + y runs along whichever
    face we are on (one of x/y is constant per face), v = z. A cell is a window
    where fract(u) and fract(v) sit inside the frame, and it is lit when its
    per-cell white noise falls under `density`.
    """
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    p = N['Principled BSDF']; p.inputs['Base Color'].default_value = (*base, 1); p.inputs['Roughness'].default_value = 0.85

    def math(op, a, b=None, val=None):
        n = N.new('ShaderNodeMath'); n.operation = op
        L.new(a, n.inputs[0])
        if b is not None:
            L.new(b, n.inputs[1])
        elif val is not None:
            n.inputs[1].default_value = val
        return n.outputs[0]

    geo = N.new('ShaderNodeNewGeometry'); sep = N.new('ShaderNodeSeparateXYZ'); L.new(geo.outputs['Position'], sep.inputs[0])
    u = math('MULTIPLY', math('ADD', sep.outputs['X'], sep.outputs['Y']), val=scale[0])
    v = math('MULTIPLY', sep.outputs['Z'], val=scale[1])
    fu, fv = math('FRACT', u), math('FRACT', v)
    inside = math('MULTIPLY',
                  math('MULTIPLY', math('GREATER_THAN', fu, val=0.18), math('LESS_THAN', fu, val=0.82)),
                  math('MULTIPLY', math('GREATER_THAN', fv, val=0.22), math('LESS_THAN', fv, val=0.78)))
    cell = N.new('ShaderNodeCombineXYZ'); L.new(math('FLOOR', u), cell.inputs['X']); L.new(math('FLOOR', v), cell.inputs['Y'])
    oi = N.new('ShaderNodeObjectInfo')
    L.new(math('MULTIPLY', oi.outputs['Random'], val=1000.0), cell.inputs['Z'])   # constant per tower, so each window is one random draw
    wn = N.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '3D'; L.new(cell.outputs[0], wn.inputs['Vector'])
    on = math('LESS_THAN', wn.outputs['Value'], val=density)
    win = math('MULTIPLY', inside, on)
    ramp = N.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color = (*lit, 1)
    ramp.color_ramp.elements[1].color = (0.30, 0.68, 1.0, 1)
    L.new(wn.outputs['Color'], ramp.inputs['Fac'])
    L.new(ramp.outputs['Color'], p.inputs['Emission Color'])
    # each lit window gets its own brightness (0.25-1.0): offices, apartments, dead tubes
    bright = math('ADD', math('MULTIPLY', wn.outputs['Color'], val=0.75), val=0.25)
    L.new(math('MULTIPLY', math('MULTIPLY', win, bright), val=strength), p.inputs['Emission Strength'])
    return m


def box(loc, scale, material, name='box', bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(location=loc); o = bpy.context.object; o.name = name
    o.scale = scale; o.data.materials.append(material)
    if bevel:
        b = o.modifiers.new('bev', 'BEVEL'); b.width = bevel; b.segments = 2
    return o


def world(color=(0.01, 0.012, 0.018), strength=1.0, fog_density=0.0, fog_color=(0.6, 0.7, 0.8), anisotropy=0.3):
    w = bpy.data.worlds.new('W'); bpy.context.scene.world = w; w.use_nodes = True
    nt = w.node_tree
    nt.nodes['Background'].inputs['Color'].default_value = (*color, 1)
    nt.nodes['Background'].inputs['Strength'].default_value = strength
    if fog_density > 0:
        v = nt.nodes.new('ShaderNodeVolumePrincipled')
        v.inputs['Density'].default_value = fog_density; v.inputs['Color'].default_value = (*fog_color, 1)
        v.inputs['Anisotropy'].default_value = anisotropy
    if emit:                                   # (r, g, b, strength): the city's own glow held in the haze
        v.inputs['Emission Color'].default_value = (*emit[:3], 1); v.inputs['Emission Strength'].default_value = emit[3]
        nt.links.new(v.outputs[0], nt.nodes['World Output'].inputs['Volume'])
    return w


def sky_gradient(horizon=(1.0, 0.42, 0.12), zenith=(0.004, 0.005, 0.01), strength=1.0, clouds=0.6, band=0.18):
    """World: warm glow at the horizon fading to a dark zenith, broken by large noise clouds."""
    w = bpy.data.worlds.new('Sky'); bpy.context.scene.world = w; w.use_nodes = True
    nt = w.node_tree; N = nt.nodes; L = nt.links
    tc = N.new('ShaderNodeTexCoord'); sep = N.new('ShaderNodeSeparateXYZ'); L.new(tc.outputs['Generated'], sep.inputs[0])
    # Z runs -1 (straight down) .. 1 (straight up); remap so below the horizon stays dark (sea must not reflect glow)
    zr = N.new('ShaderNodeMapRange'); zr.inputs['From Min'].default_value = -0.1; zr.inputs['From Max'].default_value = 1.0
    L.new(sep.outputs['Z'], zr.inputs['Value'])
    rp = N.new('ShaderNodeValToRGB'); h0 = 0.1 / 1.1
    rp.color_ramp.elements[0].position = 0.0; rp.color_ramp.elements[0].color = (*[c * 0.15 for c in zenith], 1)
    rp.color_ramp.elements[1].position = h0 + band; rp.color_ramp.elements[1].color = (*zenith, 1)
    e = rp.color_ramp.elements.new(h0); e.color = (*horizon, 1)
    L.new(zr.outputs['Result'], rp.inputs['Fac'])
    nz = N.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 3.0; nz.inputs['Detail'].default_value = 6
    L.new(tc.outputs['Generated'], nz.inputs['Vector'])
    cm = N.new('ShaderNodeMapRange'); cm.inputs['From Min'].default_value = 0.4; cm.inputs['From Max'].default_value = 0.7
    cm.inputs['To Min'].default_value = 1.0 - clouds; cm.inputs['To Max'].default_value = 1.0
    L.new(nz.outputs['Fac'], cm.inputs['Value'])
    mul = N.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
    L.new(rp.outputs['Color'], mul.inputs['A']); L.new(cm.outputs['Result'], mul.inputs['B'])
    L.new(mul.outputs['Result'], N['Background'].inputs['Color']); N['Background'].inputs['Strength'].default_value = strength
    return w


def fog_box(center, size, density=0.002, color=(0.6, 0.7, 0.8), anisotropy=0.4, name='fog', noise=0.0, emit=None):
    """Bounded haze. An infinite world volume swallows the sky; a big box keeps the sky visible behind it.
    noise > 0 breaks the haze into drifting banks (procedural density)."""
    bpy.ops.mesh.primitive_cube_add(location=center); o = bpy.context.object; o.name = name
    o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.remove(nt.nodes['Principled BSDF'])
    v = nt.nodes.new('ShaderNodeVolumePrincipled'); v.inputs['Color'].default_value = (*color, 1)
    v.inputs['Anisotropy'].default_value = anisotropy
    if emit:                                   # (r, g, b, strength): the city's own glow held in the haze
        v.inputs['Emission Color'].default_value = (*emit[:3], 1); v.inputs['Emission Strength'].default_value = emit[3]
    if noise > 0:
        tx = nt.nodes.new('ShaderNodeTexNoise'); tx.inputs['Scale'].default_value = noise; tx.inputs['Detail'].default_value = 3
        mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.35; mr.inputs['From Max'].default_value = 0.75
        mr.inputs['To Min'].default_value = density * 0.2; mr.inputs['To Max'].default_value = density * 1.8
        nt.links.new(tx.outputs['Fac'], mr.inputs['Value']); nt.links.new(mr.outputs['Result'], v.inputs['Density'])
    else:
        v.inputs['Density'].default_value = density
    nt.links.new(v.outputs[0], nt.nodes['Material Output'].inputs['Volume'])
    o.data.materials.append(m)
    o.visible_shadow = False
    return o


def camera(loc, rot_deg, lens=35, name='Cam'):
    bpy.ops.object.camera_add(location=loc, rotation=[math.radians(a) for a in rot_deg])
    c = bpy.context.object; c.name = name; c.data.lens = lens; c.data.sensor_width = 36
    c.data.clip_end = 20000
    bpy.context.scene.camera = c
    return c


def key(obj, frame, **props):
    for k, v in props.items():
        setattr(obj, k, v); obj.keyframe_insert(data_path=k, frame=frame)


def smooth_keys(obj):
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'BEZIER'; kp.easing = 'EASE_IN_OUT'


def spinner(loc, name='spinner', scale=1.0, light_color=(1.0, 0.85, 0.7), red=(1.0, 0.08, 0.04)):
    """A compact flying police car: wedge body, glass canopy, emissive headlights, light bar and red tail lights."""
    body_m = mat(name + '_body', (0.05, 0.055, 0.06), rough=0.25, metal=0.8)
    glass_m = mat(name + '_glass', (0.02, 0.03, 0.04), rough=0.05, metal=0.0)
    head_m = mat(name + '_head', (1, 1, 1), emit=light_color, emit_strength=300)
    tail_m = mat(name + '_tail', (1, 0, 0), emit=red, emit_strength=120)
    bar_m = mat(name + '_bar', (0.2, 0.4, 1), emit=(0.25, 0.45, 1.0), emit_strength=80)
    bpy.ops.object.empty_add(location=loc); root = bpy.context.object; root.name = name
    parts = []
    b = box((0, 0, 0), (1.1, 2.6, 0.45), body_m, name + '_b', bevel=0.2); parts.append(b)
    c = box((0, -0.3, 0.55), (0.85, 1.2, 0.3), glass_m, name + '_c', bevel=0.25); parts.append(c)
    for sx in (-0.75, 0.75):
        parts.append(box((sx, 2.62, 0.0), (0.22, 0.04, 0.08), head_m, name + '_h'))
        parts.append(box((sx, -2.62, 0.05), (0.3, 0.04, 0.06), tail_m, name + '_t'))
    parts.append(box((0, -0.3, 0.9), (0.6, 0.12, 0.05), bar_m, name + '_bar'))
    under_m = mat(name + '_under', (0, 0, 0), emit=(0.3, 0.55, 1.0), emit_strength=60)
    parts.append(box((0, 0, -0.47), (0.9, 2.2, 0.02), under_m, name + '_under'))   # underglow: never a black void against light
    for sx in (-1.12, 1.12):
        parts.append(box((sx, 0.0, 0.0), (0.03, 2.2, 0.04), bar_m, name + '_side'))
    for p in parts:
        p.parent = root
    root.scale = (scale, scale, scale)
    # headlight beam
    bpy.ops.object.light_add(type='SPOT', location=(0, 2.7, 0)); sp = bpy.context.object
    sp.data.energy = 60000 * scale; sp.data.spot_size = math.radians(25); sp.data.color = light_color
    sp.rotation_euler = (math.radians(-80), 0, 0); sp.parent = root
    return root


def use_gpu():
    """BR_GPU=1: render on the GPU (OptiX, else CUDA/HIP/Metal) at final quality.
    BR_SAMPLES (default 6) multiplies each scene's sample count; fog is stepped finer.
    Returns True when a GPU was enabled."""
    if os.environ.get('BR_GPU') != '1':
        return False
    sc = bpy.context.scene
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for kind in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        devs = [d for d in prefs.devices if d.type == kind]
        if devs:
            for d in prefs.devices:
                d.use = d.type == kind
            sc.cycles.device = 'GPU'
            print('GPU render on', kind, [d.name for d in devs])
            break
    else:
        print('BR_GPU=1 but no GPU found; rendering on CPU')
        return False
    sc.cycles.samples = int(sc.cycles.samples * float(os.environ.get('BR_SAMPLES', '6')))
    sc.cycles.adaptive_threshold = 0.01
    sc.cycles.volume_step_rate = min(sc.cycles.volume_step_rate, 1.0)
    sc.cycles.volume_bounces = 1
    return True


def render_range(out_dir, f0, f1, step=1):
    sc = bpy.context.scene
    if use_gpu():
        step = 1                                   # every frame on the GPU: no flow in-betweens needed
    os.makedirs(bpy.path.abspath(out_dir), exist_ok=True)
    for f in range(f0, f1 + 1, step):
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out_dir, f'f{f:04d}.png')
        bpy.ops.render.render(write_still=True)


def args():
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return a
