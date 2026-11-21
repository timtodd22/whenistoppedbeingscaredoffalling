"""The city: one set, three cameras.

  -- cam descent    shot 5   drop from the rooftops into a neon canyon, a spinner overtakes the lens
  -- cam hologram   shot 11  street level, a towering hologram over the rain-filled street
  -- cam aerial     shot 8   high over endless towers and fog layers toward a ziggurat megablock

Neon signage and the hologram are AI-generated textures (tex/), mapped onto
emissive planes with alpha. Rain is added in compositing.
"""
import bpy, math, random, os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
import blib

a = blib.args()
which = a[a.index('cam') + 1] if 'cam' in a else 'descent'
TEX = os.path.join(os.path.dirname(__file__), 'tex')
sc = blib.reset(seed=21)
LEN = {'descent': 96, 'hologram': 48, 'aerial': 72}[which]
sc.frame_start, sc.frame_end = 1, LEN
sc.cycles.samples = 14
sc.cycles.volume_step_rate = 8.0   # coarser fog steps: the haze is soft, the saving is large


def image_mat(name, path, strength=6.0):
    """Emissive plane from an RGBA image; alpha cuts the shape out."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    N.remove(N['Principled BSDF'])
    img = N.new('ShaderNodeTexImage'); img.image = bpy.data.images.load(path); img.extension = 'CLIP'
    em = N.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = strength
    tr = N.new('ShaderNodeBsdfTransparent'); mx = N.new('ShaderNodeMixShader')
    L.new(img.outputs['Color'], em.inputs['Color'])
    L.new(img.outputs['Alpha'], mx.inputs['Fac']); L.new(tr.outputs[0], mx.inputs[1]); L.new(em.outputs[0], mx.inputs[2])
    L.new(mx.outputs[0], N['Material Output'].inputs['Surface'])
    m.blend_method = 'BLEND'
    return m, em


def plane(loc, size, rot, material, name):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=[math.radians(r) for r in rot])
    o = bpy.context.object; o.name = name; o.scale = (size[0], size[1], 1); o.data.materials.append(material)
    return o


# ---- street: wet asphalt with puddles (roughness driven by noise) ----
asph = blib.mat('asphalt', (0.012, 0.012, 0.014), rough=0.25, metal=0.0)
nN = asph.node_tree.nodes; nL = asph.node_tree.links
nz = nN.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 0.08; nz.inputs['Detail'].default_value = 4
rr = nN.new('ShaderNodeMapRange'); rr.inputs['From Min'].default_value = 0.45; rr.inputs['From Max'].default_value = 0.55
rr.inputs['To Min'].default_value = 0.02; rr.inputs['To Max'].default_value = 0.45
nL.new(nz.outputs['Fac'], rr.inputs['Value']); nL.new(rr.outputs['Result'], nN['Principled BSDF'].inputs['Roughness'])
plane((0, 600, 0), (60, 3000), (0, 0, 0), asph, 'street')

# ---- towers: two canyon walls along the street, then a deep city grid ----
facades = [blib.window_mat(f'fac{i}', lit=c, density=d, strength=s)
           for i, (c, d, s) in enumerate([((1.0, 0.55, 0.22), 0.17, 1.0), ((0.45, 0.8, 1.0), 0.13, 1.2), ((1.0, 0.68, 0.38), 0.21, 0.85)])]
random.seed(4)
for side in (-1, 1):
    y = -100
    while y < 1600:
        w = random.uniform(18, 40); h = random.uniform(90, 420); d = random.uniform(20, 45)
        x = side * (16 + d / 2 + random.uniform(0, 6))
        blib.box((x, y + w / 2, h / 2), (d / 2, w / 2, h / 2), random.choice(facades), 'tower')
        # setback crown on some towers
        if random.random() < 0.4:
            blib.box((x, y + w / 2, h + 15), (d / 3, w / 3, 15), random.choice(facades), 'crown')
        y += w + random.uniform(1, 6)
for gx in range(-1500, 1501, 70):
    for gy in range(-200, 3200, 70):
        if abs(gx) < 80:
            continue
        if random.random() < 0.15:
            continue
        h = random.uniform(60, 300) * (1.3 if random.random() < 0.1 else 1)
        blib.box((gx + random.uniform(-10, 10), gy + random.uniform(-10, 10), h / 2), (22, 22, h / 2), random.choice(facades), 'grid')

# ---- the megablock on the horizon: three colossal leaning slabs, a fractured monolith ----
zig = blib.window_mat('zig', lit=(1.0, 0.7, 0.4), density=0.10, strength=1.6, scale=(0.3, 0.35))
for (dx, w, h, lean) in ((-260, 160, 980, -4), (40, 220, 1300, 2), (330, 140, 860, 6)):
    o = blib.box((150 + dx, 4600, h / 2), (w / 2, 90, h / 2), zig, 'mega')
    o.rotation_euler = (0, math.radians(lean), 0)
beam = blib.mat('beam', (1, 0, 0), emit=(1.0, 0.08, 0.04), emit_strength=400)
for (dx, h) in ((-260, 990), (40, 1310), (330, 870)):
    blib.box((150 + dx, 4510, h), (4, 4, 4), beam, 'beacon')

# ---- neon signage on the canyon walls, low floors ----
signs = sorted(glob.glob(os.path.join(TEX, 'neon_*.png')))
sign_mats = [image_mat(f'sign{i}', p, strength=random.uniform(5, 10))[0] for i, p in enumerate(signs)]
random.seed(9)
for i in range(70):
    side = random.choice((-1, 1)); y = random.uniform(-40, 700); z = random.uniform(6, 55)
    m = random.choice(sign_mats); img = m.node_tree.nodes['Image Texture'].image
    ar = img.size[0] / img.size[1]; hgt = random.uniform(4, 11); wid = hgt * ar
    # signs hang just off the facade, facing the street
    x = side * 15.5
    plane((x, y, z), (wid, hgt), (90, 0, -90 * side), m, 'sign')
# a few coloured area lights so the neon tints the haze and the wet street
for i in range(14):
    side = random.choice((-1, 1)); col = random.choice([(1, 0.2, 0.6), (0.2, 0.8, 1), (1, 0.6, 0.2), (0.6, 0.3, 1)])
    bpy.ops.object.light_add(type='AREA', location=(side * 14, random.uniform(0, 600), random.uniform(8, 40)))
    L_ = bpy.context.object; L_.data.energy = 25000; L_.data.size = 8; L_.data.color = col
    L_.rotation_euler = (0, math.radians(90 * side), 0)

# ---- the hologram: towering projection at the far end of the street ----
holo_m, holo_em = image_mat('holo', os.path.join(TEX, 'hologram.png'), strength=4)
holo = plane((0, 420, 55), (62, 110), (90, 0, 0), holo_m, 'hologram')
for f in range(1, LEN + 1, 3):            # subtle flicker
    holo_em.inputs['Strength'].default_value = 4 * (0.75 + 0.25 * random.random()) * (0.3 if random.random() < 0.05 else 1)
    holo_em.inputs['Strength'].keyframe_insert('default_value', frame=f)
bpy.ops.object.light_add(type='AREA', location=(0, 405, 55)); hl = bpy.context.object
hl.data.energy = 400000; hl.data.size = 40; hl.data.color = (0.9, 0.4, 1.0); hl.rotation_euler = (math.radians(90), 0, 0)

# ---- spinners ----
sp1 = blib.spinner((0, -60, 120), 'sp1', scale=1.6)
sp2 = blib.spinner((-6, 300, 38), 'sp2', scale=1.6)
sp3 = blib.spinner((40, 900, 260), 'sp3', scale=2.2)

# ---- sky and haze ----
blib.sky_gradient((0.35, 0.16, 0.12), (0.003, 0.004, 0.008), 1.0, clouds=0.6, band=0.25)
blib.fog_box((0, 1200, 220), (3400, 3600, 480), 0.0022, (0.55, 0.6, 0.7), anisotropy=0.5, noise=0.01, emit=(0.32, 0.40, 0.55, 0.0007))
bpy.ops.object.light_add(type='SUN', rotation=(math.radians(20), 0, math.radians(30)))
bpy.context.object.data.energy = 0.03

# ---- cameras ----
if which == 'descent':
    cam = blib.camera((0, -120, 260), (62, 0, 0), lens=24)
    blib.key(cam, 1, location=(0, -120, 260), rotation_euler=(math.radians(62), 0, 0))
    blib.key(cam, LEN, location=(0, 40, 40), rotation_euler=(math.radians(84), 0, 0))
    blib.smooth_keys(cam)
    # spinner comes from behind/above the lens and pulls ahead down the canyon
    blib.key(sp1, 1, location=(4, -170, 300)); blib.key(sp1, LEN, location=(-3, 260, 45))
    sp1.rotation_euler = (math.radians(-10), 0, 0)
    blib.key(sp3, 1, location=(40, 900, 260)); blib.key(sp3, LEN, location=(30, 700, 250))
    sp3.rotation_euler = (0, 0, math.radians(180))
elif which == 'hologram':
    cam = blib.camera((3, 250, 1.6), (107, 0, 2), lens=18)   # low, wide, looking up: the whole figure towers over the street
    blib.key(cam, 1, location=(3, 250, 1.6)); blib.key(cam, LEN, location=(2.2, 254, 1.6))
    blib.key(sp2, 1, location=(-30, 330, 38)); blib.key(sp2, LEN, location=(30, 345, 42))
    sp2.rotation_euler = (0, 0, math.radians(-90))
else:  # aerial
    cam = blib.camera((-120, -400, 620), (80, 0, -4), lens=35)
    blib.key(cam, 1, location=(-120, -400, 620)); blib.key(cam, LEN, location=(-110, -250, 610))
    blib.key(sp3, 1, location=(-60, 200, 560)); blib.key(sp3, LEN, location=(20, 600, 520))
for o in (sp1, sp2, sp3):
    if o.animation_data:
        for fc in o.animation_data.action.fcurves:
            for k in fc.keyframe_points: k.interpolation = 'LINEAR'

# area lights are visible to the camera in Cycles; the fill lights must not show as panels
for ob in bpy.data.objects:
    if ob.type == 'LIGHT' and ob.data.type in ('AREA', 'POINT', 'SPOT'):
        ob.visible_camera = False

if __name__ == '__main__':
    out = a[a.index('out') + 1] if 'out' in a else f'//city_{which}/'
    f0 = int(a[a.index('f0') + 1]) if 'f0' in a else 1
    f1 = int(a[a.index('f1') + 1]) if 'f1' in a else LEN
    step = int(a[a.index('step') + 1]) if 'step' in a else 2
    if 'pct' in a:
        bpy.context.scene.render.resolution_percentage = int(a[a.index('pct') + 1])
    blib.render_range(out, f0, f1, step)
