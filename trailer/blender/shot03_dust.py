"""Shot 3: THE ORANGE DUST.

A near-monochrome ochre haze. Colossal brutalist slabs and a broken ring
structure dissolve into the dust; a tiny figure in a long coat walks away
from camera toward them. The camera tracks slowly sideways, so the slabs
separate in depth (parallax is what sells the scale in haze).
"""
import bpy, math, random, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import blib

a = blib.args()
sc = blib.reset(seed=11)
sc.frame_start, sc.frame_end = 1, 120
sc.cycles.samples = 16

dust = blib.mat('dust', (0.3, 0.17, 0.08), rough=0.95)
conc = blib.mat('conc', (0.07, 0.05, 0.035), rough=0.9)   # dark, so the haze turns it into graded silhouettes
# dunes: subdivided plane displaced by large noise
bpy.ops.mesh.primitive_plane_add(size=5000, location=(0, 2000, 0))
g = bpy.context.object
sub = g.modifiers.new('sub', 'SUBSURF'); sub.subdivision_type = 'SIMPLE'; sub.levels = sub.render_levels = 6
tx = bpy.data.textures.new('dunes', 'CLOUDS'); tx.noise_scale = 260
dp = g.modifiers.new('disp', 'DISPLACE'); dp.texture = tx; dp.strength = 10; dp.mid_level = 0.6   # lower dunes: keep the horizon low
g.data.materials.append(dust)
# break up the ground: large noise drives albedo between two dust tones
_dn = dust.node_tree.nodes; _dl = dust.node_tree.links
_nz = _dn.new('ShaderNodeTexNoise'); _nz.inputs['Scale'].default_value = 0.02; _nz.inputs['Detail'].default_value = 8
_cr = _dn.new('ShaderNodeValToRGB'); _cr.color_ramp.elements[0].color = (0.18, 0.1, 0.05, 1); _cr.color_ramp.elements[1].color = (0.36, 0.21, 0.1, 1)
_dl.new(_nz.outputs['Fac'], _cr.inputs['Fac']); _dl.new(_cr.outputs['Color'], _dn['Principled BSDF'].inputs['Base Color'])
# (no foreground rubble: it read as clutter and hid the walker; the frame stays minimal)

# monoliths: tall slabs at increasing depth, one tilted, one broken
random.seed(5)
# sized for a 50 mm lens: tops sit below frame edge so sky and haze layers stay visible
slabs = [(-230, 800, 30, 10, 95, 0), (160, 1000, 40, 12, 130, 0), (-60, 1300, 55, 14, 180, 5),
         (330, 1600, 45, 14, 170, 0), (-420, 1900, 70, 18, 260, -4), (90, 2300, 90, 20, 300, 0)]
for x, y, w, d, h, tilt in slabs:
    o = blib.box((x, y, h / 2 - 10), (w / 2, d / 2, h / 2), conc, 'slab')   # no bevel: it is applied before the huge non-uniform scale
    o.rotation_euler = (0, math.radians(tilt), math.radians(random.uniform(-12, 12)))
# a colossal broken ring, half buried, far left
bpy.ops.mesh.primitive_torus_add(major_radius=120, minor_radius=10, location=(-560, 1500, 30),
                                 rotation=(math.radians(90), math.radians(18), math.radians(25)))
ring = bpy.context.object; ring.data.materials.append(conc)
bv = ring.modifiers.new('cut', 'BOOLEAN'); cutter = blib.box((-560, 1500, -200), (300, 300, 170), conc, 'cutter')
cutter.hide_render = True; cutter.hide_viewport = True; bv.object = cutter

# the walker: a simple long-coated silhouette (reads only as a shape at this distance)
coat = blib.mat('coat', (0.05, 0.04, 0.035), rough=0.8)
bpy.ops.object.empty_add(location=(10, 185, 0)); walker = bpy.context.object
parts = [blib.box((0, 0, 0.95), (0.2, 0.15, 0.55), coat, 'coatbody'),
         blib.box((0, 0, 1.62), (0.14, 0.14, 0.14), coat, 'head')]
bpy.ops.mesh.primitive_cone_add(radius1=0.32, radius2=0.2, depth=1.0, location=(0, 0, 0.5))
skirt = bpy.context.object; skirt.data.materials.append(coat); parts.append(skirt)
for p in parts:
    p.parent = walker
blib.key(walker, 1, location=(10, 185, 0)); blib.key(walker, sc.frame_end, location=(11.2, 196, 0))
for fc in walker.animation_data.action.fcurves:
    for k in fc.keyframe_points: k.interpolation = 'LINEAR'

# light: a low, diffuse sun through dense haze
blib.sky_gradient((1.0, 0.55, 0.25), (0.75, 0.42, 0.2), 1.6, clouds=0.15, band=0.6)
bpy.ops.object.light_add(type='SUN', rotation=(math.radians(70), 0, math.radians(160)))
sun = bpy.context.object; sun.data.energy = 1.6; sun.data.color = (1.0, 0.62, 0.32); sun.data.angle = math.radians(8)
blib.fog_box((0, 1300, 200), (4000, 3000, 600), 0.0016, (1.0, 0.58, 0.3), anisotropy=0.6, noise=0.006)
# the sun as a pale disc burning through the dust, upper left
sun_disc = blib.mat('sundisc', (1, 1, 1), emit=(1.0, 0.7, 0.42), emit_strength=7)
bpy.ops.mesh.primitive_uv_sphere_add(radius=85, location=(-640, 2700, 330)); bpy.context.object.data.materials.append(sun_disc)

# camera: eye height behind the walker, 35 mm, slow lateral track
cam = blib.camera((-6, 120, 2.2), (90.6, 0, -1), lens=50)
blib.key(cam, 1, location=(-6, 120, 2.2)); blib.key(cam, sc.frame_end, location=(2, 121, 2.3))
blib.smooth_keys(cam)

if __name__ == '__main__':
    out = a[a.index('out') + 1] if 'out' in a else '//shot03/'
    f0 = int(a[a.index('f0') + 1]) if 'f0' in a else 1
    f1 = int(a[a.index('f1') + 1]) if 'f1' in a else sc.frame_end
    step = int(a[a.index('step') + 1]) if 'step' in a else 2
    blib.render_range(out, f0, f1, step)
