"""Shot 1 / 17: THE SEA WALL. Night (shot 1) or grey dawn (shot 17, '-- dawn').

Camera sits low over the Pacific looking at a colossal black wall that runs off
both edges of frame. Waves (Ocean modifier) heave against its foot; tiny
maintenance lights dot the wall; the city's sodium glow bleeds over the top;
one spinner crosses far away. Rain is added in compositing.
"""
import bpy, math, random, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import blib

a = blib.args()
dawn = 'dawn' in a
sc = blib.reset(seed=7)
sc.frame_start, sc.frame_end = 1, 144 if not dawn else 120
sc.cycles.samples = 20

# ocean
bpy.ops.mesh.primitive_plane_add(size=10)
sea = bpy.context.object; sea.name = 'sea'
oc = sea.modifiers.new('ocean', 'OCEAN')
oc.geometry_mode = 'GENERATE'; oc.repeat_x = oc.repeat_y = 14; oc.size = 6; oc.spatial_size = 100
oc.resolution = 12; oc.wave_scale = 2.2; oc.choppiness = 1.4; oc.wind_velocity = 26
oc.use_foam = True; oc.foam_coverage = 0.25; oc.foam_layer_name = 'foam'
sea.location = (0, -400, 0)
oc.keyframe_insert('time', frame=1); oc.time = 1 + (sc.frame_end / 24.0) * 1.0; oc.keyframe_insert('time', frame=sc.frame_end)
for fc in sea.animation_data.action.fcurves:
    for k in fc.keyframe_points: k.interpolation = 'LINEAR'
sm = bpy.data.materials.new('sea'); sm.use_nodes = True
nt = sm.node_tree; p = nt.nodes['Principled BSDF']
p.inputs['Base Color'].default_value = (0.004, 0.008, 0.012, 1); p.inputs['Roughness'].default_value = 0.12
attr = nt.nodes.new('ShaderNodeAttribute'); attr.attribute_name = 'foam'
mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
mix.inputs['A'].default_value = (0.004, 0.008, 0.012, 1); mix.inputs['B'].default_value = (0.5, 0.55, 0.6, 1)
nt.links.new(attr.outputs['Fac'], mix.inputs['Factor']); nt.links.new(mix.outputs['Result'], p.inputs['Base Color'])
sea.data.materials.append(sm)

# the wall: a slab 3 km long, 300 m tall, with buttress ribs and a stepped crown
wall_m = blib.mat('wall', (0.03, 0.032, 0.036), rough=0.55)
# panel texture: large noise-driven bump so the surface catches the city glow
_n = wall_m.node_tree.nodes; _l = wall_m.node_tree.links
_tex = _n.new('ShaderNodeTexBrick'); _tex.inputs['Scale'].default_value = 0.02; _tex.inputs['Mortar Size'].default_value = 0.01
_tc = _n.new('ShaderNodeTexCoord'); _l.new(_tc.outputs['Object'], _tex.inputs['Vector'])
_bump = _n.new('ShaderNodeBump'); _bump.inputs['Strength'].default_value = 0.6
_l.new(_tex.outputs['Fac'], _bump.inputs['Height']); _l.new(_bump.outputs['Normal'], _n['Principled BSDF'].inputs['Normal'])
WALL_Y = 260
blib.box((0, WALL_Y + 20, 150), (2000, 20, 150), wall_m, 'wall')
rib = blib.box((0, WALL_Y - 4, 150), (6, 8, 150), wall_m, 'rib')
arr = rib.modifiers.new('arr', 'ARRAY'); arr.count = 70; arr.relative_offset_displace = (0, 0, 0)
arr.use_relative_offset = False; arr.use_constant_offset = True; arr.constant_offset_displace = (58, 0, 0)
rib.location.x = -2000
blib.box((0, WALL_Y + 10, 302), (2000, 28, 4), wall_m, 'crown')
# buttress towers every 400 m, protruding toward the sea, with vertical light strips
strip_m = blib.mat('strip', (1, 1, 1), emit=(1.0, 0.7, 0.45), emit_strength=25)
for bx in range(-1600, 1601, 400):
    blib.box((bx, WALL_Y - 30, 170), (34, 40, 170), wall_m, 'buttress', bevel=2)
    blib.box((bx, WALL_Y - 70.5, 170), (1.2, 0.3, 150), strip_m, 'strip')
# searchlights on the crown, sweeping through the haze
for i, bx in enumerate((-500, 650)):
    bpy.ops.object.light_add(type='SPOT', location=(bx, WALL_Y - 10, 330))
    sl = bpy.context.object; sl.data.energy = 1.5e9;   # km-scale haze needs absurd power to show a beam
    sl.data.spot_size = math.radians(4); sl.data.spot_blend = 0.3
    sl.data.color = (0.85, 0.9, 1.0); sl.data.shadow_soft_size = 1
    a0, a1 = (-25, 20) if i == 0 else (30, -15)
    sl.rotation_euler = (math.radians(162), math.radians(a0), 0); sl.keyframe_insert('rotation_euler', frame=1)   # spots aim down -Z; ~160 deg points them skyward
    sl.rotation_euler = (math.radians(158), math.radians(a1), 0); sl.keyframe_insert('rotation_euler', frame=sc.frame_end)
blib.box((0, WALL_Y + 20, 312), (2000, 12, 8), wall_m, 'crown2')
# maintenance lights: small emissive cubes in rows on the ribs
lamp_m = blib.mat('lamp', (1, 1, 1), emit=(1.0, 0.55, 0.25), emit_strength=40)
random.seed(3)
for i in range(700):
    x = random.uniform(-900, 900); z = random.choice([40, 90, 160, 230, 296]) + random.uniform(-2, 2)
    blib.box((x, WALL_Y - 13, z), (0.6, 0.3, 0.6), lamp_m, 'lamp')
# warning beacons on the crown
red_m = blib.mat('red', (1, 0, 0), emit=(1, 0.05, 0.02), emit_strength=200)
for x in range(-900, 901, 120):
    blib.box((x, WALL_Y + 4, 322), (0.9, 0.9, 0.9), red_m, 'beacon')

# city glow beyond the wall: a huge emissive gradient plane far behind
glow = bpy.data.materials.new('glow'); glow.use_nodes = True
gn = glow.node_tree; gn.nodes.remove(gn.nodes['Principled BSDF'])
em = gn.nodes.new('ShaderNodeEmission'); tc = gn.nodes.new('ShaderNodeTexCoord'); sep = gn.nodes.new('ShaderNodeSeparateXYZ')
rp = gn.nodes.new('ShaderNodeValToRGB')
c_city = (1.0, 0.42, 0.12) if not dawn else (0.55, 0.6, 0.66)
rp.color_ramp.elements[0].color = (*c_city, 1); rp.color_ramp.elements[0].position = 0.0
rp.color_ramp.elements[1].color = (0.02, 0.025, 0.04, 1); rp.color_ramp.elements[1].position = 0.7
gn.links.new(tc.outputs['Generated'], sep.inputs[0]); gn.links.new(sep.outputs['Z'], rp.inputs['Fac'])
gn.links.new(rp.outputs['Color'], em.inputs['Color']); em.inputs['Strength'].default_value = 9.0 if not dawn else 3.0
gn.links.new(em.outputs[0], gn.nodes['Material Output'].inputs['Surface'])
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 1200, 400), rotation=(math.radians(90), 0, 0))
gp = bpy.context.object; gp.scale = (6000, 900, 1); gp.data.materials.append(glow)
gp.hide_render = True   # the sky gradient now carries the city glow; the panel left a hard edge

# sky and haze
if dawn:
    blib.sky_gradient((0.5, 0.52, 0.55), (0.12, 0.13, 0.15), 1.0, clouds=0.5, band=0.4)
    blib.fog_box((0, -150, 250), (6000, 2200, 700), 0.0012, (0.7, 0.75, 0.8), noise=0.004)
else:
    blib.sky_gradient((0.55, 0.22, 0.07), (0.004, 0.005, 0.009), 1.0, clouds=0.7, band=0.22)   # sodium glow under black cloud
    blib.fog_box((0, -150, 250), (6000, 2200, 700), 0.0010, (0.55, 0.62, 0.75), noise=0.004)
# moonless overcast: a weak cold area light from above
bpy.ops.object.light_add(type='SUN', rotation=(math.radians(35), 0, math.radians(200)))
sun = bpy.context.object; sun.data.energy = 0.05 if not dawn else 1.5; sun.data.color = (0.6, 0.7, 0.9)

# spinner crossing far away, left to right, high above the crown (shot 1 only)
if not dawn:
    s = blib.spinner((-700, WALL_Y - 120, 380), scale=3.0)
    beacon = blib.box((0, 0, 1.6), (0.5, 0.5, 0.35), blib.mat('beacon_s', (1, 1, 1), emit=(1.0, 0.25, 0.15), emit_strength=900), 'sbeacon')
    beacon.parent = s   # visible from any angle at 1 km
    blib.key(s, 1, location=(-700, WALL_Y - 120, 380)); blib.key(s, sc.frame_end, location=(520, WALL_Y - 160, 395))
    s.rotation_euler = (0, 0, math.radians(-90))
    for fc in s.animation_data.action.fcurves:
        for k in fc.keyframe_points: k.interpolation = 'LINEAR'

# camera: low over the waves, 24 mm, a barely perceptible push-in
cam = blib.camera((0, -900, 5), (95, 0, 0), lens=21)
blib.key(cam, 1, location=(0, -900, 5)); blib.key(cam, sc.frame_end, location=(0, -860, 5.6))
blib.smooth_keys(cam)
sc.world.mist_settings.start = 100; sc.world.mist_settings.depth = 1500

if __name__ == '__main__':
    out = a[a.index('out') + 1] if 'out' in a else '//shot01/'
    f0 = int(a[a.index('f0') + 1]) if 'f0' in a else 1
    f1 = int(a[a.index('f1') + 1]) if 'f1' in a else sc.frame_end
    step = int(a[a.index('step') + 1]) if 'step' in a else 2
    blib.render_range(out, f0, f1, step)
