"""The spatial mix of THE LONG RAIN: every sound as a positioned object, rendered to
binaural (headphones) and 7.1.4 from the same scene.

Run: python3 mix.py <workdir>   (workdir holds vo/, score_stems.npz, hrtf/sadie_d1.sofa)
Writes trailer_binaural.wav (stereo) and trailer_714.wav (12 ch, Dolby order).
"""
import sys, os, json, subprocess
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import spatial as S
from master import lufs
import sounds as fx
from timeline import TOTAL, DIALOGUE, LIGHTNING, TITLE_HIT, SILENCE, DAWN

W = sys.argv[1]
SR = S.SR
N = int(TOTAL * SR)


def keys(sig, start, pts):
    """Gain automation: pts = [(t_master_s, gain_linear)], linear in between, held at the ends."""
    t = start + np.arange(len(sig)) / SR
    xs, ys = zip(*pts)
    return (sig * np.interp(t, xs, ys)).astype(np.float32)


def static(az, el, d):
    return lambda t: (az, el, d)


def move(p0, p1, dur, ease=True):
    def f(t):
        u = min(1.0, max(0.0, t / dur)); u = u * u * (3 - 2 * u) if ease else u
        return tuple(a + (b - a) * u for a, b in zip(p0, p1))
    return f


def circle(az0, speed_deg_s, el, d):
    return lambda t: (az0 + speed_deg_s * t, el, d)


def load_vo(fname):
    y, _ = sf.read(os.path.join(W, 'vo', fname.replace('.mp3', '.wav')), dtype='float32')
    return y if y.ndim == 1 else y.mean(1)


objs = []
O = lambda *a, **k: objs.append(S.Obj(*a, **k))

# ---------------- rain: an immersive bed of 14 layers around and above ----------------
rain_pts = [(-0.1, 0.0), (0.0, 0.25), (3.5, 0.8), (10.0, 0.8), (10.2, 0.15), (14.0, 0.15), (14.2, 0.0),   # dust: no rain
            (18.9, 0.0), (19.1, 0.12), (23.0, 0.12), (23.2, 1.0), (27.0, 1.0), (27.1, 0.05), (30.0, 0.05),
            (32.0, 0.4), (37.0, 0.4), (37.1, 0.0), (40.0, 0.0), (40.9, 0.9), (42.0, 0.9), (44.0, 1.0),
            (47.0, 1.0), (47.0, 0.0), (48.0, 0.0), (48.1, 0.06), (TITLE_HIT, 0.06), (TITLE_HIT + 0.6, 0.45), (61.0, 0.4)]
dirs = [(30, 0), (90, 0), (150, 0), (210, 0), (270, 0), (330, 0), (0, 40), (120, 40), (240, 40), (60, 70), (180, 70), (300, 70), (0, -20), (180, -20)]
for i, (az, el) in enumerate(dirs):
    r = fx.rain_layer(TOTAL, density=0.9, bright=1.0 if el >= 0 else 0.7, seed=100 + i)
    O(keys(r, 0, rain_pts) * 0.5, 0.0, static(az, el, 5 + (i % 3) * 2), name=f'rain{i}', reverb=0.15, doppler=False)
# interiors hear the rain muffled through walls
muff = fx.lp(fx.rain_layer(TOTAL, 0.8, 0.8, seed=150), 700)
O(keys(muff, 0, [(0, 0), (10.0, 0), (10.2, 0.8), (14.0, 0.8), (14.1, 0), (19.0, 0), (19.1, 0.9), (23.0, 0.9), (23.1, 0)]),
  0.0, static(0, 30, 6), name='rain_muffled', doppler=False)

# ---------------- the sea wall ----------------
for start, dur in ((4.0, 6.0), (45.4, 0.55), (DAWN, 4.4)):
    for az, d in ((70, 40), (-50, 55)):
        O(fx.waves(dur + 0.5, seed=abs(int(start * 10 + az))) * (0.4 if start > 50 else 0.9), start, static(az, -8, d), name='waves', reverb=0.2, doppler=False)
O(fx.thunder(8.0) * 1.5, LIGHTNING, static(35, 35, 900), name='thunder', reverb=0.35, doppler=False, lfe=True)
O(fx.thunder(3.0, seed=31) * 0.9, 45.4, static(-60, 30, 700), name='thunder2', reverb=0.3, doppler=False)
O(fx.spinner(6.5, seed=1, power=0.95) * 0.7, 4.0, move((150, 45, 140), (-65, 18, 220), 6.0, ease=False), name='spinner_wall', reverb=0.25)

# ---------------- Iris at the window ----------------
O(fx.rain_on_glass(4.2) * 0.3, 10.0, static(-30, 5, 0.7), name='glass_rain', doppler=False)

# ---------------- the orange dust ----------------
O(fx.wind(5.4, seed=5) * 0.9, 14.0, circle(0, 72, 5, 5), name='wind_circle', reverb=0.2, doppler=False)
O(fx.wind(5.4, seed=6) * 0.6, 14.0, static(180, 10, 8), name='wind_back', reverb=0.2, doppler=False)
O(fx.metal_groan(4.0) * 1.4, 15.3, static(25, 60, 260), name='groan', reverb=0.5, doppler=False)

# ---------------- interrogation ----------------
O(fx.lamp_hum(4.2) * 0.6, 19.0, static(0, 70, 1.4), name='lamp', doppler=False)
O(fx.room_tone(4.2) * 0.8, 19.0, static(150, 0, 3), name='room', reverb=0.4, room='room', doppler=False)

# ---------------- descent: the spinner overtakes the lens ----------------
O(fx.spinner(4.4, seed=2, power=1.1) * 1.1, 23.0, move((178, 55, 22), (2, -5, 60), 4.0), name='spinner_descent', reverb=0.2)
O(fx.whoosh(1.0) * 1.2, 24.0, move((170, 60, 3), (0, 0, 8), 1.0), name='whoosh_descent', doppler=False)

# ---------------- the eye: everything drops away except a heartbeat ----------------
O(fx.heartbeat(3.2) * 0.45, 27.0, static(0, 0, 0.4), name='heart', doppler=False, lfe=True)

# ---------------- megacity ----------------
for az in (0, 120, 240):
    O(fx.city_hum(3.4, seed=az) * 0.5, 32.0, static(az, -50, 40), name='city', reverb=0.3, doppler=False)
O(fx.spinner(3.4, seed=3) * 0.5, 32.0, move((60, -12, 90), (-35, -8, 70), 3.0, ease=False), name='spinner_city')

# ---------------- the hologram ----------------
for st, d in ((40.6, 1.6), (46.4, 0.4)):
    O(fx.hologram_shimmer(d + 0.4, seed=int(st)) * 1.6, st, circle(0, 140, 50, 6), name='holo', reverb=0.4, doppler=False)
O(fx.spinner(1.7, seed=4) * 1.0, 40.85, move((55, 30, 70), (-55, 32, 70), 1.4, ease=False), name='spinner_holo')   # after "lasting"

# ---------------- montage whooshes on every cut, alternating sides ----------------
for i, c in enumerate((44.75, 45.4, 45.95, 46.4, 46.75)):
    side = 1 if i % 2 == 0 else -1
    O(fx.whoosh(0.5, seed=40 + i) * 1.1, c - 0.25, move((90 * side, 10, 3), (-90 * side, 10, 3), 0.5), name='whoosh', doppler=False)

# ---------------- the whisper ----------------
O(fx.breath(1.0) * 0.6, 47.95, static(-80, 0, 0.25), name='breath', doppler=False, air=False)

# ---------------- score stems as objects ----------------
st = dict(np.load(os.path.join(W, 'score_stems.npz')))
music = [('drone', [(30, 0, 3.5), (-30, 0, 3.5)], 0.9, True), ('padL', [(55, 15, 4), (130, 30, 6)], 1.0, False),
         ('padR', [(-55, 15, 4), (-130, 30, 6)], 1.0, False), ('brass', [(22, 5, 5), (-22, 5, 5)], 1.1, False),
         ('hit_low', [(0, 0, 3)], 1.3, True), ('hit_sub', [(0, 0, 3)], 1.0, True),
         ('hit_metal', [(65, 45, 6), (-65, 45, 6)], 1.0, False), ('riser', [(0, 35, 4)], 1.4, False),
         ('pulse', [(0, 0, 3)], 1.1, False), ('lead', [(0, 10, 3)], 1.6, False), ('glass', [(20, 20, 3)], 1.2, False)]
for name, positions, g, lfe in music:
    for k, (az, el, d) in enumerate(positions):
        sig = st[name] if k == 0 else np.roll(st[name], int(0.017 * SR * k))   # decorrelate the copies slightly
        O(sig * g / np.sqrt(len(positions)), 0.0, static(az, el, d), name=f'mus_{name}{k}', reverb=0.25, room='hall', lfe=lfe, doppler=False)
O(st['swell'] * 1.2, 0.0, circle(0, -200, 40, 4), name='mus_swell', reverb=0.3, room='hall', doppler=False)

# ---------------- dialogue ----------------
vo = {}
l12 = load_vo('L12_a.mp3')
# split "They built us to last four years." / "I've had seventy." at the longest pause
env = np.convolve(np.abs(l12), np.ones(2400) / 2400, 'same')
quiet = env < env.max() * 0.03
mid = len(l12) // 2; best = None; run = 0
for i in range(int(len(l12) * 0.25), int(len(l12) * 0.85)):
    run = run + 1 if quiet[i] else 0
    if best is None or run > best[0]:
        best = (run, i)
split = best[1] - best[0] // 2
vo['L1'] = l12[:split]; vo['L2'] = l12[split:]
for k in ('L3', 'L4', 'L5', 'L6', 'L7'):
    vo[k] = load_vo(DIALOGUE[k][0])
dlg_env = np.zeros(N, np.float32)
for k, sig in vo.items():
    start = DIALOGUE[k][1]
    sig = sig * 10 ** ((-23 - lufs(np.stack([sig, sig], 1)) + 3.01) / 20)   # every line at the same loudness (-23 LUFS mono)
    sig = sig * {'L4': 1.45, 'L7': 1.5, 'L5': 1.15}.get(k, 1.0)          # ride the lines that sit over busy beds
    if k in ('L3', 'L4'):
        O(sig * 3.2, start, static(18, 0, 1.8), name=k, reverb=0.25, room='room', doppler=False, air=False)
    elif k == 'L6':
        O(sig * 2.2, start, static(-75, 0, 0.3), name=k, reverb=0.02, doppler=False, air=False)
    else:
        O(sig * 3.6, start, static(0, 0, 1.0), name=k, reverb=0.08, room='voice', doppler=False, air=False)
    s0 = int(start * SR); dlg_env[s0:s0 + len(sig)] = np.maximum(dlg_env[s0:s0 + len(sig)], np.convolve(np.abs(sig), np.ones(4800) / 4800, 'same')[:max(0, min(len(sig), N - s0))])

# duck music and effects (not dialogue) under the voice, up to -14 dB
duck = 1 - 0.8 * np.clip(dlg_env / (dlg_env.max() + 1e-9) * 3, 0, 1)
duck = np.convolve(duck, np.ones(9600) / 9600, 'same').astype(np.float32)
for o in objs:
    if o.name.startswith(('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7')):
        continue
    s0 = int(o.start * SR); seg = duck[s0:s0 + len(o.sig)]
    if len(seg) == len(o.sig):
        o.sig = o.sig * seg
# the silence is absolute in every object
for o in objs:
    a0 = int((SILENCE[0] - o.start) * SR); a1 = int((SILENCE[1] - o.start) * SR)
    if a1 > 0 and a0 < len(o.sig) and not o.name.startswith('breath'):
        o.sig[max(0, a0):max(0, min(len(o.sig), a1))] = 0

S.load_hrtf(os.path.join(W, 'hrtf', 'sadie_d1.sofa'))
irs = {'city': S.reverb_ir(2.8, 0.03, 5000, seed=1), 'room': S.reverb_ir(0.7, 0.008, 7000, seed=2, er=0.4),
       'hall': S.reverb_ir(4.5, 0.04, 6000, seed=3)}
irs['voice'] = irs['hall']
print(len(objs), 'objects')
if os.environ.get('PROBE'):
    # rough per-object level (mono, pre-HRTF) in probe windows, to find what masks a line
    from master import k_weight
    for a_, z_ in [tuple(map(float, w.split(':'))) for w in os.environ['PROBE'].split(',')]:
        rows = []
        for o in objs:
            i0, i1 = int((a_ - o.start) * SR), int((z_ - o.start) * SR)
            seg = o.sig[max(0, i0):max(0, i1)] * o.gain
            if len(seg) > 4800:
                rows.append((10 * np.log10((k_weight(seg) ** 2).mean() + 1e-12), o.name))
        print(f'--- {a_}-{z_}'); [print(f'  {l:6.1f} {n}') for l, n in sorted(rows, reverse=True)[:9]]
    irs_ = {'city': S.reverb_ir(2.8, 0.03, 5000, seed=1), 'room': S.reverb_ir(0.7, 0.008, 7000, seed=2, er=0.4),
            'hall': S.reverb_ir(4.5, 0.04, 6000, seed=3)}
    from scipy.signal import fftconvolve
    for room in irs_:
        snd = np.zeros(N, np.float32)
        for o in objs:
            if o.reverb > 0 and o.room == room and not o.name[:2] in ('L1','L2','L3','L4','L5','L6','L7'):
                s0 = int(o.start * SR); e = min(N, s0 + len(o.sig)); snd[s0:e] += o.sig[:e - s0] * o.gain * o.reverb
        wet = fftconvolve(snd, irs_[room][:, 0])[:N]
        for a_, z_ in [tuple(map(float, w.split(':'))) for w in os.environ['PROBE'].split(',')]:
            seg = wet[int(a_ * SR):int(z_ * SR)]
            print(f'  reverb {room:5s} {a_}: {10*np.log10((k_weight(seg)**2).mean()+1e-12):6.1f}')
    sys.exit()
if os.environ.get('DLG_STEM'):
    # dialogue alone (with its reverb), for measuring speech-to-background ratio
    D = S.render_binaural([o for o in objs if o.name[:2] in ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7')], TOTAL, irs)
    # (dialogue sends into 'hall' would be ducked by the main render; keep the stem consistent)
    sf.write(os.path.join(W, 'dlg_stem_raw.wav'), D, SR, subtype='FLOAT')
# the city and hall tails duck with the beds (the dialogue's own room reverb is left alone)
wet = {'city': duck, 'hall': duck}
B = S.render_binaural(objs, TOTAL, irs, wet)
M = S.render_714(objs, TOTAL, irs, wet)
sf.write(os.path.join(W, 'mix_binaural_raw.wav'), B, SR, subtype='FLOAT')
sf.write(os.path.join(W, 'mix_714_raw.wav'), M, SR, subtype='FLOAT')
# hard gate the silence after reverb: absolute, with 4 ms edges
g = np.ones(len(B), np.float32); a0, a1 = int(SILENCE[0] * SR), int(SILENCE[1] * SR); e = int(0.004 * SR)
g[a0:a1] = 0; g[a0 - e:a0] = np.linspace(1, 0, e); g[a1:a1 + e] = np.linspace(0, 1, e)
# the breath leads into the whisper just before the silence ends; keep its last 50 ms
B *= g[:, None]; M *= g[:, None]
sf.write(os.path.join(W, 'mix_binaural_raw.wav'), B, SR, subtype='FLOAT')
sf.write(os.path.join(W, 'mix_714_raw.wav'), M, SR, subtype='FLOAT')
print('rendered')
