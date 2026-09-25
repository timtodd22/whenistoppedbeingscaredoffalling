"""Beat-synced kinetic re-edit for TikTok (v2: caption-free source + kinetic lyrics).

Every effect is time-preserving: shot boundaries, caption timing and the
audio stay exactly where they are, so the lyrics stay in sync.

Effects: section-scaled intensity, beat zoom pulses, cut punch-ins, whip
(motion-blur) transitions pointing "down" (falling motif), radial zoom-blur
hits, white flashes on the big moments, beat-driven camera shake/roll,
per-shot drift/dutch-angle, sync-safe speed ramps, chromatic aberration,
highlight bloom, contrast/saturation grade and vignette.
"""
import json, math, subprocess, sys
import numpy as np, cv2

FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
SRC, AUDIO = 'clean_master.mp4', 'TheDayIStoppedBeingScaredOfFalling_audio_-14LUFS.m4a'
W, H, FPS, N = 1080, 1920, 30.0, 1987
AX, AY = 540.0, 1100.0                      # transform anchor
from lyrics import Lyrics
LYR = Lyrics()
cv2.setNumThreads(4)

CUTS = [52, 83, 122, 168, 254, 271, 379, 447, 462, 493, 523, 621, 660, 681, 704, 744, 770, 780,
        846, 901, 927, 937, 961, 975, 1007, 1065, 1111, 1131, 1175, 1206, 1251, 1385, 1416,
        1431, 1478, 1517, 1556, 1615, 1686, 1765, 1843, 1911]
SHOTS = list(zip([0] + CUTS, CUTS + [N]))
END_CARD = 1911
FLASH = {52: 0.55, 1065: 0.6, 1416: 0.85, 1686: 0.7, 1911: 0.5}   # cut frame -> strength
CAPTION_CHANGES = [5.1, 5.6, 9.0, 12.55, 16.4, 20.6, 21.4, 23.8, 28.2, 31.25, 35.05, 40.65,
                   46.1, 47.2, 54.6, 60.2]

bj = json.load(open('beats.json'))
BEATS = np.array(bj['beats']); BSTR = np.array(bj['strength'])


def intensity(t):
    pts = [(0, .9), (1.6, .9), (1.9, .35), (20.4, .35), (20.9, .6), (35.3, .6), (35.6, .8),
           (47.0, .8), (47.2, 1.0), (60.6, 1.0), (61.2, .4), (63.6, .4), (63.7, 0.0), (99, 0.0)]
    xs, ys = zip(*pts)
    return float(np.interp(t, xs, ys))


def beat_env(t, tau):
    """Sum of decaying pulses from recent beats (strength-weighted)."""
    k = np.searchsorted(BEATS, t, side='right')
    e = 0.0
    for j in range(max(0, k - 3), k):
        dt = t - BEATS[j]
        e += BSTR[j] * math.exp(-dt / tau)
    return min(e, 1.5)


rng = np.random.default_rng(7)
PH = rng.uniform(0, 2 * math.pi, 6)


def noise(t, i):
    return math.sin(2 * math.pi * 7.3 * t + PH[i]) + 0.5 * math.sin(2 * math.pi * 13.1 * t + PH[i + 3])


# transition style per cut: whip down / whip side / zoom-blur hit
TRANS = {}
for n, c in enumerate(CUTS):
    I = intensity(c / FPS)
    if c == END_CARD or I < 0.55:
        TRANS[c] = 'punch'
    elif c in FLASH:
        TRANS[c] = 'zoom'
    else:
        TRANS[c] = ['whip_down', 'zoom', 'whip_side', 'whip_down'][n % 4]
DIVE = 168                     # dragon -> bridge, on the 5.60 s beat (replaces the fire portal)
TRANS[DIVE] = 'dive'
DIVE_OUT, DIVE_IN = 12, 10     # frames of push-in before / pull-out after the cut
DIVE_ANCHOR = (540.0, 860.0)   # the dragon's face


def _embers():
    r = np.random.default_rng(11)
    ps = []
    for _ in range(140):
        b = int(r.integers(DIVE - 10, DIVE + 12))
        ps.append((b, int(r.integers(7, 17)), r.uniform(-40, W + 40), r.uniform(700, H + 300),
                   -r.uniform(55, 150), r.uniform(-6, 6), r.uniform(2, 5.5),
                   (255, int(r.uniform(110, 225)), int(r.uniform(30, 110)))))
    return ps
EMBERS = _embers()


def ember_layer(gi):
    live = [p for p in EMBERS if p[0] <= gi < p[0] + p[1]]
    if not live:
        return None
    q = 2
    lay = np.zeros((H // q, W // q, 3), np.float32)
    for b, life, x, y, vy, vx, sz, col in live:
        age = gi - b
        px, py = x + vx * age, y + vy * age
        fade = math.sin(math.pi * (age + 0.5) / life)
        tail = (px - vx * 0.9, py - vy * 0.9)                      # streak trails below: camera dropping
        cv2.line(lay, (int(px / q), int(py / q)), (int(tail[0] / q), int(tail[1] / q)),
                 tuple(c * fade for c in col), max(1, int(sz / q)), cv2.LINE_AA)
    glow = cv2.GaussianBlur(lay, (0, 0), 6)
    lay = cv2.resize(lay + glow * 1.5, (W, H), interpolation=cv2.INTER_LINEAR)
    return lay


def ramp_ok(a, b):
    ta, tb = a / FPS, b / FPS
    if b - a < 24 or intensity((ta + tb) / 2) < 0.6 or b > END_CARD:
        return False
    return not (a <= DIVE < b)


# ---------- image ops ----------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIG = np.clip(1.0 - 0.28 * np.clip(r - 0.55, 0, None) ** 1.6, 0.62, 1.0)[..., None].astype(np.float32)
x = np.arange(256) / 255.0
s_curve = x + 0.18 * (x - 0.5) * (1 - np.abs(2 * x - 1))            # gentle S contrast
LUT = np.clip(s_curve * 255 + 0.5, 0, 255).astype(np.uint8)


def directional_blur(img, dx, dy):
    L = int(max(abs(dx), abs(dy)))
    if L < 3:
        return img
    q = 4
    small = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
    k = max(3, L // q) | 1
    ker = np.zeros((k, k), np.float32)
    c = k // 2
    ux, uy = dx / L, dy / L
    for i in range(k):
        o = i - c
        ker[int(round(c + o * uy)), int(round(c + o * ux))] = 1
    ker /= ker.sum()
    small = cv2.filter2D(small, -1, ker, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def radial_blur(img, amt, anchor=(AX, AY)):
    """Smooth zoom blur: 5 doubling passes = 32 effective samples, no ghost copies."""
    if amt < 0.005:
        return img
    q = 2
    w, h = W // q, H // q
    acc = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    step = amt
    for _ in range(5):
        step /= 2
        M = cv2.getRotationMatrix2D((anchor[0] / q, anchor[1] / q), 0, 1 + step)
        acc = cv2.addWeighted(acc, 0.5, cv2.warpAffine(acc, M, (w, h), borderMode=cv2.BORDER_REFLECT), 0.5, 0)
    return cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)


def chroma(img, s):
    s = int(round(s))
    if s < 1:
        return img
    out = img.copy()
    out[:, s:, 0] = img[:, :-s, 0]; out[:, :s, 0] = img[:, :1, 0]
    out[:, :-s, 2] = img[:, s:, 2]; out[:, -s:, 2] = img[:, -1:, 2]
    return out


def bloom(img, k):
    q = 4
    small = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
    hi = cv2.subtract(small, np.full_like(small, 195))
    hi = cv2.GaussianBlur(hi, (0, 0), 10)
    hi = cv2.resize(hi, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img, 1.0, hi, k, 0)


def grade(img, I):
    f = img.astype(np.float32)
    lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = 1.06 + 0.10 * I
    f = lum[..., None] + (f - lum[..., None]) * sat
    f *= VIG
    return cv2.LUT(np.clip(f, 0, 255).astype(np.uint8), LUT)


# ---------- per-frame renderer ----------
def render_frame(img, gi, shot_idx, a, b):
    t = gi / FPS
    I = intensity(t)
    card = gi >= END_CARD
    u = (gi - a) / max(1, b - a - 1)

    # camera: base zoom, per-shot drift, beat pulse, cut punch
    base = 1.03 + 0.02 * I
    drift = 0.035 * (u if shot_idx % 2 == 0 else 1 - u) * (0.4 + I)
    z = base + drift
    if card:
        z = 1.0 + 0.05 * (gi - END_CARD) / (N - END_CARD)
    z += 0.035 * I * beat_env(t, 0.09)
    since_cut = gi - a
    if a > 0 and since_cut < 10:
        z += (0.07 if not card else 0.04) * max(I, 0.5) * math.exp(-since_cut / 2.4)
    if gi < 52:                                                   # hook: roar punch-in
        z += 0.10 * math.exp(-gi / 6)
    z = z if z < 1.2 else 1.2 + 0.3 * (z - 1.2)

    sh = beat_env(t, 0.14) * I * (1.0 if I > 0.7 else 0.4)
    if gi < 52:
        sh = max(sh, 0.9)
    tx = 16 * sh * noise(t, 0) + 3 * I * math.sin(t * 1.3)
    ty = 16 * sh * noise(t, 1) + 3 * I * math.cos(t * 1.1)
    rot = 0.9 * sh * noise(t, 2)
    if I >= 0.99:                                                 # drop: dutch-angle drift per shot
        rot += (1.2 if shot_idx % 2 else -1.2) * (u - 0.5)

    # whip transitions (3 frames out of shot A, 3 frames into shot B)
    wdx = wdy = 0.0
    zb = 0.0
    prof = [0.05, 0.17, 0.40]
    for cut, kind, k in ((b, TRANS.get(b), b - 1 - gi), (a, TRANS.get(a), gi - a)):
        if kind is None or not (0 <= k < 3):
            continue
        outgoing = cut == b
        m = prof[2 - k] if outgoing else prof[2 - k]
        if kind == 'whip_down':
            wdy = (-m if outgoing else m) * H * 0.6
        elif kind == 'whip_side':
            d = 1 if (CUTS.index(cut) // 4) % 2 else -1
            wdx = (-m if outgoing else m) * W * 0.9 * d
        elif kind == 'zoom':
            zb = max(zb, m * 0.55)
            if outgoing:
                z += m * 0.35
    tx += wdx; ty += wdy
    anchor = (AX, AY)
    warm = 0.0
    if DIVE - DIVE_OUT <= gi < DIVE:
        p = 1 - (DIVE - gi) / DIVE_OUT
        z += 1.1 * p ** 3; zb = max(zb, 0.40 * p ** 2); rot += 4 * p ** 3; anchor = DIVE_ANCHOR
        warm = 0.7 * p ** 5
    elif DIVE <= gi < DIVE + DIVE_IN:
        p = 1 - (gi - DIVE) / DIVE_IN
        z += 0.7 * p ** 3; zb = max(zb, 0.32 * p ** 2); rot -= 3 * p ** 3
        warm = 0.8 * math.exp(-(gi - DIVE) / 1.3)

    M = cv2.getRotationMatrix2D(anchor, rot, z)
    M[0, 2] += tx; M[1, 2] += ty
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    if wdx or wdy:
        out = directional_blur(out, wdx * 0.7, wdy * 0.7)
    if zb:
        out = radial_blur(out, zb, anchor)

    if not card:
        out = bloom(out, 0.20 + 0.35 * I)
        ca = 1.0 + 7 * I * beat_env(t, 0.07)
        if a > 0 and since_cut < 4:
            ca += 10 * I * math.exp(-since_cut / 1.5)
        out = chroma(out, ca)
        out = grade(out, I)

    if warm > 0.01:                                          # ember flash, not white
        glow = np.array((255, 150, 60), np.float32) * warm      # screen blend keeps the image under the flash
        out = (255 - (255 - out.astype(np.float32)) * (255 - glow) / 255).astype(np.uint8)
    emb = ember_layer(gi)
    if emb is not None:
        out = np.clip(out.astype(np.float32) + emb, 0, 255).astype(np.uint8)

    fa = FLASH.get(a, 0) * math.exp(-since_cut / 1.8) if a in FLASH and since_cut < 8 else 0
    if fa > 0.01:
        out = cv2.addWeighted(out, 1 - fa, np.full_like(out, 255), fa, 0)
    out = LYR.draw(out, t, beat_env(t, 0.08), I)             # lyrics stay crisp: drawn after all camera FX
    return out


def remap_shot(frames, a, b):
    n = len(frames)
    if not ramp_ok(a, b):
        return frames
    amp = 0.55 * intensity((a + b) / 2 / FPS)
    out = []
    for j in range(n):
        uu = j / (n - 1)
        s = uu + amp * math.sin(2 * math.pi * uu) / (2 * math.pi)   # fast-slow-fast, same endpoints
        p = s * (n - 1)
        i0 = int(math.floor(p)); i1 = min(n - 1, i0 + 1); fr = p - i0
        if fr < 0.05 or i0 == i1:
            out.append(frames[i0])
        else:
            out.append(cv2.addWeighted(frames[i0], 1 - fr, frames[i1], fr, 0))
    return out


def main(f0, f1, out_path, crf):
    ss = f0 / FPS
    dec = subprocess.Popen([FF, '-loglevel', 'error', '-ss', f'{ss:.4f}', '-i', SRC,
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    enc = subprocess.Popen([FF, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                            '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                            '-ss', f'{ss:.4f}', '-i', AUDIO, '-map', '0:v', '-map', '1:a',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                            '-profile:v', 'high', '-c:a', 'copy', '-shortest', '-movflags', '+faststart',
                            out_path], stdin=subprocess.PIPE)
    gi = f0
    for si, (a, b) in enumerate(SHOTS):
        if b <= f0 or a >= f1:
            continue
        frames = []
        for g in range(max(a, gi), min(b, f1)):
            buf = dec.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            frames.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
        if a >= f0 and b <= f1:
            frames = remap_shot(frames, a, b)
        for k, fr in enumerate(frames):
            g = max(a, gi) + k
            enc.stdin.write(render_frame(fr, g, si, a, b).tobytes())
        gi = max(a, gi) + len(frames)
        print(f'{gi}/{f1}', flush=True, file=sys.stderr)
    enc.stdin.close(); enc.wait(); dec.kill()


if __name__ == '__main__':
    f0, f1 = int(sys.argv[1]), int(sys.argv[2])
    main(f0, f1, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 16)
