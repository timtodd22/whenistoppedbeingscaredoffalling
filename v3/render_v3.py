"""v3 showpiece render on top of conform.mp4 (beat-snapped, ramped, cold open).

Camera: beat zoom pulses, cut punch-ins, beat shake/roll, per-shot drift.
Transitions: whips (down/side), zoom-blur hits, the dive (dragon -> bridge),
procedural film burns, the drop (letterbox snaps open + flash + slam), and an
animated end title.
Look: section grades with split-toning, halation, anamorphic streaks,
chromatic aberration, vignette, film grain, shrinking letterbox.
Lyrics: word-accurate kinetic typography (words.json from Whisper alignment),
with camera hits on "NO, NO, NO" and the hook words.
"""
import json, math, os, subprocess, sys
import numpy as np, cv2
from plan_v3 import CUTS, TRANS, FLASH, COLD, DIVE, CHORUS, DROP, END, N, FPS, section_intensity as intensity
import lyrics_v3

FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
SRC, AUDIO = 'conform.mp4', 'audio_v3.m4a'
W, H = 1080, 1920
AX, AY = 540.0, 1100.0
cv2.setNumThreads(int(os.environ.get('CV_THREADS', '4')))
SHOTS = list(zip([0] + CUTS, CUTS + [N]))
bj = json.load(open('beats.json'))
BEATS = np.array(bj['beats']); BSTR = np.array(bj['strength'])
LYR = lyrics_v3.Lyrics()
HITS = LYR.hits                     # word times that get a camera hit ("NO", "GO", "FALLING", ...)


def beat_env(t, tau):
    k = np.searchsorted(BEATS, t, side='right'); e = 0.0
    for j in range(max(0, k - 3), k):
        e += BSTR[j] * math.exp(-(t - BEATS[j]) / tau)
    return min(e, 1.5)


def hit_env(t, tau=0.07):
    e = 0.0
    for h, s in HITS:
        if 0 <= t - h < 0.5:
            e = max(e, s * math.exp(-(t - h) / tau))
    return e


rng = np.random.default_rng(7)
PH = rng.uniform(0, 2 * math.pi, 6)


def noise(t, i):
    return math.sin(2 * math.pi * 7.3 * t + PH[i]) + 0.5 * math.sin(2 * math.pi * 13.1 * t + PH[i + 3])


# ---------------- look ----------------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIG = np.clip(1.0 - 0.30 * np.clip(rr - 0.5, 0, None) ** 1.5, 0.6, 1.0)[..., None].astype(np.float32)
_x = np.arange(256) / 255.0
LUT = np.clip((_x + 0.20 * (_x - 0.5) * (1 - np.abs(2 * _x - 1))) * 255 + 0.5, 0, 255).astype(np.uint8)
GRAIN = [cv2.resize(np.random.default_rng(i).normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))
         for i in range(8)]
# per-section split-tone: (shadow tint, highlight tint, saturation)
GRADES = [(0.0, ((-6, 4, 12), (4, 2, -4), 1.00)),     # verse: cold teal shadows
          (20.7, ((-3, 2, 6), (8, 4, -4), 1.06)),      # build
          (35.5, ((2, -1, 2), (14, 6, -8), 1.10)),     # chorus: warmer
          (47.2, ((-2, 2, 8), (18, 8, -10), 1.16)),    # drop: teal/ember contrast
          (60.8, ((-6, 3, 10), (6, 3, -4), 1.00))]


def grade_params(t):
    for i in range(len(GRADES) - 1, -1, -1):
        if t >= GRADES[i][0]:
            cur = GRADES[i]; prev = GRADES[max(0, i - 1)]
            k = min(1.0, (t - cur[0]) / 0.4)                    # 0.4 s blend between sections
            lerp = lambda a, b: tuple(x + (y - x) * k for x, y in zip(a, b))
            return lerp(prev[1][0], cur[1][0]), lerp(prev[1][1], cur[1][1]), prev[1][2] + (cur[1][2] - prev[1][2]) * k


def look(img, t, I, gi):
    f = img.astype(np.float32)
    lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
    sh_t, hi_t, sat = grade_params(t)
    L = (lum / 255)[..., None]
    f = lum[..., None] + (f - lum[..., None]) * sat
    f += np.array(sh_t, np.float32) * (1 - L) ** 2 + np.array(hi_t, np.float32) * L ** 2
    # halation + anamorphic streaks from highlights (quarter res)
    q = 4
    small = cv2.resize(lum, (W // q, H // q), interpolation=cv2.INTER_AREA)
    hal = cv2.GaussianBlur(np.clip(small - 190, 0, None), (0, 0), 7)
    hal = cv2.resize(hal, (W, H))[..., None] * np.array([1.0, 0.32, 0.12], np.float32) * (0.35 + 0.35 * I)
    streak = cv2.blur(np.clip(small - 232, 0, None) * 6, (171, 1))
    streak = cv2.GaussianBlur(streak, (0, 0), 1.5)
    streak = cv2.resize(streak, (W, H))[..., None] * np.array([0.45, 0.72, 1.0], np.float32) * (0.5 + 0.8 * I)
    f += hal + np.minimum(streak, 90)
    f *= VIG
    f += GRAIN[gi % 8][..., None] * (4.0 + 3.0 * (1 - np.abs(2 * L - 1)))
    return cv2.LUT(np.clip(f, 0, 255).astype(np.uint8), LUT)


def chroma(img, s):
    s = int(round(s))
    if s < 1:
        return img
    out = img.copy()
    out[:, s:, 0] = img[:, :-s, 0]; out[:, :-s, 2] = img[:, s:, 2]
    return out


def directional_blur(img, dx, dy):
    L = int(max(abs(dx), abs(dy)))
    if L < 3:
        return img
    q = 4
    small = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
    k = max(3, L // q) | 1
    ker = np.zeros((k, k), np.float32); c = k // 2
    for i in range(k):
        o = i - c
        ker[int(round(c + o * dy / L)), int(round(c + o * dx / L))] = 1
    small = cv2.filter2D(small, -1, ker / ker.sum(), borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (W, H))


def radial_blur(img, amt, anchor):
    if amt < 0.005:
        return img
    q = 2; w, h = W // q, H // q
    acc = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA); step = amt
    for _ in range(5):
        step /= 2
        M = cv2.getRotationMatrix2D((anchor[0] / q, anchor[1] / q), 0, 1 + step)
        acc = cv2.addWeighted(acc, 0.5, cv2.warpAffine(acc, M, (w, h), borderMode=cv2.BORDER_REFLECT), 0.5, 0)
    return cv2.resize(acc, (W, H))


def screen(img, rgb, a):
    g = np.array(rgb, np.float32) * a
    return (255 - (255 - img.astype(np.float32)) * (255 - g) / 255).astype(np.uint8)


# ---------------- film burn ----------------
def burn_layer(cut, gi):
    k = gi - cut
    if not -6 <= k <= 5:
        return None
    r = np.random.default_rng(cut)
    base = cv2.resize(r.random((6, 4)).astype(np.float32), (W // 4, H // 4), interpolation=cv2.INTER_CUBIC)
    base += 0.35 * cv2.resize(r.random((24, 14)).astype(np.float32), (W // 4, H // 4), interpolation=cv2.INTER_CUBIC)
    edge = (np.linspace(0, 1, W // 4)[None, :] if cut % 2 else np.linspace(1, 0, W // 4)[None, :]).astype(np.float32)
    field = base / base.max() * 0.6 + edge * 0.7
    p = 1 - abs(k) / 6.5                                         # peaks on the cut
    th = 1.3 - 1.05 * p
    m = np.clip((field - th) / 0.25, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    col = np.dstack([np.ones_like(m), 0.28 + 0.45 * m, 0.05 + 0.30 * m ** 3]) * 235 * np.clip(m * 1.5, 0, 1)[..., None]
    return cv2.resize(col.astype(np.float32), (W, H))


# ---------------- embers (dive + end title) ----------------
def make_embers(seed, n, b0, b1, y_range, vy_range, life):
    r = np.random.default_rng(seed)
    return [(int(r.integers(b0, b1)), int(r.integers(*life)), r.uniform(-40, W + 40), r.uniform(*y_range),
             -r.uniform(*vy_range), r.uniform(-6, 6), r.uniform(2, 5.5),
             (255, int(r.uniform(110, 225)), int(r.uniform(30, 110)))) for _ in range(n)]


EMB = make_embers(11, 150, DIVE - 10, DIVE + 12, (700, H + 300), (55, 150), (7, 17)) + \
      make_embers(12, 110, END - 5, N, (H * 0.55, H + 100), (2.5, 7), (40, 90))


def ember_layer(gi):
    live = [p for p in EMB if p[0] <= gi < p[0] + p[1]]
    if not live:
        return None
    q = 2
    lay = np.zeros((H // q, W // q, 3), np.float32)
    for b, life, x, y, vy, vx, sz, col in live:
        age = gi - b
        px, py = x + vx * age + 8 * math.sin(age * 0.2 + x), y + vy * age
        fade = math.sin(math.pi * (age + 0.5) / life)
        tl = min(1.0, abs(vy) / 40)
        cv2.line(lay, (int(px / q), int(py / q)), (int((px - vx * tl) / q), int((py - vy * 0.9) / q)),
                 tuple(c * fade for c in col), max(1, int(sz / q)), cv2.LINE_AA)
    return cv2.resize(lay + cv2.GaussianBlur(lay, (0, 0), 6) * 1.5, (W, H))


# ---------------- letterbox ----------------
def bar_height(t):
    pts = [(0, 170), (20.5, 170), (21.0, 120), (35.3, 120), (35.6, 80), (DROP / FPS - 0.001, 80),
           (DROP / FPS, 0), (60.6, 0), (62.5, 150), (99, 150)]
    xs, ys = zip(*pts)
    return int(np.interp(t, xs, ys))


# ---------------- end title ----------------
from PIL import Image, ImageDraw, ImageFont, ImageFilter
_TF = ImageFont.truetype('fonts/cinzel-latin-700-normal.woff', 92)


def end_title(gi):
    t = (gi - END) / FPS
    img = Image.new('L', (W, H), 0); d = ImageDraw.Draw(img)
    for row, (text, y) in enumerate((("THE FALL", 860), ("BEFORE FLIGHT", 985))):
        track = 34 * math.exp(-t / 0.45)                        # letters close in
        widths = [_TF.getlength(ch) for ch in text]
        total = sum(widths) + track * (len(text) - 1)
        x = W / 2 - total / 2
        for i, ch in enumerate(text):
            a = min(1.0, max(0.0, (t - 0.05 * i - 0.25 * row) / 0.35))
            if a > 0:
                d.text((x, y), ch, font=_TF, fill=int(235 * a))
            x += widths[i] + track
    m = np.asarray(img, np.float32) / 255
    glow = cv2.GaussianBlur(m, (0, 0), 14)
    sweep_x = -300 + (t - 0.9) * 900                              # light sweep across the title
    band = np.exp(-((xx - sweep_x - (yy - 900) * 0.4) / 90) ** 2)
    col = np.dstack([m * 238 + glow * 120, m * 226 + glow * 70, m * 205 + glow * 30])
    col += (m * band)[..., None] * np.array([60, 50, 30], np.float32)
    return np.clip(col, 0, 255).astype(np.uint8)


# ---------------- frame ----------------
def render_frame(img, gi, si, a, b):
    t = gi / FPS
    I = intensity(t)
    if gi >= END:
        out = end_title(gi).astype(np.float32)
        emb = ember_layer(gi)
        if emb is not None:
            out += emb * 0.8
        fade_in = min(1.0, (gi - END) / 6)
        return LYR.draw(np.clip(out * fade_in, 0, 255).astype(np.uint8), t, 0.0, 0.0)
    u = (gi - a) / max(1, b - a - 1)
    since = gi - a
    z = 1.03 + 0.02 * I + 0.035 * (u if si % 2 == 0 else 1 - u) * (0.4 + I)
    z += 0.035 * I * beat_env(t, 0.09) + 0.06 * hit_env(t)
    if a > 0 and since < 10:
        z += 0.08 * max(I, 0.5) * math.exp(-since / 2.4)
    sh = beat_env(t, 0.14) * I * (1.0 if I > 0.7 else 0.4) + 0.8 * hit_env(t, 0.1)
    tx = 16 * sh * noise(t, 0) + 3 * I * math.sin(t * 1.3)
    ty = 16 * sh * noise(t, 1) + 3 * I * math.cos(t * 1.1)
    rot = 0.9 * sh * noise(t, 2)
    if I >= 0.99:
        rot += (1.2 if si % 2 else -1.2) * (u - 0.5)
    anchor = (AX, AY); zb = 0.0; wdx = wdy = 0.0; warm = 0.0
    prof = [0.05, 0.17, 0.40]
    for cut, k, outgoing in ((b, b - 1 - gi, True), (a, gi - a, False)):
        kind = TRANS.get(cut)
        if kind in ('whip_down', 'whip_side', 'zoom') and 0 <= k < 3:
            m = prof[2 - k]
            if kind == 'whip_down':
                wdy = (-m if outgoing else m) * H * 0.6
            elif kind == 'whip_side':
                d = 1 if (CUTS.index(cut) // 4) % 2 else -1
                wdx = (-m if outgoing else m) * W * 0.9 * d
            else:
                zb = max(zb, m * 0.55); z += m * 0.35 if outgoing else 0
    if DIVE - 12 <= gi < DIVE:
        p = 1 - (DIVE - gi) / 12
        z += 1.1 * p ** 3; zb = max(zb, 0.40 * p ** 2); rot += 4 * p ** 3; anchor = (540.0, 860.0); warm = 0.7 * p ** 5
    elif DIVE <= gi < DIVE + 10:
        p = 1 - (gi - DIVE) / 10
        z += 0.7 * p ** 3; zb = max(zb, 0.32 * p ** 2); rot -= 3 * p ** 3; warm = 0.8 * math.exp(-(gi - DIVE) / 1.3)
    if DROP - 8 <= gi < DROP:                                      # suck-in before the drop
        p = 1 - (DROP - gi) / 8
        z -= 0.05 * p ** 2; zb = max(zb, 0.25 * p ** 3)
    elif DROP <= gi < DROP + 12:                                   # slam
        p = 1 - (gi - DROP) / 12
        z += 0.30 * p ** 2; zb = max(zb, 0.30 * p ** 2); sh += 1.2 * p
        tx += 30 * p * noise(t * 3, 0); ty += 30 * p * noise(t * 3, 1)
    z = z if z < 1.25 else 1.25 + 0.3 * (z - 1.25)
    M = cv2.getRotationMatrix2D(anchor, rot, z)
    M[0, 2] += tx + wdx; M[1, 2] += ty + wdy
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if wdx or wdy:
        out = directional_blur(out, wdx * 0.7, wdy * 0.7)
    if zb:
        out = radial_blur(out, zb, anchor)
    out = look(out, t, I, gi)
    ca = 1.0 + 7 * I * beat_env(t, 0.07) + 12 * hit_env(t)
    if a > 0 and since < 4:
        ca += 10 * I * math.exp(-since / 1.5)
    out = chroma(out, ca)
    for c in (a, b):
        if TRANS.get(c) == 'burn':
            bl = burn_layer(c, gi)
            if bl is not None:
                out = np.clip(out.astype(np.float32) + bl, 0, 255).astype(np.uint8)
    if warm > 0.01:
        out = screen(out, (255, 150, 60), warm)
    emb = ember_layer(gi)
    if emb is not None:
        out = np.clip(out.astype(np.float32) + emb, 0, 255).astype(np.uint8)
    fa = FLASH.get(a, 0) * math.exp(-since / 1.8) if a in FLASH and since < 8 else 0
    fa = max(fa, 0.35 * hit_env(t, 0.04))
    if fa > 0.01:
        out = cv2.addWeighted(out, 1 - fa, np.full_like(out, 255), fa, 0)
    bh = bar_height(t)
    if gi >= END - 8:                                              # bars close to black into the title
        bh = int(bh + (H / 2 - bh) * ((gi - END + 8) / 8) ** 2)
    if bh > 0:
        out[:bh] = 0; out[H - bh:] = 0
    return LYR.draw(out, t, beat_env(t, 0.08), I)


def main(f0, f1, out_path, crf, audio=True):
    dec = subprocess.Popen([FF, '-loglevel', 'error', '-ss', f'{f0 / FPS:.4f}', '-i', SRC, '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    enc = subprocess.Popen([FF, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                            '-r', '30', '-i', '-'] +
                           (['-ss', f'{f0 / FPS:.4f}', '-i', AUDIO, '-map', '0:v', '-map', '1:a', '-c:a', 'copy', '-shortest']
                            if audio else ['-an']) +
                           ['-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                            '-profile:v', 'high', '-movflags', '+faststart', out_path],
                           stdin=subprocess.PIPE)
    for gi in range(f0, f1):
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        si = max(i for i, (a, b) in enumerate(SHOTS) if a <= gi)
        a, b = SHOTS[si]
        enc.stdin.write(render_frame(np.frombuffer(buf, np.uint8).reshape(H, W, 3), gi, si, a, b).tobytes())
        if gi % 100 == 0:
            print(f'{gi}/{f1}', file=sys.stderr, flush=True)
    enc.stdin.close(); enc.wait(); dec.kill()


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 16,
         audio='--no-audio' not in sys.argv)
