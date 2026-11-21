"""Picture edit for THE LONG RAIN: conform, grade, 2D rain, lightning, cards, title, letterbox, encode.

python3 assemble.py <workdir> <out.mp4> [--range a:b] [--stills t1,t2,...] [--uhd]

Reads the shared edit decision list (timeline.py), so picture and the spatial mix cut together.
  blender shots: renders/<name>/fNNNN.png (16-bit, rendered on twos/threes), optical-flow
                 in-betweens to 24 fps
  ai shots:      ai/<name>.mp4 (2944x1248), scaled and centre-cropped to 2.39:1
Output: 2.39:1 picture letterboxed in 16:9 (1920x1080, or 3840x2160 with --uhd), x264, no audio
(the audio is muxed afterwards).
"""
import sys, os, math, subprocess, argparse
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from timeline import SHOTS, FPS, TOTAL, LIGHTNING, TITLE_HIT

FF = '/usr/bin/ffmpeg'
W, H = 1920, 804                    # working picture (2.39:1)
cv2.setNumThreads(4)
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
FONT_DIR = '/usr/share/fonts/opentype/inter/'
FLASHES = (LIGHTNING, 45.42)        # second strike lands on the montage cut to the wall

# per-source look: exposure, black lift, RGB gain, saturation, rain amount (0 = none)
LOOK = {
    'shot01':        dict(exp=1.45, lift=0.010, rgb=(1.00, 0.97, 0.93), sat=0.92, rain=1.0, sea=0.652),
    'shot17':        dict(exp=1.15, lift=0.014, rgb=(0.96, 0.99, 1.04), sat=0.80, rain=0.35),
    'shot03':        dict(exp=1.05, lift=0.012, rgb=(1.06, 0.98, 0.86), sat=1.05, rain=0.0),
    'city_descent':  dict(exp=1.15, lift=0.008, rgb=(0.95, 1.00, 1.06), sat=1.00, rain=0.9),
    'city_aerial':   dict(exp=1.10, lift=0.008, rgb=(0.95, 1.00, 1.06), sat=1.00, rain=0.6),
    'city_hologram': dict(exp=1.10, lift=0.008, rgb=(0.97, 0.99, 1.04), sat=1.00, rain=0.9),
    'A02_window':    dict(exp=1.00, lift=0.006, rgb=(0.98, 1.00, 1.02), sat=0.90, rain=0.0),
    'A04_interrogation': dict(exp=1.00, lift=0.006, rgb=(1.02, 1.00, 0.97), sat=0.88, rain=0.0),
    'A06_eye':       dict(exp=0.95, lift=0.004, rgb=(1.00, 1.00, 1.00), sat=0.90, rain=0.0),
    'A10_hand':      dict(exp=1.00, lift=0.006, rgb=(0.97, 1.00, 1.03), sat=0.88, rain=0.0),
    'A13_rain':      dict(exp=1.00, lift=0.006, rgb=(0.97, 1.00, 1.03), sat=0.92, rain=0.45),
    'A15_whisper':   dict(exp=1.05, lift=0.004, rgb=(1.00, 1.00, 1.00), sat=0.88, rain=0.0),
}


# ------------------------------------------------------------------ sources
def flow(a, b):
    q = 2
    ga = cv2.cvtColor((np.clip(cv2.resize(a, (W // q, H // q)), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor((np.clip(cv2.resize(b, (W // q, H // q)), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.resize(DIS.calc(ga, gb, None), (W, H)) * q


GX, GY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


class BlenderSrc:
    """Frames rendered every `step` frames; in-betweens by bidirectional flow warping."""
    def __init__(self, workdir, name):
        d = os.path.join(workdir, 'renders', name)
        self.files = sorted(f for f in os.listdir(d) if f.endswith('.png'))
        self.nums = [int(f[1:5]) for f in self.files]
        self.step = self.nums[1] - self.nums[0] if len(self.nums) > 1 else 1
        self.d = d; self.img = {}; self.fl = {}

    def load(self, i):
        if i not in self.img:
            im = cv2.imread(os.path.join(self.d, self.files[i]), cv2.IMREAD_UNCHANGED)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
            if im.shape[1] != W:
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            self.img[i] = im
            for k in [k for k in self.img if k < i - 2]:
                del self.img[k]
        return self.img[i]

    def at(self, t):
        """t: seconds into the shot's own timeline."""
        p = (1 + t * FPS - self.nums[0]) / self.step
        p = min(max(p, 0.0), len(self.files) - 1.0)
        i0 = int(math.floor(p)); u = p - i0
        if u < 0.02 or i0 + 1 >= len(self.files):
            return self.load(i0).copy()
        a, b = self.load(i0), self.load(i0 + 1)
        if i0 not in self.fl:
            self.fl = {i0: (flow(a, b), flow(b, a))}
        f01, f10 = self.fl[i0]
        wa = cv2.remap(a, GX - u * f01[..., 0], GY - u * f01[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wb = cv2.remap(b, GX - (1 - u) * f10[..., 0], GY - (1 - u) * f10[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return wa * (1 - u) + wb * u


class AISrc:
    """Decode exactly the frames a shot uses; scale 2944x1248 -> 1920x814, centre-crop to 804."""
    def __init__(self, workdir, name, t0, dur):
        cmd = [FF, '-loglevel', 'error', '-ss', f'{t0:.4f}', '-i', os.path.join(workdir, 'ai', name + '.mp4'),
               '-t', f'{dur + 0.1:.4f}', '-vf', 'scale=1920:814:flags=lanczos,crop=1920:804:0:5',
               '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-']
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        self.f = np.frombuffer(raw, np.uint16).reshape(-1, H, W, 3)

    def at(self, t):
        i = min(int(round(t * FPS)), len(self.f) - 1)
        return self.f[i].astype(np.float32) / 65535.0


class Placeholder:
    def __init__(self, name):
        self.im = np.full((H, W, 3), 0.04, np.float32)
        cv2.putText(self.im, f'{name}: rendering', (60, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0.3, 0.3, 0.3), 2)

    def at(self, t):
        return self.im.copy()


# ------------------------------------------------------------------ 2D effects
class Rain:
    """Two depths of motion-blurred streaks, scrolling; lit by the plate they fall in front of."""
    def __init__(self, seed=7):
        r = np.random.default_rng(seed)
        self.layers = []
        for n, ln, wd, speed, blur, amp in ((900, 26, 1, 38, 0.6, 0.35), (160, 70, 2, 95, 1.6, 0.55)):
            tile = np.zeros((H * 2, W), np.float32)
            for _ in range(n):
                x, y = r.integers(0, W), r.integers(0, H * 2)
                L = int(ln * r.uniform(0.6, 1.4)); dx = int(L * 0.12)
                cv2.line(tile, (x, y), (x + dx, y + L), float(r.uniform(0.4, 1.0)), wd, cv2.LINE_AA)
            tile = cv2.GaussianBlur(tile, (0, 0), blur)
            tile = np.vstack([tile, tile])                      # wraps vertically
            self.layers.append((tile, speed, amp))

    def __call__(self, img, frame, amount):
        if amount <= 0:
            return img
        lum = cv2.resize(cv2.resize(img.mean(2), (W // 16, H // 16), interpolation=cv2.INTER_AREA), (W, H))
        light = 0.03 + 1.6 * lum                                  # streaks catch the light around them
        out = img
        for k, (tile, speed, amp) in enumerate(self.layers):
            off = int(frame * speed + k * 311) % (H * 2)
            s = tile[off:off + H] * amp * amount
            out = out + (s * light)[..., None] * np.array([0.85, 0.92, 1.0], np.float32)
        return out


def sea_reflect(img, t, y0, amt=0.45):
    """The sea renders near-black at this distance; give it the wall's lights back as long, broken
    reflections on moving swell (mirror about the waterline, streak vertically, ripple horizontally)."""
    yi = int(y0 * H); n = H - yi
    src = img[max(0, yi - n):yi][::-1]
    if len(src) < n:
        src = np.vstack([src, np.zeros((n - len(src), W, 3), np.float32)])
    yy = np.arange(n, dtype=np.float32)[:, None]
    dx = (3 + yy * 0.05) * np.sin(yy * 0.21 + t * 2.3) + (2 + yy * 0.04) * np.sin(yy * 0.057 - t * 1.4 + GX[:n] * 0.004)
    ref = cv2.remap(src, GX[:n] + dx.astype(np.float32), GY[:n], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ref = cv2.GaussianBlur(ref, (0, 0), sigmaX=1.5, sigmaY=9)
    fall = (amt * np.exp(-yy / (n * 0.8)))[..., None]
    out = img.copy()
    out[yi:] = img[yi:] + ref * fall
    return out


def fractal(w, h, seed, octaves=6, base=4):
    r = np.random.default_rng(seed); out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        g = r.random((base * 2 ** o + 1, int(base * 2 ** o * w / h) + 1)).astype(np.float32)
        out += cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) * amp; tot += amp; amp *= 0.55
    out /= tot
    return (out - out.min()) / (out.max() - out.min())


class Dust:
    """Orange-dust comp for the desert: dune shading in perspective, drifting dust banks, heavy foreground."""
    def __init__(self, yh=0.54):
        self.yh = yh
        self.n1 = np.tile(fractal(W, H, 11), (1, 3))               # wide, wraps horizontally
        self.n2 = np.tile(fractal(W, H, 12, base=8), (1, 3))
        self.ground = fractal(1024, 1024, 13, octaves=5, base=3)

    def __call__(self, img, t):
        yi = int(self.yh * H)
        out = img.copy()
        # dunes: sample a noise field through a ground-plane projection, so bands crowd toward the horizon
        yy = np.arange(H - yi, dtype=np.float32)[:, None] + 2.0
        z = 220.0 / yy                                              # distance along the ground
        u = ((GX[yi:] - W / 2) * z * 0.035 + 512 + t * 3) % 1023
        v = (z * 9.0 + 40) % 1023
        g = cv2.remap(self.ground, u.astype(np.float32), np.broadcast_to(v, u.shape).astype(np.float32), cv2.INTER_LINEAR)
        gc = (g - g.mean()) / (g.std() + 1e-6)
        shade = 1.0 + 0.22 * gc * np.clip(1.4 - z / 30, 0.2, 1)        # contrast fades into the haze
        near = np.clip((yy - 40) / (H - yi), 0, 1)                   # foreground falls into shadow
        out[yi:] *= (shade * (1 - 0.55 * near ** 1.3))[..., None]
        out[:yi] *= (0.72 + 0.28 * (np.arange(yi, dtype=np.float32) / yi) ** 0.7)[:, None, None]   # sky darkens upward
        # drifting dust banks, densest at the horizon, two speeds for parallax
        prof = np.exp(-((np.arange(H) / H - self.yh + 0.04) / 0.22) ** 2)[:, None]
        o1 = int(t * 40) % W; o2 = int(t * 95) % W
        d = self.n1[:, o1:o1 + W] * 0.55 + self.n2[:, o2:o2 + W] * 0.45
        a = np.clip((d - 0.35) * 1.6, 0, 1) * (0.25 + 0.55 * prof)
        dust_col = np.array([0.80, 0.42, 0.16], np.float32) * (0.6 + 0.6 * cv2.blur(img.mean(2), (65, 65)))[..., None]
        return out * (1 - a[..., None] * 0.7) + dust_col * a[..., None] * 0.7


def lightning(t):
    """Flash envelope: strike, dip, restrike, decay."""
    v = 0.0
    for t0 in FLASHES:
        d = (t - t0) * FPS
        if 0 <= d < 14:
            v = max(v, [1.0, 0.35, 0.15, 0.9, 0.55][int(d)] if d < 5 else 0.55 * math.exp(-(d - 4) / 2.5))
    return v


def flash(img, v):
    if v <= 0:
        return img
    # lightning is white-blue: it bleaches the sodium orange out of whatever it reveals
    sky = np.linspace(1.0, 0.2, H, dtype=np.float32)[:, None, None]
    cold = np.array([0.78, 0.88, 1.0], np.float32)
    lum = img.mean(2, keepdims=True)
    lit = lum * 4.5 * cold + 0.11 * sky * cold
    return img * (1 - 0.7 * v) + lit * v


def film(img, frame, look, rng):
    x = img * look['exp'] * np.array(look['rgb'], np.float32)
    y = x.mean(2, keepdims=True)
    x = y + (x - y) * look['sat']
    # halation: highlights bleed warm into the emulsion
    hi = np.clip(x - 0.75, 0, None)
    small = cv2.resize(hi, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bloom = cv2.GaussianBlur(small, (0, 0), 3) * 0.6 + cv2.GaussianBlur(small, (0, 0), 12) * 0.4
    x = x + cv2.resize(bloom, (W, H)) * np.array([0.55, 0.30, 0.18], np.float32)
    # soft shoulder, lifted toe with a cool shadow tint
    x = x / (1 + 0.35 * np.clip(x - 0.6, 0, None))
    x = look['lift'] * np.array([0.8, 1.0, 1.15], np.float32) + x * (1 - look['lift'])
    # vignette
    r2 = ((GX - W / 2) / (W / 2)) ** 2 + ((GY - H / 2) / (H / 2)) ** 2 * 0.6
    x = x * (1 - 0.22 * np.clip(r2, 0, 1.6))[..., None]
    # lateral chromatic aberration, very slight
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1.0007)
    x[..., 0] = cv2.warpAffine(x[..., 0], M, (W, H), borderMode=cv2.BORDER_REFLECT)
    # grain: mostly luma, strongest in the mid-tones
    g = rng.standard_normal((H, W)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.65)
    yl = np.clip(x.mean(2), 0, 1)
    x = x + (g * 0.030 * (0.35 + yl * (1 - yl) * 2.6))[..., None]
    return x


# ------------------------------------------------------------------ type
def tracked(draw, cx, cy, text, font, track, fill):
    widths = [font.getlength(c) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    x = cx - total / 2
    for c, w in zip(text, widths):
        draw.text((x, cy), c, font=font, fill=fill, anchor='lm')
        x += w + track
    return total


def text_layer(lines):
    """lines: [(text, font_file, size, track_em, y_frac, grey)] -> float alpha image (H, W)."""
    im = Image.new('L', (W * 2, H * 2), 0); d = ImageDraw.Draw(im)       # 2x supersampled
    for text, ff, size, tr, yf, grey in lines:
        f = ImageFont.truetype(FONT_DIR + ff, size * 2)
        tracked(d, W, int(H * 2 * yf), text, f, tr * size * 2, int(255 * grey))
    a = np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255
    return a


CARD_CACHE = {}


def card(text, t, dur):
    if text not in CARD_CACHE:
        CARD_CACHE[text] = text_layer([(text, 'Inter-ExtraLight.otf', 40, 0.62, 0.5, 0.92)])
    a = CARD_CACHE[text]
    fade = min(1.0, t / 0.33, max(0.0, (dur - t) / 0.4))
    s = 1.0 + 0.025 * t / dur                                       # slow push
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
    a = cv2.warpAffine(a, M, (W, H))
    return np.repeat((a * fade)[..., None], 3, 2)


def title(t, dur):
    if 'title' not in CARD_CACHE:
        CARD_CACHE['title'] = text_layer([('THE LONG RAIN', 'Inter-Light.otf', 96, 0.55, 0.47, 1.0)])
        CARD_CACHE['sub'] = text_layer([('A  BLADE  RUNNER  FAN  FILM', 'Inter-Light.otf', 22, 0.5, 0.585, 0.8)])
        CARD_CACHE['legal'] = text_layer([
            ('UNOFFICIAL NON-COMMERCIAL FAN FILM.  NOT AFFILIATED WITH OR ENDORSED BY', 'Inter-Light.otf', 14, 0.25, 0.885, 0.58),
            ('ALCON ENTERTAINMENT, AMAZON MGM STUDIOS OR WARNER BROS.  ORIGINAL STORY, MUSIC AND SOUND.', 'Inter-Light.otf', 14, 0.25, 0.92, 0.58)])
    a = CARD_CACHE['title']
    # on the hit: the letters burn in hot and settle; a glint passes across them
    heat = 1.0 + 1.8 * math.exp(-t / 0.18)
    glint = np.exp(-((GX - (-200 + t / 1.6 * (W + 400))) / 90) ** 2) * (t < 1.6) * 0.9
    s = 1.0 + 0.03 * t / dur
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
    a = cv2.warpAffine(a, M, (W, H))
    glow = cv2.GaussianBlur(a, (0, 0), 14) * 0.5 * heat
    fade_out = min(1.0, max(0.0, (dur - t) / 0.5))
    sub = CARD_CACHE['sub'] * np.clip((t - 1.0) / 0.8, 0, 1)
    legal = CARD_CACHE['legal'] * np.clip((t - 1.4) / 0.6, 0, 1)
    v = (a * heat + glow + a * glint) * fade_out + (sub + legal) * fade_out
    rgb = np.stack([v * 1.0, v * 0.97, v * 0.92], 2)
    return rgb


# ------------------------------------------------------------------ edit
def shot_at(t):
    for i, s in enumerate(SHOTS):
        if s[0] <= t < s[1]:
            return i, s
    return len(SHOTS) - 1, SHOTS[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workdir'); ap.add_argument('out')
    ap.add_argument('--range', default=None); ap.add_argument('--stills', default=None); ap.add_argument('--uhd', action='store_true')
    a = ap.parse_args()
    OW, OH = (3840, 2160) if a.uhd else (1920, 1080)
    PW, PH = OW, int(round(OW / 2.39 / 2)) * 2                      # active picture
    top = (OH - PH) // 2
    nfr = int(round(TOTAL * FPS))
    f0, f1 = (0, nfr)
    if a.range:
        f0, f1 = [int(round(float(v) * FPS)) for v in a.range.split(':')]
    stills = None
    if a.stills:
        stills = sorted({int(round(float(v) * FPS)) for v in a.stills.split(',')})
        frames = stills
    else:
        frames = range(f0, f1)
    enc = None
    if not stills:
        enc = subprocess.Popen([FF, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', f'{OW}x{OH}',
                                '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12' if a.uhd else '11',
                                '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'grain',
                                '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                                '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    global DUST
    DUST = Dust()
    rain = Rain(); rng = np.random.default_rng(2099)
    cur = (None, None); src = None
    for fr in frames:
        t = fr / FPS
        i, (s0, s1, kind, name, sin, note) = shot_at(t)
        lt = t - s0
        if kind in ('blender', 'ai') and cur != (i, name):
            if kind == 'blender' and not os.path.isdir(os.path.join(a.workdir, 'renders', name)):
                src = Placeholder(name)                                  # still rendering
            else:
                src = BlenderSrc(a.workdir, name) if kind == 'blender' else AISrc(a.workdir, name, sin, s1 - s0)
            cur = (i, name)
        if kind == 'blender':
            img = src.at(sin + lt)
        elif kind == 'ai':
            img = src.at(lt)
        if kind in ('blender', 'ai'):
            look = LOOK[name]
            if name == 'shot03':
                img = DUST(img, lt + sin)
            if look.get('sea'):
                img = sea_reflect(img, t, look['sea'])
            img = rain(img, fr, look['rain'])
            if name == 'shot01':
                img = flash(img, lightning(t))
            img = film(img, fr, look, rng)
            if s0 == 4.0:                                            # out of black into the wall
                img *= min(1.0, lt / 0.75)
            if s1 >= TOTAL - 0.01:                                   # fade out the last 0.7 s
                img *= min(1.0, (TOTAL - t) / 0.7)
        elif kind == 'card':
            img = card(name, lt, s1 - s0)
        elif kind == 'title':
            img = title(lt, s1 - s0)
        else:
            img = np.zeros((H, W, 3), np.float32)
        img = np.clip(img, 0, 1)
        if (PW, PH) != (W, H):
            img = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_LANCZOS4)
            img = np.clip(img * 1.25 - cv2.GaussianBlur(img, (0, 0), 1.4) * 0.25, 0, 1)
        canvas = np.zeros((OH, OW, 3), np.float32)
        canvas[top:top + PH] = img
        out = (canvas * 65535 + 0.5).astype(np.uint16)
        if stills:
            cv2.imwrite(f'{a.out}_{fr:04d}.jpg', cv2.cvtColor((canvas[top:top + PH] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            enc.stdin.write(out.tobytes())
        if fr % 48 == 0:
            print(f'{t:5.1f}s {kind} {name}', flush=True)
    if enc:
        enc.stdin.close(); enc.wait()


if __name__ == '__main__':
    main()
