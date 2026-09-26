"""Kinetic lyric typography, composited on the output timeline.

Line timings are the original burned-in captions' on-screen times (read back
with Tesseract), so each line lands exactly where it did in the source edit.
Words pop in with a short stagger, lines pulse on the beat, one key word per
line glows ember-orange, and the two hook lines slam in large in Cinzel to
match the "THE FALL BEFORE FLIGHT" title card.
"""
import math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1080, 1920
CX = 510                     # a little left of centre: clears TikTok's right-hand button rail
MAXW = 820
WHITE, EMBER = (246, 242, 234), (255, 156, 60)
F_BODY = 'fonts/barlow-condensed-latin-800-normal.woff'
F_HERO = 'fonts/cinzel-latin-900-normal.woff'

# (start, end, text, emphasis words, style, forced row breaks)
LINES = [
    (0.6, 5.1, "I FELL OUT OF THE SKY ON PURPOSE.", {"SKY"}, 'body', None),
    (5.6, 9.0, "THEY SAID I'D DIE ON THE PARAPET", {"PARAPET"}, 'body', None),
    (9.0, 12.5, "WET STONE, WIND, AND A NAME THEY'D FORGET", {"FORGET"}, 'body', None),
    (12.6, 16.4, "SCRIBE'S DAUGHTER, BRITTLE BONES", {"BONES"}, 'body', None),
    (16.4, 20.6, 'INES CAUGHT MY WRIST, SAID "YOU\'RE NOT ALONE"', {'ALONE"'}, 'body', None),
    (21.4, 23.8, "MY BROTHER BURNED THREE YEARS AGO", {"BURNED"}, 'body', None),
    (23.8, 28.2, "HIS DRAGON WOULDN'T CHOOSE, SAID NO, NO, NO", {"NO,", "NO"}, 'body', None),
    (28.2, 31.2, "THRESHING TOOK HER, TOOK THE GROUND", {"THRESHING"}, 'body', None),
    (31.3, 35.0, "THEN THE OLD GREEN LANDED AND SHE LOOKED ME DOWN", {"GREEN"}, 'body', None),
    (35.1, 40.6, "LET HIM GO,|LET HIM GO", {"GO,", "GO"}, 'big', True),
    (40.7, 46.1, "THE BRONZE CRACKED OPEN IN MY HAND", {"BRONZE"}, 'body', None),
    (47.2, 54.5, "THE DAY I STOPPED|BEING SCARED|OF FALLING", {"FALLING"}, 'hero', True),
    (54.7, 60.2, "WAS THE DAY|I CHOSE|WHAT TO FALL FOR", {"FALL", "FOR"}, 'hero', True),
]
STYLE = {'body': (F_BODY, 96, 0.075), 'big': (F_BODY, 130, 0.11), 'hero': (F_HERO, 118, 0.12)}
FIT_W = {'body': MAXW, 'big': MAXW, 'hero': 900}   # hero lines shrink to fit this width
CENTER_Y = {'body': 1250, 'big': 1230, 'hero': 1150}


def _sprite(word, font, emph, heavy=False):
    l, t, r, b = font.getbbox(word, stroke_width=3)
    pad = 36
    w, h = r - l + 2 * pad, b - t + 2 * pad
    org = (pad - l, pad - t)
    shadow = Image.new('L', (w, h), 0)
    ImageDraw.Draw(shadow).text(org, word, font=font, fill=255, stroke_width=6, stroke_fill=255)
    shadow = np.asarray(shadow.filter(ImageFilter.GaussianBlur(14 if heavy else 10)), np.float32) / 255 * (0.95 if heavy else 0.8)
    glow = None
    if emph:
        g = Image.new('L', (w, h), 0)
        ImageDraw.Draw(g).text(org, word, font=font, fill=255, stroke_width=4, stroke_fill=255)
        glow = np.asarray(g.filter(ImageFilter.GaussianBlur(16)), np.float32) / 255
    txt = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(txt).text(org, word, font=font, fill=EMBER if emph else WHITE,
                             stroke_width=3, stroke_fill=(20, 14, 10))
    t_ = np.asarray(txt, np.float32) / 255
    # composite shadow (black), glow (ember) and text into one premultiplied RGBA sprite
    rgb = np.zeros((h, w, 3), np.float32); a = shadow.copy()
    if glow is not None:
        ga = np.clip(glow * 1.1, 0, 1)
        rgb = rgb * (1 - ga[..., None]) + np.array(EMBER, np.float32) / 255 * ga[..., None]
        a = a + ga * (1 - a)
    ta = t_[..., 3]
    rgb = rgb * (1 - ta[..., None]) + t_[..., :3] * ta[..., None]
    a = ta + a * (1 - ta)
    return np.dstack([rgb * 255, a])          # colour already weighted by coverage (premultiplied-ish)


class Lyrics:
    def __init__(self):
        self.lines = []
        for start, end, text, emph, style, forced in LINES:
            fpath, size, stagger = STYLE[style]
            font = ImageFont.truetype(fpath, size)
            if forced:                                   # fixed row breaks: shrink until the widest row fits
                widest = max(font.getlength(r) for r in text.split('|'))
                if widest > FIT_W[style]:
                    size = int(size * FIT_W[style] / widest)
                    font = ImageFont.truetype(fpath, size)
            space = font.getlength(' ')
            rows = []
            for chunk in text.split('|'):
                cur = []
                for wd in chunk.split(' '):
                    trial = ' '.join(cur + [wd])
                    if cur and font.getlength(trial) > MAXW and not forced:
                        rows.append(cur); cur = [wd]
                    else:
                        cur.append(wd)
                rows.append(cur)
            lh = size * 1.12
            y0 = CENTER_Y[style] - lh * (len(rows) - 1) / 2
            words = []
            for ri, row in enumerate(rows):
                widths = [font.getlength(wd) for wd in row]
                tot = sum(widths) + space * (len(row) - 1)
                x = CX - tot / 2
                for wd, wdt in zip(row, widths):
                    spr = _sprite(wd, font, wd in emph, heavy=style == 'hero')
                    words.append((spr, x + wdt / 2, y0 + ri * lh))
                    x += wdt + space
            self.lines.append((start, end, style, stagger, words))

    @staticmethod
    def _ease_back(a):
        c1 = 1.9; c3 = c1 + 1
        return 1 + c3 * (a - 1) ** 3 + c1 * (a - 1) ** 2

    def draw(self, img, t, beat, intensity):
        out = None
        for start, end, style, stagger, words in self.lines:
            if not (start - 0.05 <= t <= end + 0.02):
                continue
            if out is None:
                out = img.astype(np.float32)
            ex = min(1.0, max(0.0, (t - (end - 0.16)) / 0.16))
            pulse = 1 + (0.03 if style == 'body' else 0.05) * beat * max(intensity, 0.4)
            for k, (spr, cx, cy) in enumerate(words):
                a = (t - (start + k * stagger)) / (0.13 if style == 'hero' else 0.16)
                if a <= 0:
                    continue
                a = min(1.0, a)
                if style == 'hero':          # slam: big -> 1.0 with a hard landing
                    s = 1 + 0.7 * (1 - a) ** 2
                else:
                    s = 1.3 - 0.3 * self._ease_back(a)
                s *= pulse * (1 + 0.06 * ex)
                alpha = min(1.0, a * 1.8) * (1 - ex)
                dy = 28 * (1 - a) - 26 * ex
                shake = (7 * (1 - a) * math.sin(97 * t + k)) if style == 'hero' else 0
                self._blit(out, spr, cx + shake, cy + dy, s, alpha)
        return img if out is None else np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _blit(out, spr, cx, cy, s, alpha):
        if alpha <= 0.01:
            return
        h, w = spr.shape[:2]
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        sp = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + nw), min(H, y0 + nh)
        if xa >= xb or ya >= yb:
            return
        crop = sp[ya - y0:yb - y0, xa - x0:xb - x0]
        a = crop[..., 3:4] * alpha
        region = out[ya:yb, xa:xb]
        region[:] = region * (1 - a) + crop[..., :3] * alpha      # sprite colour is premultiplied
