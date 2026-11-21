"""v3 lyrics: every word appears when it is sung.

Word times come from Whisper word timestamps (whisper_raw.json), matched to
the known lyric text with difflib (so mishearings like "wooden shoes" still
map onto "wouldn't choose"), then constrained by the original caption
windows: words stay in order, never start before the previous line has
ended, and in-line gaps are capped so a line never stalls half-revealed.
Camera hits are exported for "NO, NO, NO", "GO", "FALLING", etc.
"""
import difflib, json, math, re
import numpy as np, cv2
from PIL import ImageFont
from lyrics import _sprite, W, H, CX, MAXW, F_BODY, F_HERO

F_CODA = 'fonts/cinzel-latin-700-normal.woff'
# (caption start, caption end, text, emphasis, style)
LINES = [
    (0.6, 5.1, "I FELL OUT OF THE SKY ON PURPOSE.", {"SKY"}, 'body'),
    (5.6, 9.0, "THEY SAID I'D DIE ON THE PARAPET", {"PARAPET"}, 'body'),
    (9.0, 12.5, "WET STONE, WIND, AND A NAME THEY'D FORGET", {"FORGET"}, 'body'),
    (12.6, 16.4, "SCRIBE'S DAUGHTER, BRITTLE BONES", {"BONES"}, 'body'),
    (16.4, 20.6, 'INES CAUGHT MY WRIST, SAID "YOU\'RE NOT ALONE"', {'ALONE"'}, 'body'),
    (21.4, 23.8, "MY BROTHER BURNED THREE YEARS AGO", {"BURNED"}, 'body'),
    (23.8, 28.2, "HIS DRAGON WOULDN'T CHOOSE, SAID NO, NO, NO", {"NO,", "NO"}, 'body'),
    (28.2, 31.2, "THRESHING TOOK HER, TOOK THE GROUND", {"THRESHING"}, 'body'),
    (31.3, 35.0, "THEN THE OLD GREEN LANDED AND SHE LOOKED ME DOWN", {"GREEN"}, 'body'),
    (35.1, 40.6, "LET HIM GO,|LET HIM GO", {"GO,", "GO"}, 'big'),
    (40.7, 46.1, "THE BRONZE CRACKED OPEN IN MY HAND", {"BRONZE"}, 'body'),
    (47.2, 54.5, "THE DAY I STOPPED|BEING SCARED|OF FALLING", {"FALLING"}, 'hero'),
    (54.7, 60.2, "WAS THE DAY|I CHOSE|WHAT TO FALL FOR", {"FALL", "FOR"}, 'hero'),
    (60.25, 63.45, "SHE CAUGHT US BOTH,", {"BOTH,"}, 'body'),
    (63.9, 66.3, "THAT'S WHAT WINGS ARE FOR", {"WINGS"}, 'coda'),
]
STYLE = {'body': (F_BODY, 96), 'big': (F_BODY, 130), 'hero': (F_HERO, 118), 'coda': (F_CODA, 46)}
FIT_W = {'body': MAXW, 'big': MAXW, 'hero': 900, 'coda': 900}
CENTER_Y = {'body': 1250, 'big': 1230, 'hero': 1150, 'coda': 1150}
HIT_WORDS = {'NO,': 1.0, 'NO': 1.0, 'GO,': 0.8, 'GO': 0.8, 'FALLING': 0.8, 'FALL': 0.7, 'BURNED': 0.5,
             'BRONZE': 0.6, 'DAY': 0.0}
LEAD = 0.04                                   # show a word a hair before it is sung


def norm(w):
    return re.sub(r"[^a-z']", '', w.lower())


def word_times():
    """Word start times from forced alignment (fa_align.py, MMS_FA on the full mix).

    The aligner is given the exact lyrics, so it only has to find when each
    word is sung; it agrees with Whisper within 0.4 s on 85 of 99 words and
    is preferred where they differ. Only ordering is enforced, plus the
    end-card tagline waiting for the title card.
    """
    fa = json.load(open('fa_times.json'))
    starts = [s for s, _ in fa['mix']]
    # where the aligner bunches a word onto the previous one (<0.1 s), trust Whisper's later time
    heard = json.load(open('whisper_raw.json'))['heard']
    a = [norm(w) for w in fa['words']]
    wt = [None] * len(a)
    for blk in difflib.SequenceMatcher(None, a, [h[0] for h in heard], autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            wt[blk.a + k] = heard[blk.b + k][1]
    for i in range(1, len(starts)):
        if starts[i] - starts[i - 1] < 0.1 and wt[i] is not None and wt[i] > starts[i]:
            starts[i] = wt[i]
    out, i = [], 0
    for li, (cs, ce, text, _, style) in enumerate(LINES):
        n = len(text.replace('|', ' ').split(' '))
        ts = []
        for t in starts[i:i + n]:
            if style == 'coda':
                t = max(t, 63.72)                    # tagline waits for the end title
            if ts:
                t = max(t, ts[-1] + 0.06)
            ts.append(t)
        out.append(ts); i += n
    return out


class Lyrics:
    def __init__(self):
        times = word_times()
        json.dump(times, open('words.json', 'w'), indent=0)
        self.lines, self.hits = [], []
        for li, ((cs, ce, text, emph, style), wt) in enumerate(zip(LINES, times)):
            if li + 1 < len(LINES):
                ce = max(min(ce, times[li + 1][0] - 0.08), wt[-1] + 0.4)
            fpath, size = STYLE[style]
            font = ImageFont.truetype(fpath, size)
            forced = '|' in text or style in ('hero', 'coda')
            if forced:
                widest = max(font.getlength(r) for r in text.split('|'))
                if widest > FIT_W[style]:
                    size = int(size * FIT_W[style] / widest); font = ImageFont.truetype(fpath, size)
            space = font.getlength(' ')
            rows = []
            for chunk in text.split('|'):
                cur = []
                for wd in chunk.split(' '):
                    if cur and font.getlength(' '.join(cur + [wd])) > MAXW and not forced:
                        rows.append(cur); cur = [wd]
                    else:
                        cur.append(wd)
                rows.append(cur)
            lh = size * (1.12 if style != 'coda' else 1.3)
            y0 = CENTER_Y[style] - lh * (len(rows) - 1) / 2
            words, k = [], 0
            for ri, row in enumerate(rows):
                widths = [font.getlength(wd) for wd in row]
                x = CX - (sum(widths) + space * (len(row) - 1)) / 2 if style != 'coda' else \
                    W / 2 - (sum(widths) + space * (len(row) - 1)) / 2
                for wd, wdt in zip(row, widths):
                    spr = _sprite(wd, font, wd in emph, heavy=style == 'hero')
                    words.append((spr, x + wdt / 2, y0 + ri * lh, wt[k] - LEAD))
                    if HIT_WORDS.get(wd):
                        self.hits.append((wt[k], HIT_WORDS[wd]))
                    x += wdt + space; k += 1
            self.lines.append((wt[0] - LEAD, ce, style, words))

    @staticmethod
    def _ease_back(a):
        return 1 + 2.9 * (a - 1) ** 3 + 1.9 * (a - 1) ** 2

    def draw(self, img, t, beat, intensity):
        out = None
        for start, end, style, words in self.lines:
            if not (start - 0.05 <= t <= end + 0.02):
                continue
            if out is None:
                out = img.astype(np.float32)
            ex = min(1.0, max(0.0, (t - (end - 0.16)) / 0.16)) if style != 'coda' else 0.0
            pulse = 1 + (0.03 if style == 'body' else 0.05) * beat * max(intensity, 0.4) if style != 'coda' else 1
            for k, (spr, cx, cy, tw) in enumerate(words):
                dur = {'hero': 0.13, 'coda': 0.6}.get(style, 0.16)
                a = (t - tw) / dur
                if a <= 0:
                    continue
                a = min(1.0, a)
                if style == 'hero':
                    s = 1 + 0.7 * (1 - a) ** 2
                elif style == 'coda':
                    s = 1.0
                else:
                    s = 1.3 - 0.3 * self._ease_back(a)
                s *= pulse * (1 + 0.06 * ex)
                alpha = (min(1.0, a * 1.8) if style != 'coda' else a) * (1 - ex)
                dy = (28 * (1 - a) - 26 * ex) if style != 'coda' else 10 * (1 - a)
                shake = (7 * (1 - a) * math.sin(97 * t + k)) if style == 'hero' else 0
                self._blit(out, spr, cx + shake, cy + dy, s, alpha)
        return img if out is None else np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _blit(out, spr, cx, cy, s, alpha):
        if alpha <= 0.01:
            return
        h, w = spr.shape[:2]
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        sp = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_LINEAR) if s != 1 else spr
        x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + nw), min(H, y0 + nh)
        if xa >= xb or ya >= yb:
            return
        crop = sp[ya - y0:yb - y0, xa - x0:xb - x0]
        a = crop[..., 3:4] * alpha
        region = out[ya:yb, xa:xb]
        region[:] = region * (1 - a) + crop[..., :3] * alpha
