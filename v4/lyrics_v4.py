"""v4 lyrics: timing from forced alignment on the isolated vocal stem, voice-reactive
words, and depth occlusion (subjects in front of the text can pass over it).

Timing: fa_times_v4.json (MMS_FA on the Demucs vocal of the v4 audio). Around the
a cappella bridge (4.05-5.50 s) the aligner is thrown by the compressed held note,
so those words use known values: "ON" starts the held note at 2.74 s (from the
stem alignment before the bridge was built), "PURPOSE" begins where the bridge
places it (4.75 s), and "THEY SAID I'D DIE" keep their stem-aligned times.
"""
import difflib, json, math
import numpy as np, cv2
import lyrics_v3 as L3
from lyrics_v3 import LINES, norm, HIT_WORDS, LEAD

OVERRIDES = {6: 2.74, 7: 4.75, 8: 5.48, 9: 5.94, 10: 6.76, 11: 6.96}   # word index -> start (s)
ENV = np.load('vocal_env.npy')


def word_times():
    fa = json.load(open('fa_times_v4.json'))
    starts = [s for s, _ in fa['mix']]
    for i, t in OVERRIDES.items():
        starts[i] = t
    heard = json.load(open('whisper_raw.json'))['heard']
    a = [norm(w) for w in fa['words']]
    wt = [None] * len(a)
    for blk in difflib.SequenceMatcher(None, a, [h[0] for h in heard], autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            wt[blk.a + k] = heard[blk.b + k][1]
    for i in range(1, len(starts)):
        if i not in OVERRIDES and starts[i] - starts[i - 1] < 0.1 and wt[i] is not None and wt[i] > starts[i]:
            starts[i] = wt[i]
    out, i = [], 0
    for cs, ce, text, _, style in LINES:
        n = len(text.replace('|', ' ').split(' '))
        ts = []
        for t in starts[i:i + n]:
            if style == 'coda':
                t = max(t, 63.72)
            if ts:
                t = max(t, ts[-1] + 0.06)
            ts.append(t)
        out.append(ts); i += n
    return out


L3.word_times = word_times            # reuse v3 layout/sprites with the v4 timing


class Lyrics(L3.Lyrics):
    def draw(self, img, t, beat, intensity, occ=None):
        out = None
        v = float(ENV[min(len(ENV) - 1, int(t * 30))])
        for start, end, style, words in self.lines:
            if not (start - 0.05 <= t <= end + 0.02):
                continue
            if out is None:
                out = img.astype(np.float32)
            ex = min(1.0, max(0.0, (t - (end - 0.16)) / 0.16)) if style != 'coda' else 0.0
            pulse = 1 + (0.03 if style == 'body' else 0.05) * beat * max(intensity, 0.4) if style != 'coda' else 1
            latest = max((k for k, w in enumerate(words) if w[3] <= t), default=-1)
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
                bright = 1.0
                if k == latest and style != 'coda':
                    # the word being sung swells while the voice holds it
                    hold = min(1.0, max(0.0, (t - tw - 0.15) / 0.6))
                    s *= 1 + 0.10 * v * hold
                    bright = 1 + 0.18 * v
                s *= pulse * (1 + 0.06 * ex)
                alpha = (min(1.0, a * 1.8) if style != 'coda' else a) * (1 - ex)
                dy = (28 * (1 - a) - 26 * ex) if style != 'coda' else 10 * (1 - a)
                shake = (7 * (1 - a) * math.sin(97 * t + k)) if style == 'hero' else 0
                self._blit_occ(out, spr, cx + shake, cy + dy, s, alpha, bright, occ)
        return img if out is None else np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _blit_occ(out, spr, cx, cy, s, alpha, bright, occ):
        if alpha <= 0.01:
            return
        H, W = out.shape[:2]
        h, w = spr.shape[:2]
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        sp = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_LINEAR) if s != 1 else spr
        x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + nw), min(H, y0 + nh)
        if xa >= xb or ya >= yb:
            return
        crop = sp[ya - y0:yb - y0, xa - x0:xb - x0]
        a = crop[..., 3:4] * alpha
        col = crop[..., :3] * alpha * bright
        if occ is not None:                                     # subject in front hides the text...
            o = occ[ya:yb, xa:xb]
            glyph = crop[..., 3]
            hidden = float((o * glyph).sum() / (glyph.sum() + 1e-6))
            if hidden <= 0.40:                                  # ...but never more than 40% of any word
                keep = (1 - o)[..., None]
                a = a * keep; col = col * keep
        region = out[ya:yb, xa:xb]
        region[:] = region * (1 - a) + col
