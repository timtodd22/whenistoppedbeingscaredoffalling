"""Foreground masks for lyrics-behind-subject, from Depth Anything V2 Small (Apache-2.0).

For every frame where a lyric is on screen: relative depth -> foreground = clearly
nearer than the frame's own depth split (Otsu). A shot only gets occlusion when its
foreground actually crosses the lyric zone but covers less than ~40% of it (words
stay readable) and the depth contrast is strong. Masks are smoothed over time and
saved at quarter resolution in occ_v4.npz.
"""
import json, subprocess
import numpy as np, cv2, torch
from transformers import pipeline
from PIL import Image
from plan_v4 import CUTS, N
import lyrics_v4

torch.set_num_threads(4)
W, H, q = 1080, 1920, 4
pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device='cpu')
LY = lyrics_v4.Lyrics()
active = np.zeros(N, bool)
for start, end, style, words in LY.lines:
    if style in ('body', 'big', 'hero'):
        active[max(0, int(start * 30)):min(N, int(end * 30) + 1)] = True
ZONE = (slice(1060 // q, 1440 // q), slice(90 // q, 930 // q))      # where lyric rows sit
masks = np.zeros((N, H // q, W // q), np.uint8)
stats = {}
dec = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', 'conform_v4.mp4', '-vf', f'scale={W // 2}:{H // 2}',
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
shots = list(zip([0] + CUTS, CUTS + [N]))
for g in range(N):
    fr = np.frombuffer(dec.stdout.read(W * H * 3 // 4), np.uint8).reshape(H // 2, W // 2, 3)
    if not active[g]:
        continue
    d = np.asarray(pipe(Image.fromarray(fr))['predicted_depth'], np.float32)
    d = cv2.resize(d, (W // q, H // q), interpolation=cv2.INTER_LINEAR)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    t, _ = cv2.threshold((d * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    T = max(t / 255, 0.5)
    fg = np.clip((d - T) / 0.08 + 0.5, 0, 1)
    contrast = d[fg > 0.5].mean() - d[fg < 0.5].mean() if (fg > 0.5).any() and (fg < 0.5).any() else 0
    masks[g] = (fg * 255).astype(np.uint8)
    si = max(i for i, (a, b) in enumerate(shots) if a <= g)
    stats.setdefault(si, []).append((float(fg[ZONE].mean()), float(contrast)))
    if g % 100 == 0:
        print(g, flush=True)
enabled = []
for si, v in stats.items():
    cov = np.median([x for x, _ in v]); con = np.median([c for _, c in v])
    ok = 0.04 < cov < 0.40 and con > 0.30
    a, b = shots[si]
    if ok:
        enabled.append((a, b))
    else:
        masks[a:b] = 0
    print(f'shot {a:4d}-{b:4d}: zone coverage {cov:.2f}, contrast {con:.2f} -> {"OCCLUDE" if ok else "off"}')
# temporal smoothing inside shots
for a, b in enabled:
    acc = masks[a].astype(np.float32)
    for g in range(a, b):
        acc = 0.55 * acc + 0.45 * masks[g]; masks[g] = acc.astype(np.uint8)
np.savez_compressed('occ_v4.npz', masks=masks, enabled=np.array(enabled))
