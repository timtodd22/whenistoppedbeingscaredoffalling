"""Remove burned-in captions by replacing the caption box with matched source footage.

For each edited frame: take the source clip frame it came from (found by
match.json), search nearby source frames and 2-frame blends for the best
temporal match, upscale, fit a per-channel color transform to the edit on a
ring around the caption box, and feather it into the box. Everything outside
the box is the untouched edit.
"""
import json, subprocess, sys
import numpy as np, cv2
from multiprocessing import Pool

FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
W, H, SW, SH = 1080, 1920, 480, 854
N = 1987
BOX = (40, 1480, 1040, 1715)             # x0, y0, x1, y1 caption box (union of 1- and 2-line boxes)
RING = (0, 1300, W, H)                   # color-fit / temporal-match neighborhood
FIRST, LAST = 15, 1823                   # captions exist between 0.5 s and 60.77 s
match = json.load(open('match.json'))

mask = np.zeros((H, W), np.float32)
cv2.rectangle(mask, BOX[:2], (BOX[2] - 1, BOX[3] - 1), 1.0, -1)
mask_soft = np.zeros((H, W), np.float32)
cv2.rectangle(mask_soft, (BOX[0] - 45, BOX[1] - 45), (BOX[2] + 44, BOX[3] + 44), 1.0, -1)
mask_soft = cv2.GaussianBlur(mask_soft, (0, 0), 16)[..., None]   # fully covers the box edge, then fades
mask = cv2.GaussianBlur(mask, (0, 0), 7)[..., None]
ring = np.zeros((H, W), bool)
ring[RING[1]:RING[3], RING[0]:RING[2]] = True
ring[BOX[1] - 30:BOX[3] + 30, BOX[0] - 30:BOX[2] + 30] = False
q = 4
ring_s = ring[::q, ::q]

_src = {}


def src_frames(c):
    if c not in _src:
        b = subprocess.run([FF, '-loglevel', 'error', '-i', f'src/{c}.mp4', '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
        _src[c] = np.frombuffer(b, np.uint8).reshape(-1, SH, SW, 3)
    return _src[c]


def up(img):
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)


def color_fit(src, dst, sel):
    out = np.empty_like(src, np.float32)
    for ch in range(3):
        x = src[..., ch][sel].astype(np.float32); y = dst[..., ch][sel].astype(np.float32)
        A = np.vstack([x, np.ones_like(x)]).T
        g, o = np.linalg.lstsq(A, y, rcond=None)[0]
        g = float(np.clip(g, 0.6, 1.6))
        out[..., ch] = src[..., ch] * g + o
    return out


def clean_frame(g, fin):
    if not (FIRST <= g <= LAST):
        return fin, None
    c, f, _ = match[g]
    S = src_frames(c)
    fin_s = cv2.resize(fin, (W // q, H // q), interpolation=cv2.INTER_AREA).astype(np.float32)
    best = None
    for i in range(max(0, f - 3), min(len(S), f + 4)):
        for j, wts in ((i, (1.0,)), (i + 1, (0.75, 0.5, 0.25))):
            if j >= len(S):
                continue
            for w in wts:
                cand = S[i] if j == i else cv2.addWeighted(S[i], w, S[j], 1 - w, 0)
                cs = cv2.resize(cand, (W // q, H // q), interpolation=cv2.INTER_AREA).astype(np.float32)
                cf = color_fit(cs, fin_s, ring_s)
                err = np.abs(cf - fin_s)[ring_s].mean()
                if best is None or err < best[0]:
                    best = (err, cand)
    err, cand = best
    u = up(cand)
    # sharpen to approximate the edit's AI-upscale crispness
    u = cv2.addWeighted(u, 1.6, cv2.GaussianBlur(u, (0, 0), 1.2), -0.6, 0)
    # per-channel gain/offset fitted on the ring around the box (half-res samples)
    gains = []
    for ch in range(3):
        x = u[::2, ::2, ch][ring[::2, ::2]].astype(np.float32); y = fin[::2, ::2, ch][ring[::2, ::2]].astype(np.float32)
        A = np.vstack([x, np.ones_like(x)]).T
        gains.append(np.linalg.lstsq(A, y, rcond=None)[0])
    patch = np.empty_like(u, np.float32)
    for ch in range(3):
        gch, och = gains[ch]
        patch[..., ch] = u[..., ch] * np.clip(gch, 0.6, 1.6) + och
    m = mask
    if err > 10:
        # blown-out highlights break the linear fit: match per-channel histograms instead
        for ch in range(3):
            sv = np.sort(u[..., ch][ring].ravel()); dv = np.sort(fin[..., ch][ring].ravel())
            qs = np.linspace(0, 1, 256)
            lut = np.interp(np.arange(256), np.quantile(sv, qs), np.quantile(dv, qs))
            patch[..., ch] = lut[u[..., ch]]
        m = mask_soft
    out = fin.astype(np.float32) * (1 - m) + patch * m
    return np.clip(out, 0, 255).astype(np.uint8), float(err)


def work(args):
    f0, f1 = args
    dec = subprocess.Popen([FF, '-loglevel', 'error', '-ss', f'{f0 / 30:.4f}', '-i', 'video.mp4',
                            '-frames:v', str(f1 - f0), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen([FF, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                            '-s', f'{W}x{H}', '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
                            '-crf', '12', '-pix_fmt', 'yuv420p', f'parts/p{f0:04d}.mp4'], stdin=subprocess.PIPE)
    errs = []
    for g in range(f0, f1):
        b = dec.stdout.read(W * H * 3)
        fin = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        out, e = clean_frame(g, fin)
        errs.append((g, e))
        enc.stdin.write(out.tobytes())
    enc.stdin.close(); enc.wait(); dec.kill()
    return errs


if __name__ == '__main__':
    cv2.setNumThreads(1)
    step = 100
    chunks = [(a, min(N, a + step)) for a in range(0, N, step)]
    if len(sys.argv) > 1:                 # re-render only the chunks containing these frames
        redo = [int(x) for x in sys.argv[1:]]
        with Pool(4) as p:
            p.map(work, [c for c in chunks if any(c[0] <= r < c[1] for r in redo)])
        sys.exit()
    with Pool(4) as p:
        errs = sum(p.map(work, chunks), [])
    json.dump(errs, open('clean_errs.json', 'w'))
    open('parts/list.txt', 'w').write(''.join(f"file 'p{a:04d}.mp4'\n" for a, _ in chunks))
