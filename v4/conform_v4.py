"""v4 conform: v3's beat-snapped timeline, rebuilt on the ESRGAN clean master, plus

- cold open from Real-ESRGAN-upscaled source (c01 72-96),
- unused-footage inserts replacing the edit's repeated shots and adding a
  closing ride-away beat (all upscaled, graded to the edit with a colour
  transform fitted on matched source/edit frame pairs),
- a longer ending: black frames for the end title while the song rings out.
"""
import json, math, subprocess
import numpy as np, cv2
from conform import Interp, RAMPS

W, H = 1080, 1920
N_OLD = 1987
rt = json.load(open('retime.json'))
OLD, NEW = rt['old'], rt['new']
O2N = dict(zip(OLD, NEW))
N4 = json.load(open('audio_v4.json'))['frames']
RIDE = 1868                                         # beat at 62.28 s: split the final catch shot
INSERTS = [  # (new-timeline start, end, clip, source frames)
    (O2N[704], O2N[744], 'c16', list(range(0, 54))),     # "his dragon wouldn't choose": hand on scales
    (O2N[744], O2N[780], 'c19', list(range(45, 75))),    # bloodied hand sliding off the scales
    (O2N[1007], O2N[1065], 'c13', list(range(0, 55))),   # "she looked me down": face to face with the old green
    (RIDE, O2N[1911], 'c19', list(range(0, 40))),        # "she caught us both": riding away
]
COLD = (0, 28, 'c01', list(range(72, 97)))


def colour_transform():
    """4x3 matrix mapping source-clip colours to the edit's grade, fitted on 40 matched frame pairs."""
    match = json.load(open('match.json'))
    xs, ys = [], []
    cache = {}
    for g in range(60, 1900, 45):
        c, f, s = match[g]
        if s < 0.97 or 150 <= g < 183:
            continue
        if c not in cache:
            b = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', f'src/{c}.mp4', '-vf', 'scale=120:214',
                                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
            cache[c] = np.frombuffer(b, np.uint8).reshape(-1, 214, 120, 3)
        e = subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', f'{g / 30:.4f}', '-i', 'clean_master_v4.mp4',
                            '-frames:v', '1', '-vf', 'scale=120:214', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                           capture_output=True).stdout
        xs.append(cache[c][f].reshape(-1, 3)); ys.append(np.frombuffer(e, np.uint8).reshape(-1, 3))
    X = np.concatenate(xs).astype(np.float32); Y = np.concatenate(ys).astype(np.float32)
    return np.linalg.lstsq(np.hstack([X, np.ones((len(X), 1), np.float32)]), Y, rcond=None)[0]


def graded(img, M):
    g = np.hstack([img.reshape(-1, 3).astype(np.float32), np.ones((W * H, 1), np.float32)]) @ M
    return np.clip(g, 0, 255).astype(np.uint8).reshape(H, W, 3)


def insert_frames(spec, M):
    a, b, c, frames = spec
    src = [cv2.cvtColor(cv2.imread(f'srcup/{c}_{f:03d}.png'), cv2.COLOR_BGR2RGB) for f in frames]
    ip = Interp(src)
    n = b - a
    return [graded(ip.at(j * (len(src) - 1) / max(1, n - 1)), M) for j in range(n)]


def main():
    M = colour_transform()
    dec = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', 'clean_master_v4.mp4', '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                            '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-crf', '12',
                            '-pix_fmt', 'yuv420p', 'conform_v4.mp4'], stdin=subprocess.PIPE)
    # build the v3 timeline in memory shot by shot, substituting inserts as we go
    out_idx = 0
    pending = {s[0]: s for s in INSERTS}
    skip_until = -1
    for fr in insert_frames(COLD, M):
        enc.stdin.write(fr.tobytes()); out_idx += 1
    for k in range(len(OLD) - 1):
        a, b = OLD[k], OLD[k + 1]
        frames = [np.frombuffer(dec.stdout.read(W * H * 3), np.uint8).reshape(H, W, 3) for _ in range(b - a)]
        if k == 0:
            continue
        na, nb = NEW[k], NEW[k + 1]
        ip = Interp(frames); amp = RAMPS.get(a, 0.0)
        for j in range(nb - na):
            g = na + j
            if g in pending:
                ins = pending.pop(g)
                for fr in insert_frames(ins, M):
                    enc.stdin.write(fr.tobytes()); out_idx += 1
                skip_until = ins[1]
            if g < skip_until:
                continue
            u = j / max(1, nb - na - 1)
            s = u + amp * math.sin(2 * math.pi * u) / (2 * math.pi)
            enc.stdin.write(ip.at(s * (b - a - 1)).tobytes()); out_idx += 1
        print(f'{na}-{nb}', flush=True)
    black = np.zeros((H, W, 3), np.uint8)
    while out_idx < N4:                                        # extended end-title frames
        enc.stdin.write(black.tobytes()); out_idx += 1
    enc.stdin.close(); enc.wait(); dec.kill()
    print('frames', out_idx)


if __name__ == '__main__':
    main()
