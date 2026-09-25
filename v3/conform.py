"""v3 conform: rebuild the picture timeline before effects.

- Cold open: the unused "girl falling through the sky" shot (c01 frames 72-96)
  replaces the dark smoke under "I fell out of the sky", cutting to the
  dragon on the 0.93 s beat.
- Beat-snapped cuts: every shot is resampled to its new beat-aligned length
  (retime.json) with optical-flow frame interpolation, so no frames are
  dropped or doubled.
- Hero speed ramps: fast-slow-fast inside chosen shots, same start/end frames.
Total length and audio sync are unchanged (1987 frames).
"""
import json, math, subprocess
import numpy as np, cv2

FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
W, H, N = 1080, 1920, 1987
cv2.setNumThreads(4)
rt = json.load(open('retime.json'))
OLD, NEW = rt['old'], rt['new']
# hero ramps keyed by original shot start: amplitude of fast-slow-fast curve
RAMPS = {1131: 0.55, 1478: 0.5, 1517: 0.5, 1615: 0.6, 1686: 0.6, 780: 0.45, 1007: 0.45}
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def flow(a, b):
    q = 2
    ga = cv2.cvtColor(cv2.resize(a, (a.shape[1] // q, a.shape[0] // q)), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (b.shape[1] // q, b.shape[0] // q)), cv2.COLOR_RGB2GRAY)
    f = DIS.calc(ga, gb, None)
    return cv2.resize(f, (a.shape[1], a.shape[0])) * q


class Interp:
    """Optical-flow in-betweens for one shot."""
    def __init__(self, frames):
        self.f = frames; self.cache = {}
        h, w = frames[0].shape[:2]
        self.gx, self.gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    def at(self, p):
        p = min(max(p, 0.0), len(self.f) - 1.0)
        i0 = int(math.floor(p)); t = p - i0
        if t < 0.04 or i0 + 1 >= len(self.f):
            return self.f[i0]
        if t > 0.96:
            return self.f[i0 + 1]
        if i0 not in self.cache:
            self.cache = {i0: (flow(self.f[i0], self.f[i0 + 1]), flow(self.f[i0 + 1], self.f[i0]))}
        f01, f10 = self.cache[i0]
        a = cv2.remap(self.f[i0], self.gx - t * f01[..., 0], self.gy - t * f01[..., 1], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
        b = cv2.remap(self.f[i0 + 1], self.gx - (1 - t) * f10[..., 0], self.gy - (1 - t) * f10[..., 1],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return cv2.addWeighted(a, 1 - t, b, t, 0)


def cold_open():
    """c01 frames 72-96 (24 fps, 480p) -> 28 output frames, upscaled and graded to the edit."""
    b = subprocess.run([FF, '-loglevel', 'error', '-i', 'src/c01.mp4', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                       capture_output=True).stdout
    S = np.frombuffer(b, np.uint8).reshape(-1, 854, 480, 3)
    # colour transform learned from c01 frames the edit itself used (edit frames 83-151 <- c01 24-63)
    match = json.load(open('match.json'))
    xs, ys = [], []
    for g in range(90, 150, 6):
        c, f, _ = match[g]
        if c != 'c01':
            continue
        e = subprocess.run([FF, '-loglevel', 'error', '-ss', f'{g / 30:.4f}', '-i', 'clean_master.mp4', '-frames:v', '1',
                            '-vf', 'scale=480:854', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
        xs.append(S[f].reshape(-1, 3)); ys.append(np.frombuffer(e, np.uint8).reshape(-1, 3))
    X = np.concatenate(xs).astype(np.float32); Y = np.concatenate(ys).astype(np.float32)
    A = np.hstack([X, np.ones((len(X), 1), np.float32)])
    M = np.linalg.lstsq(A, Y, rcond=None)[0]                       # 4x3 colour matrix
    ip = Interp([S[i] for i in range(72, 97)])
    out = []
    for j in range(28):
        fr = ip.at(j * 24 / 30)
        up = cv2.resize(fr, (W, H), interpolation=cv2.INTER_LANCZOS4)
        up = cv2.addWeighted(up, 1.6, cv2.GaussianBlur(up, (0, 0), 1.2), -0.6, 0)
        g = np.hstack([up.reshape(-1, 3).astype(np.float32), np.ones((W * H, 1), np.float32)]) @ M
        out.append(np.clip(g, 0, 255).astype(np.uint8).reshape(H, W, 3))
    return out


def main():
    dec = subprocess.Popen([FF, '-loglevel', 'error', '-i', 'clean_master.mp4', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen([FF, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                            '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-crf', '12',
                            '-pix_fmt', 'yuv420p', 'conform.mp4'], stdin=subprocess.PIPE)
    written = 0
    for fr in cold_open():
        enc.stdin.write(fr.tobytes()); written += 1
    for k in range(len(OLD) - 1):
        a, b = OLD[k], OLD[k + 1]
        frames = [np.frombuffer(dec.stdout.read(W * H * 3), np.uint8).reshape(H, W, 3) for _ in range(b - a)]
        if k == 0:
            continue                                                # replaced by the cold open
        na, nb = NEW[k], NEW[k + 1]
        n_new, n_old = nb - na, b - a
        ip = Interp(frames)
        amp = RAMPS.get(a, 0.0)
        for j in range(n_new):
            u = j / max(1, n_new - 1)
            s = u + amp * math.sin(2 * math.pi * u) / (2 * math.pi)
            enc.stdin.write(ip.at(s * (n_old - 1)).tobytes()); written += 1
        print(f'shot {a}-{b} -> {na}-{nb} ({n_old / n_new:.2f}x){" ramp" if amp else ""}', flush=True)
    enc.stdin.close(); enc.wait(); dec.kill()
    json.dump({'cuts': NEW[1:-1], 'ramps': [NEW[OLD.index(a)] for a in RAMPS]}, open('conform.json', 'w'))
    print('frames', written)


if __name__ == '__main__':
    main()
