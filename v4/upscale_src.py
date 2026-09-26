"""Real-ESRGAN upscale of the source frames v4 uses directly (cold open + inserts).

Each 480x854 frame is upscaled x4, area-downscaled to 1080x1920, and blended
70/30 with a sharpened Lanczos upscale so skin keeps natural texture instead
of the painterly GAN look. Frames are saved as PNGs in srcup/.
"""
import os, subprocess, sys
import numpy as np, cv2
from multiprocessing import Pool

NEED = {'c01': range(72, 97), 'c16': range(0, 54), 'c19': list(range(0, 40)) + list(range(45, 75)),
        'c13': range(0, 55)}


def job(args):
    c, frames = args
    os.environ['SR_THREADS'] = '1'
    from esrgan import upscale4
    b = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', f'src/{c}.mp4', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                       capture_output=True).stdout
    S = np.frombuffer(b, np.uint8).reshape(-1, 854, 480, 3)
    for f in frames:
        out = f'srcup/{c}_{f:03d}.png'
        if os.path.exists(out):
            continue
        sr = cv2.resize(upscale4(S[f].copy()), (1080, 1920), interpolation=cv2.INTER_AREA)
        lz = cv2.resize(S[f], (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
        lz = cv2.addWeighted(lz, 1.5, cv2.GaussianBlur(lz, (0, 0), 1.2), -0.5, 0)
        cv2.imwrite(out, cv2.cvtColor(cv2.addWeighted(sr, 0.7, lz, 0.3, 0), cv2.COLOR_RGB2BGR))
    return c


if __name__ == '__main__':
    os.makedirs('srcup', exist_ok=True)
    jobs = []
    for c, fr in NEED.items():                   # split each clip into chunks so 4 workers stay busy
        fr = list(fr)
        for i in range(0, len(fr), 14):
            jobs.append((c, fr[i:i + 14]))
    with Pool(4) as p:
        for c in p.imap_unordered(job, jobs):
            print('done chunk', c, flush=True)
