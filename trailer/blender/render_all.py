"""Render every Blender shot of THE LONG RAIN on a local GPU, then pack each shot into one video file.

    python render_all.py --blender "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
    python render_all.py --blender blender --only city_descent,shot17
    python render_all.py --pack-only          # just (re)pack existing frames

Frames go to trailer/renders/<shot>/fNNNN.png (16-bit). Packed shots go to
trailer/renders_gpu/<shot>.mov (ProRes 4444, 12-bit): small enough to commit through Git LFS,
lossless enough to grade. The edit (trailer/edit/assemble.py) uses renders_gpu/<shot>.mov
when it exists and falls back to frame folders otherwise.
Needs ffmpeg on PATH for packing.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
# script, output name, first frame, last frame, extra args
SHOTS = [
    ('shot01_seawall.py', 'shot01', 1, 144, []),
    ('shot03_dust.py', 'shot03', 1, 121, []),
    ('city.py', 'city_descent', 1, 97, ['cam', 'descent']),
    ('city.py', 'city_hologram', 1, 49, ['cam', 'hologram']),
    ('city.py', 'city_aerial', 1, 73, ['cam', 'aerial']),
    ('shot01_seawall.py', 'shot17', 1, 109, ['dawn']),
]


def render(blender, shot):
    script, name, f0, f1, extra = shot
    out = os.path.join(ROOT, 'renders', name) + os.sep
    env = dict(os.environ, BR_GPU='1')
    t = time.time()
    cmd = [blender, '-b', '--python', os.path.join(HERE, script), '--', 'out', out,
           'f0', str(f0), 'f1', str(f1), 'step', '1', *extra]
    print('>>', name, flush=True)
    subprocess.run(cmd, env=env, check=True)
    print(f'<< {name}: {f1 - f0 + 1} frames in {time.time() - t:.0f} s', flush=True)


def pack(name):
    src = os.path.join(ROOT, 'renders', name)
    if not os.path.isdir(src):
        return
    first = min(int(f[1:5]) for f in os.listdir(src) if f.endswith('.png'))
    os.makedirs(os.path.join(ROOT, 'renders_gpu'), exist_ok=True)
    dst = os.path.join(ROOT, 'renders_gpu', name + '.mov')
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate', '24', '-start_number', str(first),
                    '-i', os.path.join(src, 'f%04d.png'), '-c:v', 'prores_ks', '-profile:v', '4444',
                    '-pix_fmt', 'yuv444p12le', '-vendor', 'apl0', dst], check=True)
    print('packed', dst, f'{os.path.getsize(dst) / 1e6:.0f} MB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--blender', default='blender')
    ap.add_argument('--only', default='')
    ap.add_argument('--pack-only', action='store_true')
    a = ap.parse_args()
    todo = [s for s in SHOTS if not a.only or s[1] in a.only.split(',')]
    for s in todo:
        if not a.pack_only:
            render(a.blender, s)
        pack(s[1])
