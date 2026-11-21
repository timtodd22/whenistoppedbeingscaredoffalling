"""v4: same hits and whooshes as v3b, mixed over the stem-rebuilt song (audio_v4_raw.wav).

Impact sound design mixed under the song, then renormalized to -14 LUFS / -1 dBTP.

Synthesized (no samples): sub-bass booms with a transient click on the big
cuts, noise+chirp risers into the chorus and the drop, a reverse swell into
the dive, and short panned whooshes on whip transitions.
"""
import json, subprocess
import numpy as np
from plan_v4 import CUTS, TRANS, DIVE, CHORUS, DROP, REACH, ROAR, FPS

FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
SR = 48000
rng = np.random.default_rng(3)
pcm = subprocess.run([FF, '-loglevel', 'error', '-i', 'audio_v4_raw.wav',
                      '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'], capture_output=True).stdout
song = np.frombuffer(pcm, np.float32).reshape(-1, 2).copy()
fx = np.zeros_like(song)


def onepole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR); y = np.empty_like(x); s = 0.0
    for i in range(len(x)):
        s = (1 - a) * x[i] + a * s; y[i] = s
    return y


def bandnoise(n, f_lo, f_hi):
    """Noise whose band sweeps from f_lo to f_hi, via FFT-domain shaping in short blocks."""
    out = np.zeros(n, np.float32); blk = 2048
    for s in range(0, n, blk // 2):
        seg = rng.standard_normal(blk)
        u = min(1, s / max(1, n)); fc = f_lo * (f_hi / f_lo) ** u
        F = np.fft.rfft(seg); fr = np.fft.rfftfreq(blk, 1 / SR)
        F *= np.exp(-0.5 * (np.log(np.maximum(fr, 1) / fc) / 0.45) ** 2)
        seg = np.fft.irfft(F) * np.hanning(blk)
        e = min(n, s + blk); out[s:e] += seg[:e - s].astype(np.float32)
    return out / (np.abs(out).max() + 1e-9)


def place(sig, t, gain, pan=0.0):
    i = int(t * SR)
    if i >= len(fx): return
    sig = sig[:len(fx) - i] * gain
    l, r = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
    fx[i:i + len(sig), 0] += sig * l; fx[i:i + len(sig), 1] += sig * r


def boom(dur=1.4, f0=62, f1=30):
    n = int(dur * SR); t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.18)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.5)
    click = rng.standard_normal(n) * np.exp(-t / 0.006)
    thump = onepole_lp(rng.standard_normal(n), 180) * np.exp(-t / 0.08) * 4
    s = sub + 0.25 * click + thump
    return (s / np.abs(s).max()).astype(np.float32)


def riser(dur):
    n = int(dur * SR); t = np.arange(n) / SR
    env = (t / dur) ** 2.2
    noise = bandnoise(n, 350, 7000)
    chirp = np.sin(2 * np.pi * np.cumsum(180 * (7 ** (t / dur))) / SR)
    s = (noise * 0.8 + chirp * 0.35) * env
    s[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))       # hard stop right on the hit
    return (s / np.abs(s).max()).astype(np.float32)


def whoosh(dur=0.42):
    n = int(dur * SR); t = np.arange(n) / SR
    env = np.sin(np.pi * t / dur) ** 1.5
    s = bandnoise(n, 4200, 450) * env
    return (s / np.abs(s).max()).astype(np.float32)


# no risers (the repeated sweep read as one identical effect); each hit is its own sound
place(boom(dur=0.55, f0=95, f1=48), DIVE / FPS - 0.01, 0.42)      # short, punchy: the dive
place(boom(dur=1.1, f0=50, f1=30), CHORUS / FPS - 0.01, 0.30)     # soft, deep: into the chorus
place(boom(dur=1.6, f0=62, f1=28), DROP / FPS - 0.01, 0.58)       # full: the drop
place(boom(dur=0.9, f0=75, f1=38), REACH / FPS - 0.01, 0.34)
# whooshes centred on whip cuts, alternating pan
k = 0
for c in CUTS:
    if TRANS[c] in ('whip_down', 'whip_side', 'burn'):
        place(whoosh(), c / FPS - 0.21, 0.10, pan=0.6 if k % 2 else -0.6); k += 1

mix = song + fx
raw = 'mix_raw.f32'
mix.astype(np.float32).tofile(raw)
# two-pass loudnorm back to -14 LUFS / -1 dBTP
meas = subprocess.run([FF, '-hide_banner', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', raw, '-af',
                       'loudnorm=I=-14:TP=-1:LRA=11:print_format=json', '-f', 'null', '-'], capture_output=True, text=True).stderr
j = json.loads(meas[meas.rindex('{'):meas.rindex('}') + 1])
af = (f"loudnorm=I=-14:TP=-1:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
      f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true" +
      ",aresample=192000,alimiter=limit=0.85:attack=1:release=60:level=false,aresample=48000")   # true-peak safety
subprocess.run([FF, '-loglevel', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', raw, '-af', af,
                '-ar', str(SR), '-c:a', 'aac', '-b:a', '256k', 'audio_v4.m4a'], check=True)
print('mix measured', j['input_i'], 'LUFS, TP', j['input_tp'])
