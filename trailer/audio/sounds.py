"""Synthesized sound design for THE LONG RAIN. Everything is generated; no samples.

All functions return mono float32 at 48 kHz unless noted.
"""
import numpy as np
from scipy.signal import butter, sosfilt, lfilter

SR = 48000
rng = np.random.default_rng(2099)


def t_(dur):
    return np.arange(int(dur * SR)) / SR


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], 'band', fs=SR, output='sos'), x).astype(np.float32)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, 'low', fs=SR, output='sos'), x).astype(np.float32)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, 'high', fs=SR, output='sos'), x).astype(np.float32)


def pink(n):
    w = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]; a = [1, -2.494956002, 2.017265875, -0.522189400]
    return (lfilter(b, a, w) * 3.5).astype(np.float32)


def env_ar(n, a, r, sustain=None):
    e = np.ones(n, np.float32)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na) if na else 1
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def smooth_noise(n, rate_hz, seed=None):
    """Slow random modulation 0..1."""
    r = np.random.default_rng(seed)
    k = max(2, int(n / SR * rate_hz) + 2)
    pts = r.random(k)
    return np.interp(np.linspace(0, k - 1, n), np.arange(k), pts).astype(np.float32)


# ---------------- weather ----------------
def rain_layer(dur, density=1.0, bright=1.0, seed=0):
    """One layer of rain: hiss bed plus thousands of individual drop ticks."""
    r = np.random.default_rng(seed); n = int(dur * SR)
    bed = bp(pink(n), 900 * bright, 9000) * 0.35
    drops = np.zeros(n, np.float32)
    count = int(dur * 900 * density)
    pos = r.integers(0, n - 400, count)
    for p_ in pos:
        f = r.uniform(1800, 7000) * bright; L = r.integers(40, 260)
        tt = np.arange(L) / SR
        drops[p_:p_ + L] += (np.sin(2 * np.pi * f * tt) * np.exp(-tt * r.uniform(300, 900)) * r.uniform(0.05, 0.4)).astype(np.float32)
    return (bed + drops * 0.6) * (0.85 + 0.3 * smooth_noise(n, 0.3, seed))


def rain_on_glass(dur, seed=1):
    """Close, heavier, lower drops striking a window."""
    r = np.random.default_rng(seed); n = int(dur * SR)
    x = bp(pink(n), 400, 5000) * 0.12
    for p_ in r.integers(0, n - 2000, int(dur * 70)):
        L = r.integers(300, 1800); tt = np.arange(L) / SR; f = r.uniform(600, 2400)
        x[p_:p_ + L] += (np.sin(2 * np.pi * f * tt) * np.exp(-tt * r.uniform(60, 200)) * r.uniform(0.1, 0.5)).astype(np.float32)
    return x


def thunder(dur=7.0, seed=3):
    n = int(dur * SR); tt = t_(dur); r = np.random.default_rng(seed)
    crack = hp(r.standard_normal(n).astype(np.float32), 1500) * np.exp(-tt * 18)
    rumble = np.zeros(n, np.float32)
    for k in range(9):
        st = r.uniform(0.05, 2.5); a = r.uniform(0.3, 1.0)
        e = np.clip((tt - st) * 8, 0, 1) * np.exp(-np.clip(tt - st, 0, None) * r.uniform(0.8, 2.0))
        rumble += lp(r.standard_normal(n).astype(np.float32), r.uniform(120, 400), 4) * e * a
    return crack * 0.6 + rumble * 1.6


def waves(dur, seed=4):
    """Pacific swell against the wall: rolling body plus crashing bursts every ~5 s."""
    n = int(dur * SR); tt = t_(dur); r = np.random.default_rng(seed)
    body = lp(pink(n), 700) * (0.5 + 0.5 * np.sin(2 * np.pi * tt / 7.5) ** 2)
    crash = np.zeros(n, np.float32)
    for st in np.arange(0.8, dur, r.uniform(4.0, 6.0)):
        e = np.clip((tt - st) * 3, 0, 1) * np.exp(-np.clip(tt - st, 0, None) * 0.9)
        crash += bp(r.standard_normal(n).astype(np.float32), 200, 6000) * e
    return body * 0.9 + crash * 0.5


def wind(dur, seed=5):
    n = int(dur * SR); m = smooth_noise(n, 0.4, seed)
    x = pink(n)
    # sweep a band through the noise with the gusts
    out = np.zeros(n, np.float32); hop = 2400
    for i in range(0, n, hop):
        seg = x[i:i + hop + 2400]
        fc = 250 + 900 * m[i]
        out[i:i + hop + 2400][:len(seg)] += bp(seg, fc * 0.6, fc * 1.6) * np.hanning(len(seg))
    return out * (0.3 + 0.9 * m)


# ---------------- city and machines ----------------
def spinner(dur, seed=6, power=1.0):
    """Flying-car drive: turbine whine with harmonics, low thrust rumble, electric hum, slight pitch wobble."""
    n = int(dur * SR); tt = t_(dur); r = np.random.default_rng(seed)
    wob = 1 + 0.004 * np.sin(2 * np.pi * 3.1 * tt) + 0.002 * (smooth_noise(n, 2, seed) - 0.5)
    f0 = 1180 * power
    ph = 2 * np.pi * np.cumsum(f0 * wob) / SR
    whine = (np.sin(ph) + 0.45 * np.sin(2.01 * ph) + 0.2 * np.sin(3.03 * ph)) * 0.18
    hum = np.sign(np.sin(2 * np.pi * np.cumsum(55 * wob) / SR)) * 0.08
    hum = lp(hum, 600)
    thrust = lp(r.standard_normal(n).astype(np.float32), 260, 4) * 1.4
    air = bp(r.standard_normal(n).astype(np.float32), 1500, 7000) * 0.25
    return (whine + hum + thrust + air).astype(np.float32)


def city_hum(dur, seed=7):
    n = int(dur * SR); r = np.random.default_rng(seed)
    base = lp(pink(n), 300) * 1.2
    tones = sum(np.sin(2 * np.pi * f * t_(dur) + r.uniform(0, 6)) * a for f, a in ((60, .05), (120, .04), (180, .02), (97, .03)))
    horns = np.zeros(n, np.float32)
    for st in r.uniform(0, dur, 5):
        L = int(r.uniform(0.3, 0.9) * SR); s0 = int(st * SR)
        tt = np.arange(L) / SR; f = r.uniform(300, 520)
        seg = (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.26 * tt)) * np.hanning(L) * 0.04
        horns[s0:s0 + L] += seg[:max(0, min(L, n - s0))]
    return base + tones.astype(np.float32) + bp(horns, 200, 2000)


def lamp_hum(dur):
    tt = t_(dur)
    x = sum(np.sin(2 * np.pi * 60 * k * tt) / k ** 1.3 for k in range(1, 9))
    buzz = bp(rng.standard_normal(len(tt)).astype(np.float32), 2000, 6000) * 0.04 * (np.sin(2 * np.pi * 120 * tt) > 0.7)
    return (x * 0.05 + buzz).astype(np.float32)


def metal_groan(dur=4.0, seed=8):
    n = int(dur * SR); tt = t_(dur); r = np.random.default_rng(seed)
    f = 48 + 30 * smooth_noise(n, 0.8, seed)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.6 * np.sin(2.7 * ph) + 0.35 * np.sin(4.1 * ph)
    grit = bp(r.standard_normal(n).astype(np.float32), 300, 1200) * (smooth_noise(n, 6, seed + 1) ** 3)
    return (x * 0.3 + grit * 0.5) * env_ar(n, 1.2, 1.5)


def hologram_shimmer(dur, seed=9):
    n = int(dur * SR); r = np.random.default_rng(seed); x = np.zeros(n, np.float32)
    for st in r.uniform(0, dur, int(dur * 35)):
        L = int(r.uniform(0.05, 0.4) * SR); s0 = int(st * SR); tt = np.arange(L) / SR
        f = r.choice([1760, 2093, 2349, 2637, 3136, 3520, 4186]) * r.uniform(0.995, 1.005)
        seg = np.sin(2 * np.pi * f * tt) * np.hanning(L) * r.uniform(0.02, 0.08)
        x[s0:s0 + L] += seg[:max(0, min(L, n - s0))]
    swirl = bp(pink(n), 3000, 9000) * 0.05 * smooth_noise(n, 1.5, seed)
    return x + swirl


def heartbeat(dur, bpm=40):
    n = int(dur * SR); x = np.zeros(n, np.float32)
    for st in np.arange(0.2, dur, 60 / bpm):
        for off, a in ((0, 1.0), (0.28, 0.7)):
            s0 = int((st + off) * SR); L = int(0.25 * SR); tt = np.arange(L) / SR
            seg = np.sin(2 * np.pi * 48 * tt * (1 - 0.3 * tt)) * np.exp(-tt * 18) * a
            x[s0:s0 + L] += seg[:max(0, min(L, n - s0))].astype(np.float32)
    return x


def breath(dur=1.6, seed=10):
    n = int(dur * SR); tt = t_(dur)
    e = np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 2
    return bp(np.random.default_rng(seed).standard_normal(n).astype(np.float32), 500, 4500) * e * 0.25


def room_tone(dur, seed=11):
    return lp(pink(int(dur * SR)), 400) * 0.08


def whoosh(dur=0.8, seed=12):
    n = int(dur * SR); tt = t_(dur); r = np.random.default_rng(seed)
    x = r.standard_normal(n).astype(np.float32); out = np.zeros(n, np.float32); hop = 1200
    for i in range(0, n, hop):
        fc = 300 + 5000 * (i / n); seg = x[i:i + hop * 2]
        out[i:i + hop * 2][:len(seg)] += bp(seg, fc * 0.7, min(fc * 1.5, 20000)) * np.hanning(len(seg))
    return out * np.sin(np.pi * tt / dur) ** 2
