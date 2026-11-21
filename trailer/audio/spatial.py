"""Object-based spatial audio renderer: binaural (HRTF) and 7.1.4 from the same scene.

Every sound is an Obj: a mono signal plus a path (azimuth, elevation, distance over
time). Azimuth follows the SOFA convention: 0 = front, +90 = left, 180 = behind,
elevation +90 = straight up.

Binaural: SADIE II KEMAR-style HRIRs (8802 directions, 48 kHz). Each object is
processed in 256-sample blocks; every block is convolved with both the previous and
the current HRIR and crossfaded, so moving sources glide without zipper noise.
Distance gives 1/r gain, air absorption (high frequencies fall away with distance)
and a propagation delay, so flybys get real Doppler shift.

7.1.4: the same objects, distance and Doppler, panned with smooth power-normalised
cosine-lobe weights over a standard 7.1.4 layout (Dolby channel order
L R C LFE Ls Rs Lrs Rrs Ltf Rtf Ltr Rtr). LFE gets the sub content of objects
flagged lfe=True. This is a channel-based 7.1.4 render, not a Dolby Atmos (ADM)
object master; that needs Dolby's own tools.
"""
import numpy as np, h5py
from scipy.signal import fftconvolve, butter, sosfilt

SR = 48000
BLK = 256
C = 343.0
HRTF_PATH = None
_H = {}


def load_hrtf(path):
    f = h5py.File(path, 'r')
    ir = f['Data.IR'][:].astype(np.float32)          # (M, 2, 256)
    pos = f['SourcePosition'][:]
    az, el = np.radians(pos[:, 0]), np.radians(pos[:, 1])
    vec = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    _H['ir'] = ir; _H['vec'] = vec
    _H['fft'] = np.fft.rfft(ir, 2 * BLK, axis=2)     # (M, 2, BLK+1)


def dirvec(az, el):
    az, el = np.radians(az), np.radians(el)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


class Obj:
    """A sound object. path(t) -> (az_deg, el_deg, dist_m), t in seconds of the object's own timeline."""

    def __init__(self, sig, start, path, gain=1.0, name='', reverb=0.0, room='city', lfe=False, doppler=True, air=True):
        self.sig = np.asarray(sig, np.float32); self.start = start; self.path = path
        self.gain = gain; self.name = name; self.reverb = reverb; self.room = room
        self.lfe = lfe; self.doppler = doppler; self.air = air


def _prep(o, total_len):
    """Place the object on the master timeline with distance gain, air absorption and Doppler delay.
    Returns (signal on master timeline, per-block direction vectors)."""
    n = len(o.sig)
    t_local = np.arange(n) / SR
    nb = int(np.ceil(total_len / BLK))
    blk_t = (np.arange(nb) * BLK + BLK / 2) / SR - o.start
    # path sampled per block (direction) and per 64 samples (distance)
    P = np.array([o.path(max(0.0, min(t, n / SR))) for t in blk_t])
    dirs = dirvec(P[:, 0], P[:, 1])
    ds = np.arange(0, n, 64)
    D = np.array([o.path(t / SR)[2] for t in ds])
    dist = np.interp(np.arange(n), ds, D).astype(np.float32)
    # level is set at the listener: distance only changes level as the object moves (approach / recede),
    # so a static 900 m thunder is not 59 dB down, while a flyby still swells and fades
    ref = float(np.median(dist))
    x = o.sig * o.gain * (ref / np.maximum(dist, 0.3))
    if o.doppler:
        # source emitted at t arrives at t + d(t)/c: resample onto the arrival grid
        arrive = t_local + dist / C
        x = np.interp(t_local + dist[0] / C, arrive, x, left=0, right=0).astype(np.float32)
    if o.air:
        # air absorption: blend toward a lowpassed copy with distance (about -6 dB at 6 kHz per 100 m)
        sos = butter(2, 3000, 'low', fs=SR, output='sos')
        lp = sosfilt(sos, x).astype(np.float32)
        k = np.clip(dist / 150.0, 0, 0.85)
        x = x * (1 - k) + lp * k
    out = np.zeros(nb * BLK, np.float32)
    s0 = int(round(o.start * SR))
    a, b = max(0, s0), min(nb * BLK, s0 + n)
    if b > a:
        out[a:b] = x[a - s0:b - s0]
    return out, dirs, P


def render_binaural(objs, total_s, reverb_irs, wet_env=None):
    """wet_env: optional per-sample gain on the reverb returns (ducks tails under dialogue)."""
    total = int(total_s * SR)
    nb = int(np.ceil(total / BLK))
    L = np.zeros(nb * BLK + 2 * BLK, np.float32); R = np.zeros_like(L)
    sends = {}
    fade = np.linspace(0, 1, 2 * BLK, dtype=np.float32)
    for o in objs:
        x, dirs, _ = _prep(o, total)
        if o.reverb > 0:
            sends.setdefault(o.room, np.zeros(nb * BLK, np.float32))
            sends[o.room][:len(x)] += x * o.reverb
        idx = np.argmax(dirs @ _H['vec'].T, axis=1)               # nearest measured direction per block
        blocks = x.reshape(nb, BLK)
        X = np.fft.rfft(blocks, 2 * BLK, axis=1)                  # (nb, BLK+1)
        Hc = _H['fft'][idx]                                       # (nb, 2, BLK+1)
        Hp = _H['fft'][np.concatenate([[idx[0]], idx[:-1]])]
        yc = np.fft.irfft(X[:, None, :] * Hc, 2 * BLK, axis=2)
        yp = np.fft.irfft(X[:, None, :] * Hp, 2 * BLK, axis=2)
        y = yp * (1 - fade) + yc * fade                           # glide from the previous HRIR to the current one
        for ch, buf in ((0, L), (1, R)):
            seg = y[:, ch, :]
            buf[:nb * BLK] += seg[:, :BLK].reshape(-1)
            buf[BLK:nb * BLK + BLK] += seg[:, BLK:].reshape(-1)
    for room, s in sends.items():
        ir = reverb_irs[room]
        for ch, buf in ((0, L), (1, R)):
            conv = fftconvolve(s, ir[:, ch])[:len(buf)]
            if wet_env is not None and room in wet_env:
                conv[:len(wet_env[room])] *= wet_env[room][:len(conv)]
            buf[:len(conv)] += conv
    return np.stack([L[:total], R[:total]], 1)


# 7.1.4 layout: (name, az, el); az +left
LAYOUT = [('L', 30, 0), ('R', -30, 0), ('C', 0, 0), ('LFE', None, None), ('Ls', 90, 0), ('Rs', -90, 0),
          ('Lrs', 140, 0), ('Rrs', -140, 0), ('Ltf', 45, 45), ('Rtf', -45, 45), ('Ltr', 135, 45), ('Rtr', -135, 45)]


def render_714(objs, total_s, reverb_irs, wet_env=None):
    total = int(total_s * SR)
    nb = int(np.ceil(total / BLK))
    spk = [(i, dirvec(a, e)) for i, (n, a, e) in enumerate(LAYOUT) if a is not None]
    out = np.zeros((nb * BLK, 12), np.float32)
    lfe_bus = np.zeros(nb * BLK, np.float32)
    sends = {}
    for o in objs:
        x, dirs, _ = _prep(o, total)
        if o.reverb > 0:
            sends.setdefault(o.room, np.zeros(nb * BLK, np.float32))
            sends[o.room][:len(x)] += x * o.reverb
        if o.lfe:
            lfe_bus[:len(x)] += x
        G = np.stack([np.maximum(0, dirs @ v) ** 3 for _, v in spk], 1)       # cosine lobes
        G /= np.sqrt((G ** 2).sum(1, keepdims=True)) + 1e-9                  # constant power
        centers = np.arange(nb) * BLK + BLK / 2
        tt = np.arange(nb * BLK)
        for j, (ci, _) in enumerate(spk):
            out[:, ci] += x * np.interp(tt, centers, G[:, j]).astype(np.float32)   # per-sample gain glide
    # reverb: decorrelated tails into every speaker
    for room, s in sends.items():
        ir = reverb_irs[room]
        for j, (ci, _) in enumerate(spk):
            tail = fftconvolve(s, ir[:, j % 2] if j < 2 else np.roll(ir[:, j % 2], 37 * j))[:len(out)]
            if wet_env is not None and room in wet_env:
                tail[:len(wet_env[room])] *= wet_env[room][:len(tail)]
            out[:len(tail), ci] += tail * 0.55
    sos = butter(4, 120, 'low', fs=SR, output='sos')
    out[:, 3] = sosfilt(sos, lfe_bus).astype(np.float32)
    return out[:total]


def reverb_ir(t60=2.5, predelay=0.02, bright=6000, seed=0, length=None, er=0.0):
    """Stereo decorrelated exponential-decay noise tail (+ optional early reflections)."""
    rng = np.random.default_rng(seed)
    n = int((length or t60 * 1.2) * SR)
    t = np.arange(n) / SR
    env = np.exp(-6.91 * t / t60)
    ir = rng.standard_normal((n, 2)).astype(np.float32) * env[:, None]
    sos = butter(2, bright, 'low', fs=SR, output='sos')
    ir = sosfilt(sos, ir, axis=0)
    pre = np.zeros((int(predelay * SR), 2), np.float32)
    ir = np.vstack([pre, ir])
    if er > 0:
        for k in range(12):
            d = int(rng.uniform(0.004, 0.06) * SR); ir[d, rng.integers(0, 2)] += er * rng.uniform(0.3, 1)
    return (ir / np.sqrt((ir ** 2).sum() / 2)).astype(np.float32)
