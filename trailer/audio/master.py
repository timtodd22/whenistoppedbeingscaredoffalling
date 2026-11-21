"""Master THE LONG RAIN: BS.1770-4 loudness, true-peak lookahead limiter, -14 LUFS / -1 dBTP.

python3 master.py <workdir>
  reads  mix_binaural_raw.wav, mix_714_raw.wav
  writes trailer_binaural.wav (24-bit stereo) and trailer_714.wav (24-bit, 12 ch, Dolby order)

The binaural mix sets the gain; the 7.1.4 render gets the same make-up gain (so the two stay
level-matched scene for scene) and its own limiter at the same ceiling.
"""
import os, sys
import numpy as np, soundfile as sf
from scipy.signal import lfilter, resample_poly
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
TARGET, CEIL_DBTP = -14.0, -1.0
# 7.1.4: BS.1770 channel weights (LFE excluded, surrounds +1.5 dB, heights treated as surrounds)
W714 = [1, 1, 1, 0, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41]


def k_weight(x):
    # BS.1770-4 pre-filter (high shelf) + RLB high-pass, 48 kHz coefficients
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1, -1.99004745483398, 0.99007225036621]
    return lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)


def lufs(x, weights=None):
    y = k_weight(x.astype(np.float64))
    w = np.ones(y.shape[1]) if weights is None else np.asarray(weights, float)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    z = np.array([(y[i:i + blk] ** 2).mean(0) @ w for i in range(0, len(y) - blk, hop)])
    l = -0.691 + 10 * np.log10(z + 1e-12)
    z1 = z[l > -70]
    rel = -0.691 + 10 * np.log10(z1.mean()) - 10
    return -0.691 + 10 * np.log10(z1[(-0.691 + 10 * np.log10(z1)) > rel].mean())


def true_peak(x):
    return np.abs(resample_poly(x, 4, 1, axis=0)).max()


def limit(x, ceil_lin, look_ms=3.0, rel_ms=120.0):
    """Lookahead brick-wall on 4x-oversampled peaks; gain reaches its floor before each peak, releases smoothly."""
    pk = np.abs(resample_poly(x, 4, 1, axis=0)).max(1).reshape(-1, 4).max(1)[:len(x)]
    need = np.minimum(1.0, ceil_lin / np.maximum(pk, 1e-9))
    L = int(look_ms * 1e-3 * SR) | 1
    g = minimum_filter1d(need, L * 2 + 1, origin=0)                 # hold the floor across the lookahead
    g = uniform_filter1d(g, L)                                      # ramp into it (no clicks)
    a = np.exp(-1 / (rel_ms * 1e-3 * SR))
    out = np.empty_like(g); cur = 1.0
    for i, v in enumerate(g):                                       # instant attack (already ramped), slow release
        cur = v if v < cur else a * cur + (1 - a) * v
        out[i] = cur
    y = x * out[:, None]
    # delay compensation is unnecessary: the min/avg filters are centred, so the ramp straddles the peak
    return y, out


def master(binaural, m714):
    ceil = 10 ** ((CEIL_DBTP - 0.15) / 20)                         # 0.15 dB margin for codec inter-sample overs
    gain_db = TARGET - lufs(binaural)
    for _ in range(6):                                             # limiter lowers loudness: iterate make-up gain
        y, g = limit(binaural * 10 ** (gain_db / 20), ceil)
        err = TARGET - lufs(y)
        if abs(err) < 0.05:
            break
        gain_db += err
    y714, g714 = limit(m714 * 10 ** (gain_db / 20), ceil)
    return y, y714, gain_db, g, g714


if __name__ == '__main__':
    W = sys.argv[1]
    b, _ = sf.read(os.path.join(W, 'mix_binaural_raw.wav'), dtype='float32')
    m, _ = sf.read(os.path.join(W, 'mix_714_raw.wav'), dtype='float32')
    print(f'in : binaural {lufs(b):.1f} LUFS, TP {20*np.log10(true_peak(b)):+.1f} dBTP')
    yb, ym, gdb, g, g7 = master(b, m)
    print(f'gain {gdb:+.1f} dB; limiter max GR binaural {-20*np.log10(g.min()):.1f} dB, 7.1.4 {-20*np.log10(g7.min()):.1f} dB')
    print(f'out: binaural {lufs(yb):.2f} LUFS, TP {20*np.log10(true_peak(yb)):+.2f} dBTP')
    print(f'     7.1.4    {lufs(ym, W714):.2f} LUFS(1770 weighted), TP {20*np.log10(true_peak(ym)):+.2f} dBTP')
    sf.write(os.path.join(W, 'trailer_binaural.wav'), yb, SR, subtype='PCM_24')
    sf.write(os.path.join(W, 'trailer_714.wav'), ym, SR, subtype='PCM_24')
    print('written')
