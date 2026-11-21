"""Original score for THE LONG RAIN, synthesized. Returns stems placed on the 61 s timeline.

Palette: analogue-style polysynth (detuned saws through a time-varying low-pass),
CS-80-style brass (filter swell, pitch scoop, delayed vibrato), low brass trailer hits
with sub drops, a heartbeat drop-out, a Shepard riser into silence, and a lonely lead
for the ending. All melodies and chords are original.
"""
import numpy as np
from scipy.signal import butter, sosfilt
import librosa
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from timeline import TITLE_HIT as TH, DAWN

SR = 48000
TOTAL = 61.0
N = int(TOTAL * SR)
rng = np.random.default_rng(7)


def hz(note):
    return librosa.note_to_hz(note)


def saw(freq, n, phase=0.0):
    """PolyBLEP band-limited saw; freq may be an array (for bends/vibrato)."""
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    dt = f / SR
    ph = (phase + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m = ph < dt; tt = ph[m] / dt[m]; y[m] -= tt + tt - tt * tt - 1
    m = ph > 1 - dt; tt = (ph[m] - 1) / dt[m]; y[m] -= tt * tt + tt + tt + 1
    return y.astype(np.float32)


def stft_lowpass(x, cutoff, q=1.0, order=3):
    """Time-varying low-pass: cutoff(t) array (Hz per sample) applied in the STFT domain."""
    S = librosa.stft(x, n_fft=2048, hop_length=256)
    fr = librosa.fft_frequencies(sr=SR, n_fft=2048)[:, None]
    fc = cutoff[::256][:S.shape[1]]
    fc = np.pad(fc, (0, S.shape[1] - len(fc)), mode='edge')[None, :]
    g = 1 / np.sqrt(1 + (fr / fc) ** (2 * order))
    # a gentle resonance bump just under the cutoff, like an analogue ladder filter
    g *= 1 + (q - 1) * np.exp(-((fr - fc * 0.9) / (fc * 0.15)) ** 2)
    return librosa.istft(S * g, hop_length=256, length=len(x)).astype(np.float32)


def place(dst, sig, start):
    s0 = int(start * SR); e = min(len(dst), s0 + len(sig))
    if e > s0:
        dst[s0:e] += sig[:e - s0]


def adsr(n, a, d, s, r):
    e = np.full(n, s, np.float32)
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na); e[na:na + nd] = np.linspace(1, s, nd)[:max(0, min(nd, n - na))]
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def pad_chord(notes, dur, bright=1200, seed=0, voices=5, detune_c=14):
    n = int(dur * SR); r = np.random.default_rng(seed); x = np.zeros(n, np.float32)
    for nt in notes:
        for v in range(voices):
            c = (v - (voices - 1) / 2) / ((voices - 1) / 2) * detune_c
            x += saw(hz(nt) * 2 ** (c / 1200), n, r.random()) / voices
    t = np.arange(n) / SR
    cut = bright * (0.7 + 0.3 * np.sin(2 * np.pi * t / 9.0 + seed)) + 200
    return stft_lowpass(x, cut, q=1.4) * adsr(n, 1.8, 0.5, 0.85, 2.2) * 0.22


def brass(note, dur, peak=3500, seed=0):
    """CS-80-style brass note: scoop up into pitch, filter swell, vibrato after 0.6 s."""
    n = int(dur * SR); t = np.arange(n) / SR
    bend = -0.45 * np.exp(-t / 0.07)
    vib = 0.14 * np.clip((t - 0.6) / 0.8, 0, 1) * np.sin(2 * np.pi * 5.4 * t)
    f = hz(note) * 2 ** ((bend + vib) / 12)
    x = saw(f, n) + 0.6 * saw(f * 1.004, n, 0.3) + 0.4 * np.sign(saw(f * 0.5, n))
    cut = 350 + peak * (1 - np.exp(-t / 0.25)) * np.exp(-np.clip(t - 0.4, 0, None) / 2.5) + 600
    return stft_lowpass(x.astype(np.float32), cut, q=1.8) * adsr(n, 0.08, 0.4, 0.75, 0.9) * 0.2


def hit(dur=4.0, size=1.0, seed=0):
    """Trailer hit: distorted low brass cluster + sub drop + metallic impact."""
    n = int(dur * SR); t = np.arange(n) / SR; r = np.random.default_rng(seed)
    low = sum(saw(hz(nt) * (1 + 0.003 * k), n, r.random()) for k, nt in enumerate(['C1', 'C2', 'G2', 'C3', 'Eb3']))
    low = np.tanh(low * 1.8 * size) * np.exp(-t / (1.6 * size))
    low = stft_lowpass(low.astype(np.float32), 300 + 2200 * np.exp(-t / 0.35), q=1.5)
    sub = np.sin(2 * np.pi * np.cumsum(28 + 30 * np.exp(-t / 0.25)) / SR) * np.exp(-t / (1.8 * size))
    imp = r.standard_normal(n).astype(np.float32) * np.exp(-t / 0.05)
    metal = np.zeros(n, np.float32)
    for f in (211, 347, 523, 787, 1163):                       # inharmonic modes of a struck metal plate
        metal += np.sin(2 * np.pi * f * t + r.random() * 6) * np.exp(-t / r.uniform(0.6, 1.4))
    metal = metal * 0.08 + sosfilt(butter(2, [800, 6000], 'band', fs=SR, output='sos'), imp).astype(np.float32) * 0.5
    return {'low': (low * 0.55).astype(np.float32), 'sub': (sub * 0.9 * size).astype(np.float32), 'metal': metal.astype(np.float32)}


def reverse_swell(dur=1.2, seed=0):
    n = int(dur * SR); t = np.arange(n) / SR
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    x = sosfilt(butter(2, 3500, 'high', fs=SR, output='sos'), x).astype(np.float32)
    return x * (t / dur) ** 3 * 0.35


def shepard(dur=3.0, base=55.0, octaves=7, rise=1.0):
    n = int(dur * SR); t = np.arange(n) / SR; x = np.zeros(n)
    for k in range(octaves):
        pos = (k + rise * t / dur) % octaves
        f = base * 2 ** pos
        amp = np.exp(-((pos - octaves / 2) / (octaves / 4)) ** 2)
        x += np.sin(2 * np.pi * np.cumsum(f) / SR) * amp
    return (x / octaves * 0.6 * (t / dur) ** 1.5).astype(np.float32)


def lead(notes, start_gap=0.0):
    """Lonely lead: soft saw + sine, slow attack, wide vibrato. notes = [(note, dur)]."""
    out = []
    for nt, d in notes:
        n = int(d * SR); t = np.arange(n) / SR
        vib = 0.2 * np.clip((t - 0.4) / 1.0, 0, 1) * np.sin(2 * np.pi * 5.0 * t)
        f = hz(nt) * 2 ** (vib / 12)
        x = 0.6 * np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.25 * saw(f, n)
        x = stft_lowpass(x.astype(np.float32), np.full(n, 2400.0), q=1.2)
        out.append(x * adsr(n, 0.35, 0.3, 0.8, 0.8) * 0.18)
    return np.concatenate(out)


def build():
    st = {k: np.zeros(N, np.float32) for k in
          ('drone', 'padL', 'padR', 'brass', 'hit_low', 'hit_sub', 'hit_metal', 'swell', 'riser', 'pulse', 'lead', 'glass')}
    t = np.arange(N) / SR
    # drone: two detuned low sines + filtered noise, alive across 0-47 s, gone in the silence, back under the whisper
    dr = (np.sin(2 * np.pi * 32.7 * t) + 0.7 * np.sin(2 * np.pi * 32.9 * t) + 0.4 * np.sin(2 * np.pi * 65.5 * t)) * 0.12
    dr *= np.clip(t / 4.0, 0, 1) * (t < 47.0) + (0.5 * ((t >= 48.0) & (t < TH)))
    st['drone'] += dr.astype(np.float32)
    # harmony: C minor -> Ab -> Eb/G -> Bb -> Cm (wide) ...
    chords = [(4.0, 6.5, ['C3', 'G3', 'Eb4']), (10.0, 4.5, ['Ab2', 'Eb3', 'C4']), (14.0, 5.5, ['Eb3', 'G3', 'Bb3']),
              (19.0, 4.5, ['F2', 'C3', 'Ab3']), (23.0, 4.2, ['G2', 'D3', 'Bb3', 'F4']),
              (30.0, 5.5, ['C2', 'G2', 'Eb3', 'G3', 'C4']), (35.0, 5.5, ['Ab1', 'Ab2', 'Eb3', 'C4']),
              (40.0, 4.5, ['Bb1', 'F2', 'D3', 'F3']), (44.0, 3.0, ['C2', 'G2', 'Eb3', 'Bb3']),
              (TH, 4.0, ['C1', 'C2', 'G2', 'Eb3']), (DAWN, 4.4, ['Ab2', 'Eb3', 'G3', 'C4'])]
    for i, (s0, d, notes) in enumerate(chords):
        bright = 900 if s0 < 30 else 1800
        place(st['padL'], pad_chord(notes, d + 1.5, bright, seed=i * 2), s0)
        place(st['padR'], pad_chord(notes, d + 1.5, bright, seed=i * 2 + 1), s0)
    # brass swell through the descent, and the motif over the megacity
    for s0, nt, d in ((23.6, 'G3', 3.4), (24.4, 'D4', 2.6), (32.1, 'C4', 1.1), (33.2, 'Eb4', 0.9), (34.1, 'G4', 1.8),
                      (40.9, 'F4', 1.3)):
        place(st['brass'], brass(nt, d, seed=int(s0)), s0)
    # hits on the title cards and the title
    for s0, size in ((30.0, 0.9), (35.0, 0.95), (42.0, 1.0), (TH, 1.6)):
        h = hit(4.5 if size < 1.5 else 6.0, size, seed=int(s0))
        place(st['hit_low'], h['low'], s0); place(st['hit_sub'], h['sub'], s0); place(st['hit_metal'], h['metal'], s0)
        place(st['swell'], reverse_swell(1.0 if size < 1.5 else 1.6, seed=int(s0)), s0 - (1.0 if size < 1.5 else 1.6))
    # montage: pulsing low eighths accelerating 44-47, Shepard riser, hard stop at 47.0
    tt = 44.0; step = 0.3
    while tt < 46.95:
        n = int(0.18 * SR); x = saw(hz('C2'), n) * np.exp(-np.arange(n) / SR * 14)
        place(st['pulse'], (np.tanh(x * 2) * 0.25).astype(np.float32), tt); tt += step; step = max(0.11, step * 0.9)
    place(st['riser'], shepard(3.0), 44.0)
    # glassy high note on the window shot
    n = int(4.0 * SR); tw = np.arange(n) / SR
    place(st['glass'], (np.sin(2 * np.pi * hz('G5') * tw) * adsr(n, 1.0, 0.2, 0.7, 1.5) * 0.04).astype(np.float32), 10.2)
    # the ending: lonely lead over the dawn sea wall, a final low tone
    # the lead sings over the title, leaving the last line to the voice alone
    place(st['lead'], lead([('G4', 1.5), ('Eb4', 1.0), ('F4', 1.7)]), TH + 0.6)
    n = int(2.5 * SR); tl = np.arange(n) / SR
    place(st['drone'], (np.sin(2 * np.pi * 32.7 * tl) * np.exp(-tl / 1.2) * 0.25).astype(np.float32), DAWN + 2.0)
    # the silence at 47-48 is absolute
    for k in st:
        st[k][int(47.0 * SR):int(48.0 * SR)] = 0
    return st
