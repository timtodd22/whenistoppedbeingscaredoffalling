"""v4 audio, rebuilt from Demucs stems of the full song.

1. Reconstruct the video's audio edit from the full-song stems using the
   measured splice map (and null-test it against the original edit).
2. Filtered build: the instrumental (drums+bass+other) closes to a muffled
   low-pass over the two bars before the drop and bursts open on the cut;
   the vocal stays clear on top.
3. Natural ending: instead of cutting 0.2 s after the last word, the song's
   own continuation plays on under the end title and fades on a downbeat.
4. Distinct bass hits from v3b, then loudness to -14 LUFS / -1 dBTP.
Writes audio_v4.m4a, stems_v4.npz (aligned vocal/instrumental) and
audio_v4.json (new total length).
"""
import json, subprocess, sys
import numpy as np, librosa, soundfile as sf

SR = 44100
# video time -> full-song time: (video start, full-song offset)
SPLICES = [(0.0, 9.445), (5.35, 26.645), (46.35, 32.845), (59.6, 41.779)]
VIDEO_LEN = 1987 / 30
XF = 0.03                                    # 30 ms equal-power crossfade at each splice
# The first splice is 35.27 beats long (not whole), so a direct join stumbles; the original edit hid it
# under a portal sound effect and a pause. v4 lets "...on purpose" play, then leaves a 0.15 s breath.
BREATH = {1: (4.05, 5.45)}                   # splice index -> (previous segment fades out by, next starts at)
# In the gap the voice continues a cappella: the held "on..." (song 4.05-5.30) compressed to 0.70 s,
# then "purpose" (song 5.30-6.05) at natural speed, ending at 5.50 as the next section enters.
ACAPPELLA = dict(start=4.05, hold=(4.05, 5.30), hold_to=0.70, word=(5.30, 6.05))


def stem(name):
    y, sr = sf.read(f'stems/htdemucs/fullsong/{name}.wav', dtype='float32')
    assert sr == SR
    return y


def assemble(full, total):
    """Cut the video's edit out of a full-song signal, extending the last segment to `total` seconds."""
    out = np.zeros((int(total * SR), full.shape[1]), np.float32)
    bounds = [s for s, _ in SPLICES] + [total]
    for k, (v0, off) in enumerate(SPLICES):
        v1 = bounds[k + 1]
        if k in BREATH:
            v0 = BREATH[k][1]
        if k + 1 in BREATH:
            v1 = BREATH[k + 1][0]
        a = max(0, int((v0 - XF / 2) * SR)); b = min(len(out), int((v1 + XF / 2) * SR))
        seg = full[a + int(off * SR): b + int(off * SR)]
        seg = np.pad(seg, ((0, b - a - len(seg)), (0, 0)))
        w = np.ones(b - a, np.float32); n = int(XF * SR)
        if k > 0:
            w[:n] = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
        if k < len(SPLICES) - 1:
            nf = int(0.10 * SR) if k + 1 in BREATH else n
            w[-nf:] = np.cos(np.linspace(0, np.pi / 2, nf)) ** 2
        out[a:b] += seg * w[:, None]
    return out


def acappella(full_voc):
    """The voice-only bridge placed into the first splice gap (video time)."""
    off = SPLICES[0][1]
    grab = lambda t0, t1: full_voc[int((t0 + off) * SR):int((t1 + off) * SR)].copy()
    h0, h1 = ACAPPELLA['hold']
    hold = np.stack([librosa.effects.time_stretch(np.ascontiguousarray(grab(h0, h1)[:, c]),
                                                  rate=(h1 - h0) / ACAPPELLA['hold_to']) for c in range(2)], 1)
    word = grab(*ACAPPELLA['word'])
    n = int(0.02 * SR)
    hold[-n:] *= np.linspace(1, 0, n)[:, None]; word[:n] *= np.linspace(0, 1, n)[:, None]
    seg = np.concatenate([hold[:-n], hold[-n:] + word[:n], word[n:]])
    seg[-int(0.04 * SR):] *= np.linspace(1, 0, int(0.04 * SR))[:, None]
    return seg.astype(np.float32)


def lowpass_sweep(x, t0, t1, f_hi=16000.0, f_lo=320.0, order=4):
    """STFT low-pass whose cutoff falls exponentially from f_hi to f_lo between t0 and t1, open elsewhere."""
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        S = librosa.stft(x[:, ch], n_fft=2048, hop_length=256)
        fr = librosa.fft_frequencies(sr=SR, n_fft=2048)[:, None]
        tt = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=256)[None, :]
        u = np.clip((tt - t0) / (t1 - t0), 0, 1)
        fc = f_hi * (f_lo / f_hi) ** (u ** 1.3)
        g = 1 / np.sqrt(1 + (fr / fc) ** (2 * order))
        g = np.where((tt >= t0) & (tt < t1), g, 1.0)
        # a touch of level dip as it closes, so the burst reads bigger
        g *= np.where((tt >= t0) & (tt < t1), 10 ** (-3.5 * u / 20), 1.0)
        out[:, ch] = librosa.istft(S * g, hop_length=256, length=len(x))
    return out


def main():
    names = ['vocals', 'drums', 'bass', 'other']
    full = {n: stem(n) for n in names}
    # null test: rebuild the original 66.2 s edit and compare it with the video's audio in 1 s windows
    rebuilt = assemble(sum(full.values()), VIDEO_LEN).mean(1)
    ref = np.frombuffer(subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', 'video.mp4', '-ac', '1', '-ar', str(SR),
                                        '-f', 'f32le', '-'], capture_output=True).stdout, np.float32)
    worst = 1.0
    for t in [1] + list(range(9, 65, 4)):                     # 4-7 s held the original's portal SFX
        x = ref[t * SR:(t + 1) * SR]
        best = max(np.dot(x, rebuilt[t * SR + l:(t + 1) * SR + l]) /
                   (np.linalg.norm(x) * np.linalg.norm(rebuilt[t * SR + l:(t + 1) * SR + l]) + 1e-9)
                   for l in range(-40, 41))
        worst = min(worst, best)
    print(f'rebuild vs original edit: worst 1 s window correlation {worst:.3f}')

    # ending: let the song continue past the last word to a downbeat, then fade
    ext_to = float(sys.argv[1]) if len(sys.argv) > 1 else 70.0
    fade = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    voc = assemble(full['vocals'], ext_to)
    ac = acappella(full['vocals'])
    a0 = int(ACAPPELLA['start'] * SR)
    voc[a0:a0 + len(ac)] += ac                              # the assembled vocal is silent in the gap
    print(f'a cappella bridge {ACAPPELLA["start"]:.2f}-{ACAPPELLA["start"] + len(ac) / SR:.2f} s')
    inst = assemble(full['drums'] + full['bass'] + full['other'], ext_to)
    drop = 1416 / 30
    bar2 = 8 * 60 / 123.05                                     # two bars at 123 BPM
    inst = lowpass_sweep(inst, drop - bar2, drop - 0.01)
    mix = voc + inst
    f0 = int((ext_to - fade) * SR)
    env = np.ones(len(mix), np.float32); env[f0:] = np.cos(np.linspace(0, np.pi / 2, len(mix) - f0)) ** 2
    mix *= env[:, None]; voc *= env[:, None]; inst *= env[:, None]
    np.savez_compressed('stems_v4.npz', vocals=voc[:, 0] + voc[:, 1], inst=inst[:, 0] + inst[:, 1], sr=SR)
    sf.write('audio_v4_raw.wav', mix, SR, subtype='FLOAT')
    json.dump({'length_s': ext_to, 'frames': int(round(ext_to * 30))}, open('audio_v4.json', 'w'))
    print('wrote', ext_to, 's')


if __name__ == '__main__':
    main()
