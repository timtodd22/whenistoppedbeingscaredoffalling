"""Forced alignment of the known lyrics with torchaudio's MMS_FA (CTC) model.
Runs on the full mix and on a vocal-emphasised (HPSS harmonic, 150-5000 Hz) copy."""
import json, re, subprocess
import numpy as np, torch, torchaudio, librosa
from lyrics_v3 import LINES
FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
pcm = subprocess.run([FF, '-loglevel', 'error', '-i', 'TheDayIStoppedBeingScaredOfFalling_audio_-14LUFS.m4a',
                      '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], capture_output=True).stdout
mix = np.frombuffer(pcm, np.float32).copy()
harm, _ = librosa.effects.hpss(mix, margin=2.0)
voc = torchaudio.functional.bandpass_biquad(torch.from_numpy(harm), 16000, 900, Q=0.35).numpy()
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).eval()
tok, aligner = bundle.get_tokenizer(), bundle.get_aligner()
words = [w for L in LINES for w in L[2].replace('|', ' ').split(' ')]
clean = [re.sub(r"[^a-z']", '', w.lower()) for w in words]
res = {}
for name, sig in (('mix', mix), ('vocal', voc)):
    wav = torch.from_numpy(sig / (np.abs(sig).max() + 1e-9)).float()[None]
    with torch.inference_mode():
        em, _ = model(wav)
    spans = aligner(em[0], tok(clean))
    ratio = wav.shape[1] / em.shape[1] / 16000
    res[name] = [(s[0].start * ratio, s[-1].end * ratio) for s in spans]
json.dump({'words': words, **res}, open('fa_times.json', 'w'))
for i, w in enumerate(words):
    a, b = res['mix'][i][0], res['vocal'][i][0]
    print(f"{w:12s} mix {a:6.2f}  vocal {b:6.2f}  {'*' if abs(a - b) > 0.25 else ''}")
