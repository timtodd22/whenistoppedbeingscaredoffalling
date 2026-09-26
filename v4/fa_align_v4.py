"""v4 word timing: MMS_FA forced alignment on the isolated vocal stem (Demucs), which is far cleaner
than the full mix. Also writes a 30 fps vocal-loudness envelope for voice-reactive typography."""
import json, re
import numpy as np, torch, torchaudio
from lyrics_v3 import LINES
d = np.load('stems_v4.npz'); voc = d['vocals'].astype(np.float32); sr = int(d['sr'])
v16 = torchaudio.functional.resample(torch.from_numpy(voc), sr, 16000)
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).eval()
tok, aligner = bundle.get_tokenizer(), bundle.get_aligner()
words = [w for L in LINES for w in L[2].replace('|', ' ').split(' ')]
clean = [re.sub(r"[^a-z']", '', w.lower()) for w in words]
wav = (v16 / (v16.abs().max() + 1e-9))[None]
with torch.inference_mode():
    em, _ = model(wav)
spans = aligner(em[0], tok(clean))
ratio = wav.shape[1] / em.shape[1] / 16000
times = [(s[0].start * ratio, s[-1].end * ratio) for s in spans]
old = json.load(open('fa_times.json'))['mix']
json.dump({'words': words, 'mix': times}, open('fa_times_v4.json', 'w'))
diffs = [abs(a[0] - b[0]) for a, b in zip(times, old)]
print(f'vs full-mix alignment: median shift {np.median(diffs):.3f}s, >0.3s on {sum(x > 0.3 for x in diffs)} words')
for w, (a, _), (b, _) in zip(words, times, old):
    if abs(a - b) > 0.3:
        print(f'  {w:10s} stem {a:6.2f}  mix {b:6.2f}')
# vocal loudness envelope, 30 fps, 0..1
hop = sr // 30
rms = np.sqrt(np.convolve(voc ** 2, np.ones(hop) / hop, 'same')[::hop])
db = 20 * np.log10(rms + 1e-6)
env = np.clip((db - np.percentile(db, 40)) / (np.percentile(db, 97) - np.percentile(db, 40)), 0, 1)
np.save('vocal_env.npy', env.astype(np.float32))
