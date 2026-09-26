"""Word-level lyric timing: Whisper word timestamps on the song, force-matched to the known lyrics."""
import json, re, difflib, subprocess, numpy as np, whisper
FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
pcm = subprocess.run([FF, '-loglevel', 'error', '-i', 'TheDayIStoppedBeingScaredOfFalling_audio_-14LUFS.m4a', '-ac', '1', '-ar', '16000',
                      '-f', 's16le', '-'], capture_output=True).stdout
audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
from lyrics import LINES
model = whisper.load_model('small.en')
prompt = ' '.join(l[2].replace('|', ' ') for l in LINES).lower()
r = model.transcribe(audio, language='en', word_timestamps=True, initial_prompt=prompt,
                     condition_on_previous_text=False, temperature=0)
heard = [(re.sub(r"[^a-z']", '', w['word'].lower()), w['start'], w['end'])
         for s in r['segments'] for w in s['words']]
json.dump({'text': r['text'], 'heard': heard}, open('whisper_raw.json', 'w'), indent=0)
print(r['text'])
