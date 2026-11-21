#!/usr/bin/env bash
# Rebuild TheDayIStoppedBeingScaredOfFalling_improved.mp4 from the original.
# Needs ffmpeg and tesseract. Usage: scripts/process.sh video.mp4
set -euo pipefail
IN=${1:-video.mp4}
OUT=TheDayIStoppedBeingScaredOfFalling_improved.mp4

# 1. Measure loudness (first loudnorm pass).
J=$(ffmpeg -hide_banner -i "$IN" -vn -af loudnorm=I=-14:TP=-1:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
g() { echo "$J" | python3 -c "import json,sys;print(json.load(sys.stdin)['$1'])"; }

# 2. Linear gain to -14 LUFS (dynamics preserved); video stream copied untouched.
ffmpeg -hide_banner -y -i "$IN" -map 0:v -map 0:a -c:v copy \
  -af "loudnorm=I=-14:TP=-1:LRA=11:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true" \
  -ar 48000 -c:a aac -b:a 256k -map_metadata 0 -movflags +faststart "$OUT"

# 3. OCR the burned-in captions (bottom band, 10 fps) with Tesseract.
mkdir -p frames
ffmpeg -hide_banner -loglevel error -y -i "$IN" -vf "fps=10,crop=1080:560:0:1200,format=gray" frames/f%04d.png
python3 "$(dirname "$0")/ocr_captions.py"   # writes ocr.json: [frame, text, mean_conf, min_conf]
