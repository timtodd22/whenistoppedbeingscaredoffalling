#!/usr/bin/env bash
# Render v3 in 4 parallel video-only parts, then join and mux the v3 audio once.
set -euo pipefail
FF=/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
B=(0 500 1000 1500 1987)
for i in 0 1 2 3; do
  CV_THREADS=1 python3 render_v3.py ${B[$i]} ${B[$((i+1))]} v3parts/p$i.mp4 16 --no-audio 2> v3parts/log$i.txt &
done
wait
printf "file 'p0.mp4'\nfile 'p1.mp4'\nfile 'p2.mp4'\nfile 'p3.mp4'\n" > v3parts/list.txt
$FF -loglevel error -y -f concat -safe 0 -i v3parts/list.txt -i audio_v3.m4a -map 0:v -map 1:a -c copy -shortest -movflags +faststart kinetic_v3_master.mp4
