#!/usr/bin/env bash
set -euo pipefail
IN=kinetic_v4_master.mp4
# preview under 30 MiB for chat (HEVC)
P="-c:v libx265 -preset slow -b:v 3100k -pix_fmt yuv420p -tag:v hvc1"
ffmpeg -hide_banner -loglevel error -y -i $IN $P -x265-params pass=1:log-level=error:aq-mode=3 -an -f null /dev/null
ffmpeg -hide_banner -loglevel error -y -i $IN $P -x265-params pass=2:log-level=error:aq-mode=3 -c:a copy -movflags +faststart TheDayIStoppedBeingScaredOfFalling_v4_preview.mp4
# upload master for TikTok / Instagram (H.264, ~10 Mbps)
U="-c:v libx264 -preset slow -b:v 10000k -maxrate 14000k -bufsize 20000k -profile:v high -level 4.2 -pix_fmt yuv420p -g 60 -bf 2"
ffmpeg -hide_banner -loglevel error -y -i $IN $U -pass 1 -passlogfile x264v4 -an -f null /dev/null
ffmpeg -hide_banner -loglevel error -y -i $IN $U -pass 2 -passlogfile x264v4 -c:a copy -movflags +faststart TheDayIStoppedBeingScaredOfFalling_v4_UPLOAD_MASTER.mp4
echo ENCODED
