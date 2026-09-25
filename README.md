# whenistoppedbeingscaredoffalling

Post-processing for *The Day I Stopped Being Scared of Falling* (1080×1920, 30 fps, 66 s).

- `captions.srt` / `captions.vtt`: closed captions for all 13 lyric lines. They were read off the burned-in text with Tesseract OCR and checked by eye; timings are accurate to about 0.1 s. Upload them as a subtitle track on YouTube/Facebook, or as the SRT on TikTok/Instagram, for accessibility and search.
- `scripts/process.sh`: rebuilds the improved MP4 (audio normalized to -14 LUFS / -1 dBTP with linear gain and faststart; video stream copied untouched) and reruns the OCR pass.

The MP4 itself (about 246 MB) is too large to store in git.

## Kinetic TikTok edit (`kinetic/`)

`kinetic/render.py` re-renders the video with beat-synced motion effects: 123 BPM, with beats in `beats.json`, detected with librosa. The effects are zoom pulses on the beat, punch-ins on cuts, downward whip and zoom-blur transitions, white flashes on the big moments, camera shake and roll, speed ramps, chromatic aberration, highlight glow, and a contrast/saturation grade with a vignette. How strong the effects are depends on the section: hook, verse, build, chorus, then the drop at 47.2 s.

Every effect keeps each shot's timing, so the burned-in lyrics stay in sync with the song. Speed ramps are only applied to shots where the caption doesn't change.

    # needs: ffmpeg, python3 with numpy, opencv-python-headless
    # inputs in the working dir: video.mp4 (original) and the -14 LUFS audio .m4a
    python3 kinetic/render.py 0 1987 kinetic_master.mp4 16     # frame range, output, x264 CRF
