# whenistoppedbeingscaredoffalling

Post-processing for *The Day I Stopped Being Scared of Falling* (1080×1920, 30 fps, 66 s).

- `captions.srt` / `captions.vtt`: closed captions for all 13 lyric lines. They were read off the burned-in text with Tesseract OCR and checked by eye; timings are accurate to about 0.1 s. Upload them as a subtitle track on YouTube/Facebook, or as the SRT on TikTok/Instagram, for accessibility and search.
- `scripts/process.sh`: rebuilds the improved MP4 (audio normalized to -14 LUFS / -1 dBTP with linear gain and faststart; video stream copied untouched) and reruns the OCR pass.

The MP4 itself (about 246 MB) is too large to store in git.
