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

## Caption-free source (`clean/`)

This rebuilds the edit without the burned-in captions. The raw Seedance clips are listed in `source_clips.json`; they're 480p and were generated without text.

1. `match.json` records which source clip and frame each edited frame came from, matched at 0.99 correlation.
2. `clean.py` replaces only the caption box in each frame. The fill comes from the matching source frame (best nearby frame or blend of two), upscaled and color-matched to the edit, with a feathered edge. Everything else is your original 1080p edit, untouched. Where highlights are blown out, it switches to histogram color matching.
3. `portal_replace.py` swaps the fire-portal transition (5.07–6.07 s), where the captions were baked into the flames, for raw source footage: the dragon shot continuing, then a hard cut to the bridge on the 5.60 s beat.
4. `assemble.py` joins the parts and adds the -14 LUFS audio. A full Tesseract pass over the result finds no caption text.

## Kinetic v2 (`kinetic/render_v2.py`, `kinetic/lyrics.py`)

v2 renders from the caption-free master and adds:

- a new "dive" transition at 5.60 s: an accelerating zoom-blur push into the dragon, an ember glow flash, a pull-out onto the bridge, and upward-streaking embers;
- animated lyrics: lines start at the original caption times, words pop in with a stagger, and one key word per line glows orange. The hook lines are set large in Cinzel to match the end card. Lyrics sit above TikTok's bottom UI and clear of the right-hand button rail.

Fonts are from @fontsource (OFL): barlow-condensed 800 and cinzel 900, as `.woff` in `fonts/`.
