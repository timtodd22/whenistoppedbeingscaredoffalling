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

## v3 showpiece (`v3/`)

The full pipeline, in order: `align.py` → `conform.py` → `sfx.py` → `run_v3.sh`. `render_v3.py` and `lyrics_v3.py` do the per-frame work.

- **Words appear as they're sung.** Word times come from forced alignment of the known lyrics with torchaudio's MMS_FA model (`fa_align.py`, `fa_times.json`). It agrees with Whisper (`whisper_raw.json`) within 0.4 s on 85 of 99 words, and Whisper is used only where the aligner bunches two words together. Whisper also found an uncaptioned outro line, "She caught us both, that's what wings are for". It's added over the final shot and as the end-title tagline.
- **Cuts land on the beat.** 30 of 43 cuts now fall exactly on a beat, up from 4. Shots are resampled to their new lengths with optical-flow interpolation (`retime.json`), and total length and audio sync are unchanged.
- **Cold open.** An unused source shot of the girl falling through the sky (c01, frames 72–96) replaces the dark first second, color-matched to the edit.
- **Speed ramps.** Fast-slow-fast optical-flow ramps on the hero shots.
- **Letterbox.** Bars shrink section by section, snap open at the drop (47.2 s), then close again for the outro.
- **Look.** Per-section split-tone grades, halation, anamorphic highlight streaks, film grain, chromatic hits and a vignette.
- **Transitions.** Whips, zoom-blur hits, the dive, procedural film burns, and the drop slam with a flash.
- **Camera hits on key words.** "NO, NO, NO", "GO", "FALLING", "FALL", "BURNED", "BRONZE".
- **Sound design.** A different synthesized bass hit at the dive, the chorus, the drop and the reach, plus panned whooshes on the whips. No risers. Everything is mixed under the song and renormalized to -14 LUFS.
- **Animated end title.** Cinzel letters close in and a light sweep crosses them. No ember trails.

## v4 (`v4/`) and release files (`release/`, Git LFS)

- `release/TheDayIStoppedBeingScaredOfFalling_v4_UPLOAD_MASTER.mp4`: v4 for TikTok and Instagram. H.264 High, 1080×1920, 30 fps, about 10 Mbps, 69.9 s, AAC at -14 LUFS.
- `release/TheDayIStoppedBeingScaredOfFalling_UPLOAD_MASTER.mp4`: the same format for v3.

**Pipeline, in order:**
1. Demucs stems of the full song, then `audio_v4.py` → `fa_align_v4.py` → `sfx_v4.py`.
2. `upscale_src.py` → `portal_v4.py` → `clean_v4.py` → `assemble_v4.py`.
3. `conform_v4.py` → `depth_v4.py` → `run_v4.sh` → `encode_v4.sh`.

**What changed from v3:**
- **Audio rebuilt from the full song.** The original 66 s edit is three splices of the full song, at 5.35 s, 46.35 s and 59.6 s. v4 rebuilds that edit from Demucs stems using the measured splice map.
- **A cappella bridge (4.05–5.45 s).** The first splice doesn't fall on the beat, and the original covered it with a sound effect, so "purpose" was never heard. In v4 the voice sings "…on purpose" alone, and the picture cools and calms for that moment.
- **Filtered build into the drop.** The instrumental closes over two bars before 47.2 s and bursts open with the letterbox; the vocal stays clear.
- **Natural ending.** The song's own continuation rings out and fades on the downbeat at 69.95 s, instead of cutting 0.2 s after the last word.
- **Lyrics.** Words are force-aligned on the isolated vocal and swell with the held notes. With Depth Anything V2 occlusion, subjects can pass in front of words, but floor and water planes are ignored and no word is ever more than 40% covered.
- **Unused source shots** replace the repeated ones at 23.5–26 s and 33.6–35.5 s, and a ride-away shot is added after "she caught us both".
- **Real-ESRGAN (realesr-general-x4v3)** upscales all 480p-sourced material: the cold open, the inserts, the portal replacement and the caption areas.
