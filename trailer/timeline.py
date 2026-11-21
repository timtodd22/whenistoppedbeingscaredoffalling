"""Shared edit decision list for THE LONG RAIN (61 s, 24 fps). Both the picture edit and the
spatial mix read this, so every sound lands on its cut.

Each shot: (start_s, end_s, kind, source, source_in_s, note)
  kind 'blender' -> rendered frames in renders/<source>/ (on twos/threes, flow-interpolated to 24 fps)
  kind 'ai'      -> AI video file ai/<source>.mp4, starting at source_in_s
  kind 'card'    -> title card text
  kind 'black'   -> black
"""
FPS = 24
TOTAL = 61.0

SHOTS = [
    (0.0, 4.0, 'black', None, 0, 'cold open: rain, drone, L1'),
    (4.0, 10.0, 'blender', 'shot01', 0.0, 'THE SEA WALL (night), lightning at 7.3'),
    (10.0, 14.0, 'ai', 'A02_window', 0.6, 'Iris at the window, L2'),
    (14.0, 19.0, 'blender', 'shot03', 0.0, 'THE ORANGE DUST'),
    (19.0, 23.0, 'ai', 'A04_interrogation', 1.0, 'Tobin, interrogation, L3'),
    (23.0, 27.0, 'blender', 'city_descent', 0.0, 'DESCENT, spinner overtakes'),
    (27.0, 30.0, 'ai', 'A06_eye', 0.0, 'THE EYE: the city in the pupil, one blink on the heartbeat'),
    (30.0, 32.0, 'card', 'IN 2099', 0, 'hit'),
    (32.0, 35.0, 'blender', 'city_aerial', 0.0, 'MEGACITY, L4'),
    (35.0, 37.0, 'card', 'THE LAST OF HER KIND', 0, 'hit'),
    (37.0, 40.0, 'ai', 'A10_hand', 1.4, 'hand on glass, L5'),
    (40.0, 42.0, 'blender', 'city_hologram', 0.0, 'THE HOLOGRAM'),
    (42.0, 44.0, 'card', 'IS STILL BREATHING', 0, 'hit'),
    # montage 44-47: accelerating cuts
    (44.0, 44.75, 'ai', 'A13_rain', 1.5, 'Tobin in the rain'),
    (44.75, 45.4, 'blender', 'city_descent', 2.6, 'spinner'),
    (45.4, 45.95, 'blender', 'shot01', 3.2, 'the wall (flash)'),
    (45.95, 46.4, 'ai', 'A02_window', 3.6, 'Iris'),
    (46.4, 46.75, 'blender', 'city_hologram', 1.2, 'hologram'),
    (46.75, 47.0, 'ai', 'A13_rain', 3.8, 'Tobin'),
    (47.0, 48.0, 'black', None, 0, 'absolute silence'),
    (48.0, 52.6, 'ai', 'A15_whisper', 0.5, 'Iris leans in, L6 (right ear)'),
    (52.6, 56.6, 'title', 'THE LONG RAIN', 0, 'the hit lands as the whisper ends'),
    (56.6, 61.0, 'blender', 'shot17', 0.0, 'THE SEA WALL (dawn), L7'),
]

# dialogue: (file, take, start_s, trim_in_s, trim_out_s or None)
DIALOGUE = {
    'L1': ('L12_a.mp3', 0.9, 0.0, 'split1'),     # "They built us to last four years."
    'L2': ('L12_a.mp3', 11.0, 'split1', None),   # "I've had seventy."
    'L3': ('L3_a.mp3', 20.3, 0.0, None),
    'L4': ('L4_a.mp3', 32.6, 0.0, None),
    'L5': ('L5_a.mp3', 37.3, 0.0, None),
    'L6': ('L6_a.mp3', 48.2, 0.0, None),
    'L7': ('L7_a.mp3', 56.75, 0.0, None),
}

LIGHTNING = 7.3          # seconds: flash in picture, thunder crack in sound
TITLE_HIT = 52.6
DAWN = 56.6
SILENCE = (47.0, 48.0)
