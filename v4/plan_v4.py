"""v4 edit plan: v3's beat-snapped cuts, adjusted for the inserts and the longer ending."""
import json
from plan_v3 import O2N, TRANS as TRANS3, FLASH, COLD, ROAR, DIVE, CHORUS, DROP, REACH, END, FPS, section_intensity
from conform_v4 import RIDE, INSERTS

N = json.load(open('audio_v4.json'))['frames']
CUTS = sorted((set(TRANS3) - {O2N[770]}) | {RIDE})          # 770 merged into the c19 insert; ride-away cut added
TRANS = {c: TRANS3.get(c, 'punch') for c in CUTS}
TRANS[RIDE] = 'punch'
ACAPPELLA = (4.05, 5.45)                                      # voice-only bridge: calm the picture
