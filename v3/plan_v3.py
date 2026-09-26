"""Shared v3 edit plan: cut list after beat-snapping and the transition/SFX at each cut."""
import json
rt = json.load(open('retime.json'))
O2N = dict(zip(rt['old'], rt['new']))
CUTS = rt['new'][1:-1]
N = 1987
COLD, ROAR, DIVE, CHORUS, DROP, REACH, END = (O2N[c] for c in (28, 52, 168, 1065, 1416, 1686, 1911))
FPS = 30.0


def section_intensity(t):
    import numpy as np
    pts = [(0, .9), (1.6, .9), (1.9, .35), (20.4, .35), (20.9, .6), (35.3, .6), (35.6, .8),
           (47.0, .8), (47.2, 1.0), (60.6, 1.0), (61.2, .4), (63.6, .4), (63.7, 0.0), (99, 0.0)]
    xs, ys = zip(*pts)
    return float(np.interp(t, xs, ys))


TRANS = {}
for n, c in enumerate(CUTS):
    I = section_intensity(c / FPS)
    if c == COLD: TRANS[c] = 'whip_down'
    elif c == DIVE: TRANS[c] = 'dive'
    elif c in (CHORUS, O2N[1251]): TRANS[c] = 'burn'          # chorus start, "the bronze cracked"
    elif c == DROP: TRANS[c] = 'drop'
    elif c == END: TRANS[c] = 'end'
    elif c in (ROAR, REACH): TRANS[c] = 'zoom'
    elif I < 0.55: TRANS[c] = 'punch'
    else: TRANS[c] = ['whip_down', 'zoom', 'whip_side', 'whip_down', 'burn', 'zoom'][n % 6]
FLASH = {ROAR: 0.55, CHORUS: 0.5, DROP: 0.9, REACH: 0.7}
