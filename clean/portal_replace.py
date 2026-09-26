"""Replace the fire-portal transition (frames 152-181) with raw source:
dragon (c01) continuing, hard cut on the 5.60 s beat (frame 168) to the
bridge (c02), timed so c02 lands on the frame the edit resumes from."""
import subprocess, numpy as np, cv2
FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
W, H, F0, F1, CUT = 1080, 1920, 152, 182, 168

def src(c):
    b = subprocess.run([FF, '-loglevel', 'error', '-i', f'src/{c}.mp4', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    return np.frombuffer(b, np.uint8).reshape(-1, 854, 480, 3)
def grab(g):
    b = subprocess.run([FF, '-loglevel', 'error', '-ss', f'{g/30:.4f}', '-i', 'video.mp4', '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    return np.frombuffer(b, np.uint8).reshape(H, W, 3)
def up(x):
    u = cv2.resize(x, (W, H), interpolation=cv2.INTER_LANCZOS4)
    return cv2.addWeighted(u, 1.6, cv2.GaussianBlur(u, (0, 0), 1.2), -0.6, 0)
sel = np.ones((H, W), bool); sel[1450:1740] = False
def coef(s, ref):
    return [np.linalg.lstsq(np.vstack([s[..., c][sel], np.ones(sel.sum())]).T.astype(np.float32),
                            ref[..., c][sel].astype(np.float32), rcond=None)[0] for c in range(3)]
def apply(img, cf):
    o = img.astype(np.float32)
    for c in range(3): o[..., c] = o[..., c] * cf[c][0] + cf[c][1]
    return np.clip(o, 0, 255).astype(np.uint8)
def at(S, p):
    i0 = int(np.floor(p)); i1 = min(len(S) - 1, i0 + 1); f = p - i0
    return S[i0] if f < 0.05 else cv2.addWeighted(S[i0], 1 - f, S[i1], f, 0)
A, B = src('c01'), src('c02')
cA = coef(up(A[63]), grab(151)); cB = coef(up(B[25]), grab(183))
out = []
for g in range(F0, F1):
    if g < CUT: out.append(apply(up(at(A, 64 + (g - 152) * 7 / 15)), cA))   # dragon shot ends at c01 frame 71
    else:       out.append(apply(up(at(B, 17 + (g - 168) * 7 / 13)), cB))  # bridge shot starts at c02 frame 17
np.save('portal_clean.npy', np.array(out))
S = [150, 151, 152, 160, 167, 168, 175, 181, 182, 183]
ref = {g: grab(g) for g in (150, 151, 182, 183)}
t = [(ref[g] if g in ref else out[g - F0])[::5, ::5] for g in S]
cv2.imwrite('portal.jpg', cv2.cvtColor(np.concatenate(t, 1), cv2.COLOR_RGB2BGR))
