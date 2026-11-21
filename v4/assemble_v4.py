import subprocess, numpy as np
FF='/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
W,H=1080,1920
portal=np.load('portal_clean_v4.npy')
dec=subprocess.Popen([FF,'-loglevel','error','-f','concat','-safe','0','-i','parts_v4/list.txt','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen([FF,'-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','30','-i','-',
    '-i','TheDayIStoppedBeingScaredOfFalling_audio_-14LUFS.m4a','-map','0:v','-map','1:a','-c:v','libx264','-preset','slow',
    '-crf','14','-pix_fmt','yuv420p','-c:a','copy','-shortest','-movflags','+faststart','clean_master_v4.mp4'],stdin=subprocess.PIPE)
n=0
while True:
    b=dec.stdout.read(W*H*3)
    if len(b)<W*H*3: break
    if 152<=n<182: b=portal[n-152].tobytes()
    enc.stdin.write(b); n+=1
enc.stdin.close(); enc.wait(); print('frames',n)
