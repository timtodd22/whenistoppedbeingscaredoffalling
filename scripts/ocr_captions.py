import subprocess, glob, json, os
from multiprocessing import Pool
from PIL import Image, ImageOps
os.environ['OMP_THREAD_LIMIT'] = '1'
def ocr(p):
    q = p.replace('.png', '_bw.png')
    ImageOps.invert(Image.open(p).point(lambda v: 255 if v > 200 else 0)).save(q)
    tsv = subprocess.run(['tesseract', q, '-', '--psm', '6', 'tsv'], capture_output=True, text=True, timeout=30).stdout
    words, confs = [], []
    for l in tsv.splitlines()[1:]:
        c = l.split('\t')
        if len(c) > 11 and c[11].strip() and float(c[10]) >= 0:
            words.append(c[11]); confs.append(float(c[10]))
    return p, ' '.join(words), (sum(confs) / len(confs) if confs else 0), min(confs, default=0)
fs = sorted(f for f in glob.glob('frames/f*.png') if '_bw' not in f)
with Pool(4) as pool: res = pool.map(ocr, fs, chunksize=8)
json.dump(res, open('ocr.json', 'w'))
