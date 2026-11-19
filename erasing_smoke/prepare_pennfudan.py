"""Prepare three real pedestrian photos and masks from the official archive."""
import argparse, hashlib, io, json, urllib.request
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import requests
from PIL import Image, ImageFilter
URL='https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/pennfudan'))
    p.add_argument('--archive',type=Path,default=Path('.cache/PennFudanPed.zip'))
    p.add_argument('--count',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.count<=170: p.error('count must be between 1 and 170')
    a.archive.parent.mkdir(parents=True,exist_ok=True); a.output.mkdir(parents=True,exist_ok=True)
    if not a.archive.exists():
        part=a.archive.with_suffix('.zip.part')
        with requests.get(URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with part.open('wb') as file:
                for chunk in response.iter_content(1024*1024): file.write(chunk)
        part.replace(a.archive)
    rows=[]
    with ZipFile(a.archive) as z:
        files=sorted(n for n in z.namelist() if '/PNGImages/' in n and n.endswith('.png'))[:a.count]
        for name in files:
            stem=Path(name).stem
            mask_name=name.replace('/PNGImages/','/PedMasks/').replace('.png','_mask.png')
            image=Image.open(io.BytesIO(z.read(name))).convert('RGB')
            ids=np.asarray(Image.open(io.BytesIO(z.read(mask_name))))
            mask=Image.fromarray(np.uint8(ids>0)*255).filter(ImageFilter.MaxFilter(15))
            arr=np.asarray(image); h,w=arr.shape[:2]; side=max(h,w)
            top=(side-h)//2; left=(side-w)//2
            arr=np.pad(arr,((top,side-h-top),(left,side-w-left),(0,0)),mode='edge')
            m=np.pad(np.asarray(mask),((top,side-h-top),(left,side-w-left)),mode='constant')
            folder=a.output/stem; folder.mkdir(exist_ok=True)
            Image.fromarray(arr).save(folder/'input.png'); Image.fromarray(m).save(folder/'mask.png')
            rows.append(dict(id=stem,image=f'{stem}/input.png',mask=f'{stem}/mask.png',
                prompt='an empty outdoor campus walkway, continuous pavement, trees and buildings, no people',
                source=URL,source_member=name))
    if not rows: raise ValueError('No dataset images found')
    (a.output/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.output/'provenance.json').write_text(json.dumps(dict(url=URL,archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),
        note='First sorted photos; union of labeled people; 15px dilation; square edge padding; no clean reference.'),indent=2)+'\n')
    print('Prepared',len(rows),'real photos:',a.output/'manifest.json')
if __name__=='__main__': main()
