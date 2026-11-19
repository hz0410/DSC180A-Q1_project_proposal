"""Three controlled scenes with a drawn person and a known empty background."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_dataset(root, size=256):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i, name in enumerate(['brick_wall', 'beach', 'park']):
        rng = np.random.default_rng(100 + i)
        yy, xx = np.mgrid[:size, :size]
        noise = rng.normal(0, 2, (size, size, 1))
        if i == 0:
            rgb = np.empty((size, size, 3)); rgb[:] = [165, 116, 88]
            mortar = (yy % 32 < 3) | ((xx + (yy // 32 % 2) * 32) % 64 < 3)
            rgb[mortar] = [210, 201, 183]
            prompt = 'an empty brick wall, continuous brickwork, no people'
        elif i == 1:
            rgb = np.empty((size, size, 3)); rgb[:] = [125, 192, 224]
            rgb[yy > size * .42] = [65, 156, 182]
            rgb[yy > size * .64] = [218, 198, 154]
            prompt = 'an empty beach, blue sea and sand, no people'
        else:
            rgb = np.empty((size, size, 3)); rgb[:] = [148, 197, 221]
            rgb[yy > size * .4] = [88, 135, 67]
            path = (yy > size * .45) & (np.abs(xx - size * .5) < (yy - size * .35) * .38)
            rgb[path] = [194, 180, 155]
            prompt = 'an empty park path surrounded by grass, no people'
        clean = Image.fromarray(np.uint8(np.clip(rgb + noise, 0, 255)))
        object_mask = Image.new('L', (size, size), 0)
        d = ImageDraw.Draw(object_mask)
        # Coordinates at 256px; this is a synthetic person, not a photograph.
        def box(values): return tuple(round(v * size / 256) for v in values)
        d.ellipse(box((112, 60, 144, 92)), fill=255)
        d.rounded_rectangle(box((106, 93, 150, 160)), radius=max(1,size//32), fill=255)
        for points in [[(108,100),(91,145),(101,149),(120,113)],
                       [(147,100),(165,144),(155,150),(137,113)],
                       [(110,154),(107,214),(120,214),(130,156)],
                       [(131,154),(139,214),(152,214),(146,154)]]:
            d.polygon([(round(x*size/256),round(y*size/256)) for x,y in points], fill=255)
        input_image = Image.composite(Image.new('RGB', clean.size, (44, 49, 75)), clean, object_mask)
        # Dilation includes person boundaries. White=edit; black=preserve.
        edit_mask = object_mask.filter(ImageFilter.MaxFilter(9))
        folder = root / name; folder.mkdir(exist_ok=True)
        for filename, image in [('input.png',input_image),('mask.png',edit_mask),('clean.png',clean)]:
            image.save(folder/filename)
        records.append(dict(id=name, image=f'{name}/input.png', mask=f'{name}/mask.png',
                            reference=f'{name}/clean.png', prompt=prompt))
    (root/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    return root/'manifest.json'

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--output',default='data'); p.add_argument('--size',type=int,default=256)
    a=p.parse_args(); print(make_dataset(a.output,a.size))
