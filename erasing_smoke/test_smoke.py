"""Fast checks of mask direction, data integrity, and background preservation."""
import tempfile
import unittest
import json
from pathlib import Path
import numpy as np
from PIL import Image
from make_dataset import make_dataset
from run_smoke import load_case, preserve_background, metrics

class SmokeTests(unittest.TestCase):
    def test_dataset_covers_all_changed_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=make_dataset(folder)
            rows=json.loads(manifest.read_text())
            self.assertEqual(len(rows),3)
            for row in rows:
                image,mask=load_case(row,manifest.parent,256)
                clean=np.asarray(Image.open(manifest.parent/row['reference']))
                changed=np.any(np.asarray(image)!=clean,axis=-1)
                self.assertTrue(np.all(np.asarray(mask)[changed]==255))
                self.assertGreater(changed.sum(),0)

    def test_composite_only_changes_white_mask(self):
        src=Image.new('RGB',(8,8),(10,20,30))
        raw=Image.new('RGB',(8,8),(200,210,220))
        m=np.zeros((8,8),dtype=np.uint8); m[2:6,2:6]=255
        mask=Image.fromarray(m)
        final=preserve_background(src,raw,mask)
        self.assertTrue(np.array_equal(np.asarray(final)[m==0],np.asarray(src)[m==0]))
        self.assertTrue(np.array_equal(np.asarray(final)[m==255],np.asarray(raw)[m==255]))
        result=metrics(src,raw,final,mask)
        self.assertEqual(result['final_outside_max_error'],0)
        self.assertGreater(result['raw_outside_mae'],0)

    def test_rejects_empty_and_full_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); Image.new('RGB',(32,32)).save(root/'image.png')
            for value in [0,255]:
                Image.new('L',(32,32),value).save(root/'mask.png')
                with self.assertRaises(ValueError):
                    load_case(dict(image='image.png',mask='mask.png'),root,32)

if __name__=='__main__': unittest.main()
