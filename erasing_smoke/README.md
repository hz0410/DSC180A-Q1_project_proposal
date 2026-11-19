# Erase and repaint: three-image diffusion smoke test

This project illustrates **input → person mask → erased-region visualization → diffusion inpainting → background-preserving composite**. It includes three generated scenes (brick wall, beach, park), each with a drawn person, a dilated removal mask, and the original empty background for evaluation. It is not a real-person photo benchmark. No training or paid API is required.

## Run

From this directory, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export HF_HOME="$PWD/.cache/huggingface"
python make_dataset.py --output data
python -m unittest -v test_smoke.py
python run_smoke.py --mode tiny --device cpu --output outputs/tiny
```

Tiny mode downloads a small internal-testing checkpoint and runs 4 denoising steps at 64×64. Its images look like noise; passing it means the software path works, not that removal works well. The tiny checkpoint's test safety checker uses 30px inputs; the script matches its processor to that configuration.

For meaningful visual inspection, use the full pretrained SD 1.5 inpainting checkpoint:

```bash
python run_smoke.py --mode quality --output outputs/quality
```

Quality defaults: 3 images, 512×512, 30 DDIM steps, CFG 7.5, seeds 7–9. Device selection is CUDA, then Apple MPS, then CPU; CPU is slower. The first full-model run downloads several GB. For the smaller CPU demonstration:

```bash
python run_smoke.py --mode quality --device cpu --size 256 --steps 16 --output outputs/quality
```

A checkpoint override is available through `--model` and an optional Hub commit through `--revision`. Hardware and package versions are recorded in `config.json`. `requirements-tested.txt` records the local environment; it includes the macOS CPU PyTorch build, so CUDA users should install an appropriate PyTorch build separately. Seeds do not guarantee identical results across hardware.

## What to inspect

- `outputs/quality/contact_sheet.png`: all three cases, side by side.
- `outputs/quality/00/process.png` (and `01`, `02`): input, mask, erased visualization, raw generated image, composited output, clean reference.
- `outputs/quality/00/trajectory.png`: decoded noisy latent snapshots during denoising. They are not predicted clean images.
- `outputs/quality/metrics.json`: runtime, mask coverage, raw outside-mask error, final outside-mask error, edit magnitude, and reference error within the mask.

**Mask convention:** white (255) means repaint; black (0) means preserve. The white erased image is only a teaching visualization. The model receives the input image plus its mask; it never receives the clean reference. Inpainting erases and fills jointly, rather than first detecting an actual empty background.

The script checks that masked pixels change and final unmasked pixels remain identical. These are smoke checks, not semantic removal scores. Final compositing enforces preservation and may introduce seams. Inspect images for coherent backgrounds, missing person fragments, and new objects. Tiny mode is not useful for semantic quality assessment. Three images are too few for FID.

## Three real pedestrian photos (Penn-Fudan)

The optional real-image demonstration uses three annotated photos from the [official Penn-Fudan dataset](https://www.cis.upenn.edu/~jshi/ped_html/). The preparer downloads the roughly 51 MB archive, selects the first three sorted photos, combines all labeled pedestrian masks, dilates the masks by 15 pixels, and edge-pads each image to a square. The archive hash and source paths are recorded. These masks need visual inspection: unlabeled people, shadows, and reflections can remain. There is no clean ground-truth background.

```bash
python prepare_pennfudan.py --count 3
python run_smoke.py --mode quality --manifest data/pennfudan/manifest.json --size 256 --steps 16 --output outputs/pennfudan
```

See `outputs/pennfudan/contact_sheet.png` and each case's `trajectory.png`. The full downloaded archive and checkpoints are excluded from the shareable ZIP; preparation can download them again. Dataset attribution and usage notices remain with its official source. `requests` is included in the requirements for verified HTTPS downloads.

## Your own photos

Create a manifest with paths relative to the manifest itself:

```json
[
  {
    "id": "photo_1",
    "image": "photos/input.png",
    "mask": "photos/person_mask.png",
    "prompt": "an empty sidewalk, continuous pavement and brick wall, no people"
  }
]
```

Then run:

```bash
python run_smoke.py --mode quality --manifest my_data/manifest.json --output outputs/my_photos
```

Image and mask dimensions must match. Cover the entire selected person and any shadows you intend to remove. The script resizes both to a square; use square crops to avoid distortion. Masks are supplied, not detected automatically. A `reference` field is optional; omit it for real photos where the hidden background is unknown. The generated background is plausible content, not recovery of the true hidden scene.

## Connection to the papers

See `PAPER_NOTES.md` for readings of Attentive Eraser, RePaint, and PowerPaint and explicit implementation boundaries. This baseline uses the standard Diffusers inpainting pipeline. It does not reproduce the papers' specialized algorithms. The notebook `erase_repaint_demo.ipynb` provides the same commands and displays results.

Pipeline API: https://huggingface.co/docs/diffusers/v0.35.1/api/pipelines/stable_diffusion/inpaint

## Shared Conda environment troubleshooting

A native abort was reproduced in the user's base Python 3.13 environment while optional Keras dependencies imported pyarrow. With the unused TensorFlow path disabled, a second incompatible optional-kernels dependency surfaced. The exact underlying native-library incompatibility was not established. The runner now disables unused TensorFlow/Flax backends before importing the diffusion stack. The notebook prefers the validated sibling `.venv-inpainting/bin/python` when present, leaving the shared Conda installation unchanged. On another machine, create a clean Python 3.12 environment using the setup instructions and choose that notebook kernel. A notebook may use Python 3.13 for display while its `run` helper launches the isolated Python 3.12 process.
