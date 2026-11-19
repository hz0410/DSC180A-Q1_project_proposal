# Smoke-test results

Executed locally on CPU. These are pipeline checks, not a benchmark or a reproduction of the three papers. Full-model runs used 256×256 images, 16 DDIM steps, CFG 7.5, and seeds 7–9. The CPU runs briefly overlapped, so timings are not a controlled speed comparison.

## Completed checks

- Three unit tests passed: mask coverage, correct mask direction/background compositing, rejection of empty/full masks.
- Tiny diffusion ran on three synthetic cases (4 steps, 64×64); output quality is intentionally not meaningful.
- Full SD 1.5 inpainting ran on three synthetic cases and three Penn-Fudan pedestrian photos.
- All nine generated cases changed masked pixels and had zero final outside-mask error relative to the resized input. This is enforced by compositing.

## Full-model measurements

| Dataset | Image | Raw outside-mask MAE | Final outside-mask max error | Masked reference MAE |
| --- | --- | ---: | ---: | ---: |
| quality | brick_wall | 8.31 | 0 | 8.61 |
| quality | beach | 3.82 | 0 | 4.53 |
| quality | park | 3.05 | 0 | 11.58 |
| pennfudan | FudanPed00001 | 11.06 | 0 | No reference |
| pennfudan | FudanPed00002 | 11.96 | 0 | No reference |
| pennfudan | FudanPed00003 | 9.44 | 0 | No reference |

Errors use 0–255 RGB pixel units. Reconstruction error is available only for the controlled synthetic scenes and does not prove perceptual or removal quality.

## Visual inspection

- Synthetic wall: brickwork is repainted, with some visible mask-boundary texture differences.
- Synthetic beach: background colors are restored, with a faint residual mask outline.
- Synthetic park: a new small object is generated at the end of the path; this is a semantic failure despite successful execution.
- FudanPed00001: the two foreground people are removed; shadows and small pavement artifacts remain.
- FudanPed00002: a person remains/is regenerated. The supplied mask does not cover every visible person and accessory.
- FudanPed00003: the model regenerates a person. Some visible accessories lie outside the annotated mask and remain available as conditioning cues.

All cases are retained. This illustrates why removal-specific guidance and careful masks matter. A next experiment should refine masks (including accessories and shadows), then compare the official paper methods on identical inputs and seeds.

## Reproducibility

See each output directory's config.json, scheduler_config.json, metrics.json, saved inputs/masks, raw images, composites, and snapshots. Checkpoint commits and package versions are recorded; requirements-tested.txt records installed packages. The notebook is validated but was not executed end-to-end as a notebook; its underlying scripts were executed.
