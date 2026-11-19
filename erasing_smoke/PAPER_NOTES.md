# Paper reading notes and implementation boundaries

Read the authors' arXiv HTML versions because the PDF reader rejected the large PDF downloads. This smoke test is a Stable Diffusion inpainting baseline, not an implementation or numerical reproduction of the three methods.

## Attentive Eraser — arXiv:2412.12974

Source: https://arxiv.org/html/2412.12974v2

AAS suppresses foreground keys in self-attention so generated foreground regions attend to background information. Early similarity suppression scales attention similarities to reduce copying of similar objects. SARG combines ordinary and modified noise predictions as `eps + s * (eps_AAS - eps)`. The paper evaluates stochastic and deterministic inpainting pipelines and finds that global image metrics alone do not adequately measure removal. Limitations include selecting suitable masks and handling object-related effects.

Our baseline saves raw output and separately preserves original unmasked pixels. It does **not** implement AAS, similarity suppression, SARG, or DDIM inversion. A faithful next step would use the authors' attention processors and compare them against this baseline with matched masks and seeds.

Official code: https://github.com/Alibaba-VELLDEPTH/AttentiveEraser

## RePaint — arXiv:2201.09865

Source: https://arxiv.org/html/2201.09865v4

RePaint uses an unconditional DDPM without mask-specific retraining. At each reverse step it combines model-generated unknown pixels with known input pixels sampled at the corresponding noise level. Forward noise jumps followed by reverse denoising give the two regions additional opportunities to harmonize. The paper evaluates mask generalization and diversity, with limitations for semantic plausibility and difficult structures.

Our test exposes white edit masks and preserves known pixels at the end. Modern Diffusers handles the denoising path. This is **not** RePaint's jump schedule, original unconditional DDPM, or an equivalent implementation of its pixel-space sampling. Decoded snapshots are noisy latent states, not RePaint clean-image predictions.

Official code: https://github.com/andreas128/RePaint

## PowerPaint — arXiv:2312.03594

Source: https://arxiv.org/html/2312.03594v4

PowerPaint fine-tunes a Stable Diffusion inpainting network and learned task prompts for context filling, object generation, and shape guidance. Removal uses context filling as the positive condition and object generation as the negative condition in classifier-free guidance. The learned embeddings are essential: ordinary English prompts are not interchangeable with these task tokens. The paper evaluates multiple editing tasks and discusses unwanted object synthesis in crowded scenes.

Our baseline uses ordinary background descriptions and a negative prompt naming people. This is **not** PowerPaint, its trained task tokens, or its fine-tuned checkpoint. A next comparison should load the official model and task embeddings rather than merely rename a prompt.

Official project: https://powerpaint.github.io/

## Experiment interpretation

Three synthetic scenes are an engineering fixture, not a representative removal benchmark. The clean background is available only for reporting pixel reconstruction errors; it is never passed to the diffusion model. No training occurs. Tiny mode verifies execution only. Full-model output still requires human inspection for remaining people, seams, invented objects, and implausible structure. Do not report FID on these three examples. Zero outside-mask error is guaranteed by final compositing, not evidence that diffusion itself preserves every unmasked pixel. Raw-output errors are also saved to make that distinction visible.
