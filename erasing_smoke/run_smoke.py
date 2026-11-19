"""Runnable diffusion inpainting baseline; not a reproduction of the three papers."""
# This is a PyTorch-only pipeline. Avoid importing optional TensorFlow/Keras
# installations, which can load incompatible native libraries in shared Conda envs.
import os
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def load_case(record, root, size):
    image = Image.open(root / record['image']).convert('RGB')
    mask = Image.open(root / record['mask']).convert('L')
    if image.size != mask.size:
        raise ValueError('Image and mask must have identical dimensions')
    image = image.resize((size,size), Image.Resampling.LANCZOS)
    mask = mask.resize((size,size), Image.Resampling.NEAREST).point(lambda x:255 if x>=128 else 0)
    fraction = (np.asarray(mask)>0).mean()
    if not 0 < fraction < 1:
        raise ValueError('Mask must include both edit and preserved pixels')
    return image, mask


def preserve_background(original, generated, mask):
    if original.size != generated.size or mask.size != original.size:
        raise ValueError('Compositing requires matching dimensions')
    return Image.composite(generated.convert('RGB'), original, mask)


def metrics(original, raw, final, mask, reference=None):
    a,b,c = [np.asarray(im).astype(np.float64) for im in (original,raw,final)]
    edit=np.asarray(mask)>0
    out=dict(mask_fraction=float(edit.mean()),
             raw_outside_mae=float(np.abs(a-b)[~edit].mean()),
             final_outside_max_error=float(np.abs(a-c)[~edit].max()),
             edited_region_change_mae=float(np.abs(a-c)[edit].mean()))
    if reference is not None:
        ref=np.asarray(reference).astype(np.float64)
        mse=float(((c-ref)[edit]**2).mean())
        out['masked_reference_mae']=float(np.abs(c-ref)[edit].mean())
        out['masked_reference_psnr_db']=float(10*np.log10(255**2/mse)) if mse else None
    return out


def panel(images, labels, path, tile=192):
    canvas=Image.new('RGB',(tile*len(images),tile+36),'white')
    d=ImageDraw.Draw(canvas)
    for j,(im,label) in enumerate(zip(images,labels)):
        canvas.paste(im.convert('RGB').resize((tile,tile)),(j*tile,36))
        d.text((j*tile+8,10),label,fill='black')
    canvas.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/manifest.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/tiny'))
    p.add_argument('--mode',choices=['tiny','quality'],default='tiny')
    p.add_argument('--model',default=None)
    p.add_argument('--revision',default=None)
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    p.add_argument('--steps',type=int,default=None)
    p.add_argument('--size',type=int,default=None)
    p.add_argument('--limit',type=int,default=3)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--guidance',type=float,default=7.5)
    a=p.parse_args()
    import torch
    import diffusers
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    size=a.size or (64 if a.mode=='tiny' else 512)
    steps=a.steps or (4 if a.mode=='tiny' else 30)
    if size<32 or size%8 or steps<1 or a.limit<1:
        p.error('size must be >=32 and divisible by 8; steps and limit must be positive')
    device=a.device
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    dtype=torch.float16 if device=='cuda' else torch.float32
    model=a.model or ('hf-internal-testing/tiny-stable-diffusion-pipe' if a.mode=='tiny'
                     else 'stable-diffusion-v1-5/stable-diffusion-inpainting')
    a.output.mkdir(parents=True,exist_ok=True)
    config=dict(mode=a.mode,model=model,requested_revision=a.revision,device=device,size=size,
                steps=steps,seed=a.seed,guidance=a.guidance,torch=torch.__version__,diffusers=diffusers.__version__,
                semantic_quality_expected=a.mode=='quality',
                manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest())
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('Loading',model,'on',device,flush=True)
    pipe=StableDiffusionInpaintPipeline.from_pretrained(model,revision=a.revision,torch_dtype=dtype)
    from huggingface_hub import try_to_load_from_cache
    cached_index = try_to_load_from_cache(model, 'model_index.json', revision=a.revision or 'main')
    if isinstance(cached_index, str):
        config['resolved_checkpoint_commit'] = Path(cached_index).parent.name
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    # The tiny fixture ships a 30px CLIP checker with a 224px processor.
    # Match processor dimensions to that checker rather than disabling it.
    if a.mode == 'tiny' and pipe.safety_checker is not None:
        clip_size = pipe.safety_checker.config.vision_config.image_size
        pipe.feature_extractor.size = {'shortest_edge': clip_size}
        pipe.feature_extractor.crop_size = {'height': clip_size, 'width': clip_size}
    # Keep the checkpoint's safety checker if provided.
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe=pipe.to(device)
    pipe.enable_attention_slicing()
    (a.output/'scheduler_config.json').write_text(pipe.scheduler.to_json_string())
    records=json.loads(a.manifest.read_text())[:a.limit]
    if not records: raise ValueError('Empty dataset')
    report=[]; contact=[]
    for index,record in enumerate(records):
        case=a.output/f'{index:02d}'; case.mkdir(exist_ok=True)
        image,mask=load_case(record,a.manifest.parent,size)
        image.save(case/'input.png'); mask.save(case/'mask.png')
        erased=Image.composite(Image.new('RGB',image.size,'white'),image,mask)
        erased.save(case/'erased_visualization.png')
        snapshots=[]
        def callback(pipeline,step,timestep,kwargs):
            # Decode a few current noisy latents, not predicted clean images.
            if step in {0,steps//2,steps-1}:
                with torch.no_grad():
                    z=kwargs['latents']/pipeline.vae.config.scaling_factor
                    decoded=pipeline.vae.decode(z.to(dtype=pipeline.vae.dtype),return_dict=False)[0]
                    im=pipeline.image_processor.postprocess(decoded,output_type='pil')[0]
                im.save(case/f'latent_step_{step:03d}.png')
                snapshots.append((step,im))
            return kwargs
        start=time.perf_counter()
        with torch.inference_mode():
            result=pipe(prompt=record['prompt'],negative_prompt='person, people, human, silhouette, body, artifacts',
                        image=image,mask_image=mask,height=size,width=size,strength=1.0,
                        num_inference_steps=steps,guidance_scale=a.guidance,
                        generator=torch.Generator(device='cpu').manual_seed(a.seed+index),
                        callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents'])
        if result.nsfw_content_detected and any(result.nsfw_content_detected):
            raise RuntimeError('Safety checker flagged the output; inspect the input and prompt')
        raw=result.images[0].convert('RGB')
        final=preserve_background(image,raw,mask)
        reference=None
        if record.get('reference'):
            reference=Image.open(a.manifest.parent/record['reference']).convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
        row=dict(id=record['id'],seconds=time.perf_counter()-start,prompt=record['prompt'],
                 **metrics(image,raw,final,mask,reference))
        if row['final_outside_max_error']!=0: raise AssertionError('Background changed after compositing')
        if row['edited_region_change_mae']<=0: raise AssertionError('Edit did not change any masked pixels')
        raw.save(case/'raw_generated.png'); final.save(case/'repainted.png')
        ims=[image,mask,erased,raw,final]; labels=['Input','Mask (white=edit)','Erased (display only)','Raw diffusion output','Repainted + preserved']
        if reference is not None: ims.append(reference); labels.append('Known clean background')
        panel(ims,labels,case/'process.png')
        if snapshots:
            panel([v for _,v in snapshots],[f'Noisy latent step {k+1}' for k,_ in snapshots],case/'trajectory.png')
        contact.append(Image.open(case/'process.png').copy()); report.append(row)
        (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(row,flush=True)
    board=Image.new('RGB',(contact[0].width,sum(im.height for im in contact)),'white')
    y=0
    for im in contact: board.paste(im,(0,y)); y+=im.height
    board.save(a.output/'contact_sheet.png')
    print('PASS: generated',len(report),'cases; unmasked pixels preserved exactly.',flush=True)

if __name__=='__main__': main()
