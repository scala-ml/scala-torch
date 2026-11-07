"""Independent SDXL Diffusers model/scheduler oracle with CFG rescaling."""
import json
from pathlib import Path
import sys
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg

root, output = map(Path, sys.argv[1:3])
torch.set_num_threads(2)
unet = UNet2DConditionModel.from_pretrained(root / 'unet', local_files_only=True).eval()
vae = AutoencoderKL.from_pretrained(root / 'vae', local_files_only=True).eval()
refs = Path('models/src/test/resources/tiny-sdxl-prompt')
def read(name, shape):
    return torch.frombuffer(bytearray((refs / (name + '.bin')).read_bytes()), dtype=torch.float32).reshape(shape)
positive = read('plain-positive', (1, 77, 64)); negative = read('plain-negative', (1, 77, 64))
pooled = read('plain-pooled', (1, 32)); negative_pooled = read('plain-negative-pooled', (1, 32))
noise = torch.cos(torch.arange(64, dtype=torch.float32) * .17).reshape(1, 4, 4, 4)
source = torch.sin(torch.arange(192, dtype=torch.float32) * .07).reshape(1, 3, 8, 8)
posterior_noise = torch.sin(torch.arange(64, dtype=torch.float32) * .11).reshape(1, 4, 4, 4)
cases = {}
with torch.no_grad():
    for dpm in (False, True):
        for img in (False, True):
            ids = torch.tensor([[12.,16.,1.,2.,8.,8.]] if img else [[8.,8.,0.,0.,8.,8.]])
            scheduler = DPMSolverMultistepScheduler(algorithm_type='dpmsolver++',solver_order=2) if dpm else EulerDiscreteScheduler()
            scheduler.set_timesteps(2); scheduler.timesteps=torch.tensor([500.,250.]);scheduler.sigmas=torch.tensor([1.,.5,0.])
            start=1 if img else 0; scheduler.set_begin_index(start); latent=noise.clone()
            if img:
                posterior=vae.encode(source).latent_dist
                clean=(posterior.mean+posterior.std*posterior_noise)*vae.config.scaling_factor
                alpha=1/torch.sqrt(1+scheduler.sigmas[start]**2) if dpm else 1
                latent=clean*alpha+noise*(scheduler.sigmas[start]*alpha)
            for i in range(start,2):
                t=scheduler.timesteps[i]; sample=scheduler.scale_model_input(latent,t)
                pos=unet(sample,t,encoder_hidden_states=positive,added_cond_kwargs=dict(text_embeds=pooled,time_ids=ids)).sample
                neg=unet(sample,t,encoder_hidden_states=negative,added_cond_kwargs=dict(text_embeds=negative_pooled,time_ids=ids)).sample
                guided=rescale_noise_cfg(neg+7.5*(pos-neg),pos,guidance_rescale=.7)
                latent=scheduler.step(guided,t,latent).prev_sample
            cases[('dpm_' if dpm else 'euler_')+('img2img' if img else 'txt2img')]=latent.flatten().tolist()
output.write_text(json.dumps(cases),encoding='utf-8')
