"""Clean-condition flow-matching ODE sampler (inference-side patch).

Replaces the DDPM unscale loop selected by translate_mgm.py with
--sample_fn ddpm --classifier_scale 0. The model is the same MultimodalUNet;
its video output is a velocity in the FM target space (scaled by
FM_TARGET_SCALE). We integrate

    du/dt = v_theta(u_t, t, clean_audio, label),  u_0 ~ N(0, I),  t: 0 -> 1

with Euler steps, then divide by FM_TARGET_SCALE to return to the dataset video
space (z_ko / rna_std), exactly like the DDPM samples.

The condition is kept CLEAN at every step, matching flow_objective_patch.py.
"""
import os

import torch as th

from scdiffusionX.DiffusionBackbone import dist_util
from scdiffusionX.DiffusionBackbone.multimodal_gaussian_diffusion import GaussianDiffusion


def _unwrap_model(model):
    """Return the raw UNet if `model` is SpacedDiffusion's _WrappedModel.

    _WrappedModel indexes `timestep_map` with integer timesteps and cannot accept
    continuous t.  Production translation passes the raw UNet; this keeps the patch
    safe if a wrapped model is supplied.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(model, "timestep_map"):
        if getattr(model, "rescale_timesteps", False):
            raise ValueError("flow_sampler_patch assumes rescale_timesteps=False")
        return inner
    return model


def _flow_sample(self, model, shape, use_fp16, noise=None, clip_denoised=True,
                 denoised_fn=None, cond_fn=None, model_kwargs=None, device=None,
                 progress=False, class_scale=0.0):
    """Euler ODE sampler; API-compatible with the DDPM unscale loop."""
    if device is None:
        device = dist_util.dev()
    model = _unwrap_model(model)

    model_kwargs = dict(model_kwargs or {})
    audio = model_kwargs.pop("audio", None)
    if audio is None:
        raise ValueError("flow_sampler_patch requires model_kwargs['audio'] "
                         "(the clean taxonomy condition); gen_mode must be 'rna'")
    if "video" in model_kwargs:
        raise ValueError("flow_sampler_patch only supports gen_mode='rna' "
                         "(no audio generation)")

    if noise is not None:
        video = noise["video"].to(device)
    else:
        video = th.randn(*shape["video"], device=device)

    steps = int(os.environ.get("FM_STEPS", "1000"))
    scale = float(os.environ.get("FM_TARGET_SCALE", "10"))
    dt = 1.0 / steps

    indices = range(steps)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    for i in indices:
        t = th.full((video.shape[0],), i * dt, device=device)
        with th.no_grad():
            v, _ = model(video, audio, t * (self.num_timesteps - 1), **model_kwargs)
        video = video + dt * v

    # conditional_p_sample_loop iterates over this method and keeps the LAST yield.
    # Yield once at the end so the final sample has the same contract as the DDPM
    # unscale generator (a dict with "video"/"audio").
    yield {"video": video / scale, "audio": audio}


def apply():
    GaussianDiffusion.conditional_p_sample_loop_progressive_unscale = _flow_sample
    print("[flow_sampler_patch] sampler -> Euler flow ODE; "
          f"FM_STEPS={os.environ.get('FM_STEPS', '1000')}, "
          f"FM_TARGET_SCALE={os.environ.get('FM_TARGET_SCALE', '10')}")
