"""Linear flow-matching objective for the taxonomy -> KO latent translation.

Training-only monkey-patch; nothing else changes. The model is the same
MultimodalUNet (DCA, biome label, clean condition). Only the objective changes:

    video = z_ko / rna_std                 # dataset video, mean per-dim std 0.1
    u     = FM_TARGET_SCALE * video        # default 10 -> unit scale
    eps  ~ N(0, I)
    u_t   = (1 - t) * eps + t * u,   t ~ U[0, 1)
    loss  = mean_flat((u - eps - model(u_t, clean_audio, t * 999)) ** 2)

The condition is CLEAN at every t.
This file is a training patch only; sampling uses flow_sampler_patch.py.
"""
import os

import torch as th

from scdiffusionX.DiffusionBackbone.multimodal_gaussian_diffusion import (
    ModelVarType, GaussianDiffusion, mean_flat)


def _unwrap_model(model):
    """Return the raw UNet/DDP model if `model` is SpacedDiffusion's _WrappedModel.

    _WrappedModel maps integer timesteps through `timestep_map`; it cannot accept the
    continuous t used by flow matching.  With 1000 diffusion steps the map is the
    identity and rescale_timesteps=False, so calling `.model` directly is equivalent.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(model, "timestep_map"):
        if getattr(model, "rescale_timesteps", False):
            raise ValueError("flow_objective_patch assumes rescale_timesteps=False")
        return inner
    return model


def multimodal_training_losses_flow(self, model, x_start, t, model_kwargs=None, noise=None):
    video_start = x_start["video"]          # [B, 1, 100] = z_ko / rna_std
    audio_start = x_start["audio"]          # [B, 1, 200] = clean z_tax / atac_std
    if model_kwargs is None:
        model_kwargs = {}
    model = _unwrap_model(model)

    scale = float(os.environ.get("FM_TARGET_SCALE", "10"))
    b = video_start.shape[0]

    # Integer t from the schedule sampler + U[0,1) jitter -> continuous t in [0, 1).
    t_cont = (t.float() + th.rand(b, device=t.device)) / self.num_timesteps

    if noise is None:
        eps = th.randn_like(video_start)
    else:
        eps = noise["video"]

    u = video_start * scale
    u_t = (1.0 - t_cont)[:, None, None] * eps + t_cont[:, None, None] * u
    v_target = u - eps

    v_pred, _ = model(u_t, audio_start,
                      t_cont * (self.num_timesteps - 1), **model_kwargs)

    # Only LEARNED_RANGE (learn_sigma=True) has a variance head; this project uses
    # learn_sigma=False, so this branch is inactive.
    if self.model_var_type == ModelVarType.LEARNED_RANGE:
        v_pred, _ = th.split(v_pred, video_start.shape[2], dim=2)

    return {"loss": mean_flat((v_target - v_pred) ** 2)}


def apply():
    GaussianDiffusion.multimodal_training_losses = multimodal_training_losses_flow
    print("[flow_objective_patch] objective -> linear flow matching; "
          f"FM_TARGET_SCALE={os.environ.get('FM_TARGET_SCALE', '10')}")
