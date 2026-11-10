"""AE v2 patch: encoder INPUT parameterization + the inherited numerical guard.

WHY THIS EXISTS (diagnosed 2026-10-13)

`RNAseqLoader.__getitem__` builds `X_norm` as:

    if mod == "atac" and (not self.is_binarized):
        X_norm[mod] = normalize_expression(X[mod], X[mod].sum(), encoder_type)
    else:
        X_norm[mod] = X[mod]

Our runs set `dataset.is_binarized=True` (needed so BOTH modalities are fed counts rather than
log1p, matching the inference path `encode_raw_data`). The side effect is that **BOTH encoders
receive RAW counts** — per-sample totals span 1 .. 11.5M for taxonomy and 1 .. 2.96e6 for KO.

The decoder is `mu = softmax(D(z)) * size_factor`, i.e. depth enters the likelihood ONLY through
the size factor, which is passed separately. So the encoder has no need to represent depth — yet
raw counts make total depth the single most salient direction of the input, and the learned latent
ends up dominated by it (an internal audit found that a large fraction of latent dimensions
correlated strongly with log depth). That is capacity spent on a direction the reconstruction objective
does not need and the downstream cross-modal task cannot use.

`AE_INPUT=logcpm` replaces the encoder input with log1p(1e6 * x / rowsum) -- depth-invariant by
construction. The DECODER and the LOSS ARE UNCHANGED: `X` (raw counts) still enters the
NegativeBinomial/Poisson likelihood and the size factor is still the observed row sum. Only what
the encoder sees changes.

Modes (env var `AE_INPUT`, default `raw` = upstream behaviour):
    raw      upstream: encoder sees the raw counts
    logcpm   encoder sees log1p(1e6 * x / rowsum)
    logcount encoder sees log1p(x)                    (depth-dependent; carried for comparison)
    prop     encoder sees x / rowsum                  (depth-invariant, no log)

Everything else is byte-for-byte upstream `EncoderModel` (commit d60d928), plus the inherited
rate floor from `mgm_patch.py` (numerical guard, not a model change).
"""
import json
import os

import torch
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial
from torch.distributions import Bernoulli, Poisson

from scdiffusionX.Autoencoder.models.base.encoder_model import EncoderModel

RATE_FLOOR = torch.finfo(torch.float32).eps     # 1.19e-07
INPUT_MODE = os.environ.get("AE_INPUT", "raw")
CPM_SCALE = float(os.environ.get("AE_CPM_SCALE", "1e6"))
CLR_SCALE = float(os.environ.get("AE_CLR_SCALE", "1e6"))
CLR_PSEUDO = float(os.environ.get("AE_CLR_PSEUDO", "1.0"))


def normalize_input(x, mode=None):
    """Depth-aware input transform used by the patched encoder only."""
    mode = INPUT_MODE if mode is None else mode
    if mode == "raw":
        return x
    if mode == "logcount":
        return torch.log1p(x)
    s = x.sum(1, keepdim=True).clamp_min(1.0)
    if mode == "prop":
        return x / s
    if mode == "logcpm":
        return torch.log1p(x / s * CPM_SCALE)
    if mode == "clr":
        # centered log-ratio (compositional standard): log(p) - mean_j log(p_j),
        # p = x / rowsum * CLR_SCALE + AE_CLR_PSEUDO.  Matches eval/phylo_clr_probe.py exactly.
        p = x / s * CLR_SCALE + CLR_PSEUDO
        L = torch.log(p)
        return L - L.mean(1, keepdim=True)
    raise ValueError(f"unknown AE_INPUT mode: {mode!r}")


def _apply_dropout_side(model):
    """Idempotently neutralise the dropout modules on the unwanted side.

    Done lazily on the first forward rather than by wrapping `__init__`: Lightning's
    `save_hyperparameters()` inspects the caller frame by ARGUMENT NAME, so a `*args/**kwargs`
    wrapper breaks it. Swapping Dropout -> Identity in place adds and removes no parameters,
    so state_dict keys and checkpoint loading are unaffected.
    """
    if DROPOUT_SIDE == "both" or getattr(model, "_ae_side_done", False):
        return
    if not getattr(model, "multimodal", False):
        model._ae_side_done = True
        return
    for mod in model.modality_list:
        sub = model.encoder[mod] if DROPOUT_SIDE == "decoder" else model.decoder[mod]
        net = getattr(sub, "net", None)
        if net is None:
            continue
        # NOTE: MLP builds `net = Sequential(Sequential(Linear, Norm, act, Dropout), ..., Linear)`
        # -- the Dropout lives INSIDE the nested block, so a flat scan of `net` finds nothing
        # (this was caught by the smoke test before any training was launched).
        _strip_dropout(net)
    model._ae_side_done = True


def _strip_dropout(module):
    """Replace every Dropout in a module tree with Identity, in place (no parameter change)."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Dropout):
            setattr(module, name, torch.nn.Identity())
        else:
            _strip_dropout(child)


DEC_DIMS = json.loads(os.environ.get("AE_DEC_DIMS", "{}") or "{}")


def apply_decoder_dims(model):
    """Give the DECODER its own width/depth, independent of the encoder.

    WHY: upstream builds the decoder as `MLP(**encoder_kwargs[mod])` with `dims[::-1]`, i.e. the
    decoder is FORCED to mirror the encoder. That ties two things that have no reason to be tied:
    the encoder's first layer (in_dim -> h1) and the decoder's last layer (h_last -> in_dim) are
    both ~14M parameters for the KO modality, and nothing says they should be equal. Decoupling is
    the one architecture degree of freedom the OFAT sweep could not
    reach, and it is the reconstruction side of the trade-off.

    Applied by REPLACING `model.decoder[mod]` after construction, so the state_dict keys are
    unchanged (same depth => same Sequential layout). `MLP.norm_type` is not stored on the module,
    so it is recovered by inspecting the existing decoder's normalisation layers.
    """
    if not DEC_DIMS or not getattr(model, "multimodal", False):
        return model
    from scdiffusionX.Autoencoder.models.base.utils import MLP
    for mod, dims in DEC_DIMS.items():
        old = model.decoder[mod]
        nt = "batchnorm" if any(isinstance(m, torch.nn.BatchNorm1d) for m in old.modules()) \
            else "layernorm"
        model.decoder[mod] = MLP(dims=list(dims), norm=old.norm, dropout=False, dropout_p=0.0,
                                 activation=old.activation, final_activation=None, norm_type=nt)
        print(f"[ae_v2_patch] decoder[{mod}] decoupled -> dims {list(dims)} (norm {nt})")
    return model


def encode_v2(self, batch):
    """Upstream EncoderModel.encode with the input transform inserted."""
    _apply_dropout_side(self)
    if not self.multimodal:
        return self.encoder(normalize_input(batch["X_norm"].to(self.device)))
    z = {}
    for mod in self.modality_list:
        z_mod = self.encoder[mod](normalize_input(batch["X_norm"][mod].to(self.device)))
        z[mod] = z_mod
    if self.encoder_multimodal_joint_layers:
        z_joint = torch.cat([z[mod] for mod in z], dim=-1)
        z = self.encoder_joint(z_joint)
    return z


LATENT_NOISE = float(os.environ.get("AE_LATENT_NOISE", "0"))
# AE_LATENT_SHRINK=1 trains the decoder on latents shrunk toward the batch mean by a random
# factor alpha ~ U[1-s, 1]. A ridge/conditional-mean estimate E[z|z_tax] is exactly a SHRUNKEN
# version of z, so this targets the measured failure mode more directly than isotropic noise.
LATENT_SHRINK = float(os.environ.get("AE_LATENT_SHRINK", "0"))
# AE_LATENT_K=k restricts BOTH perturbations above to the FIRST k latent dimensions.
#
# WHY (measured, 2026-10-13): perturbing the WHOLE latent made the encoder collapse it -- with
# AE_LATENT_SHRINK=0.5 the effective rank collapsed and both reconstruction and generation
# quality fell. The encoder's rational response to "the decoder must
# still work when these directions are shrunk" is to stop using those directions at all.
#
# Restricting the perturbation to a designated generation subspace keeps the rest of the latent
# untouched, so reconstruction-only information has somewhere to live:
#     z = [ c (first k dims: trained to survive shrinkage) | r (rest: trained clean) ]
# which is exactly the conditional/residual split the generation path needs -- the diffusion only
# has to model `c`; `r` can come from a prior. Default k = all dims = the earlier behaviour.
LATENT_K = int(os.environ.get("AE_LATENT_K", "0"))

# AE_DROPOUT_SIDE = both | decoder | encoder
#
# The config's `dropout` flag is handed to ONE kwargs dict that upstream uses to build BOTH the
# encoder and the decoder MLP, so a config-only dropout run regularises both sides at once and
# cannot say WHICH side produced the effect. This switch disables the dropout modules on the
# unwanted side *after construction* by replacing them with nn.Identity() in place.
#
# Chosen deliberately over rebuilding the MLP: it adds/removes no parameters, so the state_dict
# keys are untouched and the identical patch applies at training and at evaluation time.
#   decoder : encoder dropout off, decoder dropout on  <- the arm we care about here
#   encoder : decoder dropout off, encoder dropout on  <- the control that isolates the other side
DROPOUT_SIDE = os.environ.get("AE_DROPOUT_SIDE", "both")


def decode_guarded(self, x, size_factor):
    """Upstream EncoderModel.decode with the Poisson/RNA rate floored at float32 eps.

    Plus an OPTIONAL training-only perturbation of the latent (`AE_LATENT_NOISE=sigma`), which is
    the direct lever on the reconstruction <-> generation trade-off:

        the generation pipeline never feeds the decoder a TRUE latent.  It feeds it
        `E[z | z_tax]` -- a shrunken, off-manifold estimate -- and the measured cost of that is
        large: decoding a true latent is substantially better than decoding a linear
        prediction from taxonomy.
        Perturbing z during training forces the decoder to reconstruct from a neighbourhood of the
        latent rather than from the exact point, i.e. it buys off-manifold robustness at a small
        reconstruction cost.  sigma is swept so the trade-off CURVE, not a single point, is reported.

    Noise is scaled by the batch's per-dimension latent std, so sigma is a relative level.
    Applied to `rna` only (the generation target); inference is never perturbed (`self.training`).
    """
    if not self.multimodal:
        if self.encoder_type == "learnt_autoencoder":
            x = self.decoder(x)
        mu_hat = F.softmax(x, dim=1)
        mu_hat = mu_hat * size_factor
        mu_hat = mu_hat.clamp_min(RATE_FLOOR)
    else:
        mu_hat = {}
        for mod in self.modality_list:
            x_in = x[mod] if not self.encoder_multimodal_joint_layers else x
            if mod == "rna" and self.training and (LATENT_NOISE > 0 or LATENT_SHRINK > 0):
                k = LATENT_K if 0 < LATENT_K < x_in.shape[-1] else x_in.shape[-1]
                head, tail = x_in[..., :k], x_in[..., k:]
                if LATENT_NOISE > 0:
                    scale = head.detach().std(0, keepdim=True).clamp_min(1e-6)
                    head = head + LATENT_NOISE * torch.randn_like(head) * scale
                if LATENT_SHRINK > 0:
                    mu_ = head.detach().mean(0, keepdim=True)
                    alpha = 1.0 - LATENT_SHRINK * torch.rand(head.shape[0], 1, device=head.device,
                                                            dtype=head.dtype)
                    head = mu_ + alpha * (head - mu_)
                x_in = torch.cat([head, tail], dim=-1)
            x_mod = self.decoder[mod](x_in)

            if mod != "atac" or (mod == "atac" and not self.is_binarized):
                mu_hat_mod = F.softmax(x_mod, dim=1)
                mu_hat_mod = mu_hat_mod * size_factor[mod]
                mu_hat_mod = mu_hat_mod.clamp_min(RATE_FLOOR)      # <-- numerical guard
            else:
                mu_hat_mod = F.sigmoid(x_mod)
            mu_hat[mod] = mu_hat_mod
    return mu_hat


def _step_renorm(self, batch, dataset_type):
    """Upstream `EncoderModel._step` with EACH MODALITY's loss averaged over the rows that
    actually carry that modality.

    WHY THIS EXISTS.  Upstream computes

        loss = sum_mod  [ -log_prob(X[mod]).sum(1).mean() ]

    The `.mean()` divides by the BATCH SIZE, not by the number of rows that carry the modality.
    Zero-filling a missing modality makes that modality's per-row loss ~0, so the RNA (KO) loss
    becomes `(n_rna_carriers / batch_size) * mean_over_carriers` -- i.e. the KO autoencoder's
    effective learning rate is silently scaled by the fraction of paired rows in the batch.
    Per-sample the gradient semantics are still exactly "masked" (see
    eval/ae_zero_fill_equivalence.py), but the batch-level normalisation is NOT equivalent to
    training the KO AE on the paired rows alone.

    This variant divides each modality by its own carrier count, which makes
    "train the two AEs jointly on paired + unpaired rows" numerically identical to
    "train the KO AE on the paired rows and the taxonomy AE on all rows" -- up to minibatch
    sampling noise. It is what makes an unpaired-data experiment interpretable at all.

    Enabled only via `AE_EXT_RENORM=1`; default OFF so every existing result keeps its meaning.
    For a fully-paired file (no zero rows) this is bit-identical to upstream.
    """
    X = {mod: batch["X"][mod].to(self.device) for mod in batch["X"]}
    size_factor = {mod: X[mod].sum(1).unsqueeze(1).to(self.device) for mod in X}
    y = batch["y"][self.conditioning_covariate].to(self.device)
    z = self.encode(batch)
    mu_hat = self.decode(z, size_factor)

    loss = 0
    for mod in mu_hat:
        if mod == "rna":
            if not self.covariate_specific_theta:
                px = NegativeBinomial(mu=mu_hat[mod], theta=torch.exp(self.theta))
            else:
                px = NegativeBinomial(mu=mu_hat[mod], theta=torch.exp(self.theta[y]))
        elif mod == "atac":
            px = Poisson(rate=mu_hat[mod]) if not self.is_binarized else Bernoulli(probs=mu_hat[mod])
        else:
            raise NotImplementedError
        per_row = -px.log_prob(X[mod]).sum(1)                       # (B,)
        carrier = (X[mod].sum(1) > 0).to(per_row.dtype)             # rows carrying this modality
        loss = loss + (per_row * carrier).sum() / carrier.sum().clamp_min(1.0)

    self.log(f'{dataset_type}/loss', loss, on_epoch=True, prog_bar=True)
    return loss


def _seed_everything(tag=""):
    """Explicit, reproducible seeding.

    WHY THIS EXISTS. Upstream never seeds anything (`grep -rn seed_everything upstream/` is empty
    and the Trainer does not seed either). Empirically that produced a bizarre situation: two runs
    of `edrop05` came out with **bit-identical weights** while two runs of `c0_ctl` differed —
    i.e. "re-running the same config" is NOT a reliable independent replicate, and a variance
    estimated from such re-runs would be wrong (too small, possibly zero).

    With AE_SEED set, the run is a deterministic function of (config, seed), so choosing distinct
    seeds gives genuine, labelled replicates.
    """
    import random
    import numpy as np
    import torch
    seed = int(os.environ.get("AE_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[ae_v2_patch] AE_SEED={seed} applied{tag}")


def apply():
    EncoderModel.encode = encode_v2
    EncoderModel.decode = decode_guarded
    # `EncoderEstimator.init_model` is NOT the only place an AE gets built: the diffusion's
    # `multimodal_datasets.encode_raw_data` constructs `EncoderModel(...)` directly and then
    # loads a checkpoint. Hooking `load_state_dict` makes the decoupled decoder apply on EVERY
    # code path -- which is exactly what "plumbing the final AE into the diffusion" needs.
    _orig_lsd = EncoderModel.load_state_dict

    def _lsd_with_dec_dims(self, *a, **k):
        apply_decoder_dims(self)
        return _orig_lsd(self, *a, **k)

    EncoderModel.load_state_dict = _lsd_with_dec_dims
    msg = (f"[ae_v2_patch] EncoderModel.encode patched: AE_INPUT={INPUT_MODE!r} "
           f"(scale {CPM_SCALE:g}); decode patched with rate floor {RATE_FLOOR:.3e}; "
           f"latent noise sigma={LATENT_NOISE:g} (train-time, rna only)")
    if os.environ.get("AE_SEED"):
        _seed_everything()
    if os.environ.get("AE_EXT_RENORM", "0") == "1":
        EncoderModel._step = _step_renorm
        msg += "; _step patched: AE_EXT_RENORM=1 (per-modality carrier-count normalisation)"
    if DEC_DIMS:
        from scdiffusionX.Autoencoder.estimator.encoder_estimator import EncoderEstimator
        _orig_init_model = EncoderEstimator.init_model

        def _init_model_decoupled(self):
            _orig_init_model(self)
            apply_decoder_dims(self.encoder_model)

        EncoderEstimator.init_model = _init_model_decoupled
        msg0 = f"; AE_DEC_DIMS decoupled decoders: {DEC_DIMS}"
    else:
        msg0 = ""
    if DROPOUT_SIDE != "both":
        msg += (f"; dropout side={DROPOUT_SIDE!r} "
                f"({'encoder dropout disabled' if DROPOUT_SIDE == 'decoder' else 'decoder dropout disabled'})")
    print(msg + msg0)
