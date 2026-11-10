"""Numerical guard for the MGnifyMultiOmics run.

WHY: `torch.distributions.Poisson.log_prob` is implemented with `torch.xlogy(value, rate)`.
When a feature is absent (value == 0) AND the predicted rate underflows to exactly 0, the
backward of xlogy is 0/0 = NaN, and PyTorch's anomaly detector aborts training:

    RuntimeError: Function 'XlogyBackward0' returned nan values in its 1th output.

The taxonomy modality makes this reachable: its per-sample depth spans ~6 orders of
magnitude (1 .. 11.5M reads in the L2 train split), so the decoder's softmax has to be
extremely peaked for high-depth samples and can underflow on the rest.

FIX: floor the predicted rate at float32 eps before it enters the likelihood. This is a
numerical guard, NOT a model change -- it only ever alters rates that would otherwise be
exactly 0 (where the Poisson likelihood and its gradient are undefined anyway), and eps is
negligible next to the smallest meaningful rate (median taxonomy totals are ~1.8e2 spread
over 8,234 features).

Everything else in EncoderModel.decode is byte-for-byte the upstream implementation
(commit d60d928) so the model itself is unchanged.
"""
import torch
import torch.nn.functional as F

from scdiffusionX.Autoencoder.models.base.encoder_model import EncoderModel

RATE_FLOOR = torch.finfo(torch.float32).eps     # 1.19e-07


def decode_guarded(self, x, size_factor):
    """Upstream EncoderModel.decode with the Poisson/RNA rate floored at float32 eps."""
    if not self.multimodal:
        if self.encoder_type == "learnt_autoencoder":
            x = self.decoder(x)
        mu_hat = F.softmax(x, dim=1)
        mu_hat = mu_hat * size_factor
        mu_hat = mu_hat.clamp_min(RATE_FLOOR)
    else:
        mu_hat = {}
        for mod in self.modality_list:
            if not self.encoder_multimodal_joint_layers:
                x_mod = self.decoder[mod](x[mod])
            else:
                x_mod = self.decoder[mod](x)

            if mod != "atac" or (mod == "atac" and not self.is_binarized):
                mu_hat_mod = F.softmax(x_mod, dim=1)
                mu_hat_mod = mu_hat_mod * size_factor[mod]
                mu_hat_mod = mu_hat_mod.clamp_min(RATE_FLOOR)      # <-- the guard
            else:
                mu_hat_mod = F.sigmoid(x_mod)
            mu_hat[mod] = mu_hat_mod
    return mu_hat


def apply():
    EncoderModel.decode = decode_guarded
    print("[mgm_patch] EncoderModel.decode patched with rate floor "
          f"{RATE_FLOOR:.3e} (numerical guard, see mgm_patch.py)")
