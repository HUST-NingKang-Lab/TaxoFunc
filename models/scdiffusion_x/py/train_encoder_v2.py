"""AE v2 training entry point.

Uses the upstream `script/training_autoencoder/train_encoder.py` entry (plus
`mgm_patch`) EXCEPT that it applies `py/ae_v2_patch.py`, which additionally
parameterizes the encoder input via `$AE_INPUT`.

`py/mgm_patch.py` and the upstream scripts are left untouched.
"""
import os
import sys

SCX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = os.path.join(SCX, "upstream", "script", "training_autoencoder")
sys.path.insert(0, os.path.join(SCX, "py"))

import hydra                                     # noqa: E402
from omegaconf import DictConfig                 # noqa: E402

from ae_v2_patch import apply as apply_v2_patch  # noqa: E402
apply_v2_patch()

from scdiffusionX.Autoencoder.estimator.encoder_estimator import EncoderEstimator  # noqa: E402


@hydra.main(config_path=os.path.join(UP, "configs"), config_name="train", version_base=None)
def train(cfg: DictConfig):
    EncoderEstimator(cfg).train()


if __name__ == "__main__":
    import traceback
    try:
        train()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
