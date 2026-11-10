"""FM training entry: `ae_v2_patch` + `flow_objective_patch`, then upstream multimodal_train.py.

Uses the same flags/data/AE as the DDPM training entry (not included in this
release); only the objective is replaced by linear flow matching. The upstream seeding is disabled (`set_seed_logger_random` comments out
the seed lines), so set the seed explicitly here.

Usage: python py/train_df_flow.py <upstream flags...>
"""
import os
import random
import runpy
import sys

import numpy as np
import torch as th

SCX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCX, "py"))

SEED = int(os.environ.get("FM_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed_all(SEED)

from ae_v2_patch import apply as apply_v2_patch           # noqa: E402
apply_v2_patch()

from flow_objective_patch import apply as apply_flow      # noqa: E402
apply_flow()

UPSTREAM = os.path.join(SCX, "upstream", "script", "training_diffusion", "py_scripts",
                        "multimodal_train.py")
runpy.run_path(UPSTREAM, run_name="__main__")
