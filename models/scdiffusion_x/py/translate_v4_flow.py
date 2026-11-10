"""FM translation entry: `ae_v2_patch` + `flow_sampler_patch` (Euler ODE), then upstream.

Use with `--sample_fn ddpm --classifier_scale 0` so the unscale loop is selected and
replaced by flow_sampler_patch. SAMPLE_SEED controls the ODE initial noise.

Usage: python py/translate_v4_flow.py <translate_mgm.py flags...>
"""
import os
import runpy
import sys

SCX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCX, "py"))

from ae_v2_patch import apply as apply_v2_patch                       # noqa: E402
apply_v2_patch()

from flow_sampler_patch import apply as apply_flow_sampler           # noqa: E402
apply_flow_sampler()

if os.environ.get("SAMPLE_SEED") is not None:
    import torch as _th
    _th.manual_seed(int(os.environ["SAMPLE_SEED"]))

runpy.run_path(os.path.join(SCX, "py", "translate_mgm.py"), run_name="__main__")
