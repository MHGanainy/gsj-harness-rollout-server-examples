"""Path-loaded shim for Polar's vendored trajectory-aware LOO post-processor.

The vendored module lives at
`<server repo>/vendor/polar/src/slime_bridge/reward_post_process.py` —
stdlib-only, read at CP-16 as the loss-side counterpart of the adapter's
group contract. Its parent package is ALSO named `slime_bridge`, colliding
with this repo's directory, so it is loaded BY FILE PATH
(``POLAR_REWARD_PP_FILE``) and never via sys.path — no copy, no fork,
provenance intact (R1: the off-the-shelf candidate, used off the shelf).

slime's ``--custom-reward-post-process-path`` resolves
``cp17_loop.polar_pp.post_process_rewards`` to the vendored function.
"""

from __future__ import annotations

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "polar_reward_post_process", os.environ["POLAR_REWARD_PP_FILE"])
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

post_process_rewards = _module.post_process_rewards
