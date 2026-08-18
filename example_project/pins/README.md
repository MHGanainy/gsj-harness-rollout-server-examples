# pins/ — retired (library CP-33)

The byte-identical copy of the library's `pins/thinking-on/pins.gsj.json`
that lived here from library CP-31 (F-40's interim cure) **retired at
library CP-33 (wishlist 28)**: the wheel now packages BOTH mode files, so
`pip install gsj-harness-rollout-server>=0.1.1` carries the thinking-on
set at

```
site-packages/gsj_rollout/pins/thinking-on/pins.gsj.json
```

`train.py` locates it there itself (before its first `gsj_rollout`
import — the CP-11b once-per-process rule) and sets `GSJ_PINS_PATH` for
its own leg; the RUNBOOK's serve command carries the same path inline for
the receiver leg. The drift warning that guarded this copy retired with
it — both packaged files now build from one tree (library ADR-0019's
by-construction argument), so there is no second copy to drift.

On a pre-0.1.1 install, `train.py --thinking medium` exits naming the
cure: `pip install -U gsj-harness-rollout-server`.
