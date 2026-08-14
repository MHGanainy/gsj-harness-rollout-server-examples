# pins/ — the thinking-on approved sets, carried here on purpose

`thinking-on/pins.gsj.json` is a byte-identical copy of the library
repo's `pins/thinking-on/pins.gsj.json` (library CP-30, ADR-0024;
copied at library CP-31 from commit `2923cc0`, sha256
`832255e4c3ca0f95f326a3439c11e082ce01cdbad452bc21b30791329cf205de`).

**Why a copy lives here**: the PyPI wheel packages only the
*thinking-off* reference pins (library ADR-0019/ADR-0024 posture —
default resolution keeps meaning thinking-off). This project's shipped
default is thinking-ON (`config.yaml: thinking: "medium"`), and gate G6
is per-mode pins data — so a `pip install`-only consumer would have **no
thinking-on pins at all** and every episode of their first run would be
quarantined `G6:*`. Until the wheel carries both mode files (library
wishlist row 28 — this copy is the interim), the example must bring its
own.

**Who reads it**: both law-6 legs, via `GSJ_PINS_PATH` —

- the **trainer leg**: `train.py` sets `GSJ_PINS_PATH` to this file
  itself, before its first `gsj_rollout` import, whenever the effective
  mode is a non-off level and the variable is unset (the resolution is
  once-per-process, fixed at import — library CP-11b);
- the **receiver leg**: `gsj-rollout serve` is a separate process and
  must be *started* with the variable — the RUNBOOK's serve command
  carries it inline. A server-role host holding the library checkout can
  point at `<library>/pins/thinking-on/pins.gsj.json` instead; the two
  files are byte-equal.

**Drift**: the six non-G6 approved sets in this file must equal the
installed library's reference pins (the library's `derive_pins.py` walk
guards its own two files; nothing guards THIS copy). `train.py`'s
preflight compares them on every run and warns in words when the
installed wheel's reference sets have moved past this copy — if you see
that warning, refresh this file from the library repo at the wheel's
release commit.

Do not hand-edit the JSON; it is generated data (library
`pins/derive_pins.py`).
