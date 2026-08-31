#!/usr/bin/env python3
"""CP-17 sync probe — teacher-forced logprobs on a fixed token stream.

Moved from `slime_bridge/cp17_loop/probe_sync.py` at CP-69 (the slime tree
is frozen at tag `slime-cp17`); the instrument is trainer-agnostic and the
verl loop's sync proof reuses it verbatim. The one CP-69 delta: the model
name and base URL are parameters (the CP-17 original hardcoded the
estate's), and `probe`/`compare` return their stats so `train_loop.py` can
gate on them. The wire mechanism and the printed lines are unchanged.

The proof design rests on a CP-09' measurement: the engine's replay path is
bit-deterministic (replay-vs-replay mean|Δ| exactly 0.000000 on this
estate), so ANY nonzero per-position delta between a probe taken before the
weight sync and one taken after attributes entirely to the weight change.
The mechanism is CP-09''s replay leg verbatim: `/v1/completions` with a
token-id prompt, `max_tokens: 0, echo: true, logprobs: 1,
add_special_tokens: false`.

  probe:    probe_sync.py probe <collected.json> <out.json> [base_url] [model]
  compare:  probe_sync.py compare <a.json> <b.json>
"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3-0.6B"


def probe(body_path: str, out_path: str, base_url: str = BASE,
          model: str = MODEL) -> dict:
    body = json.loads(open(body_path).read())
    trace = body["trajectory"]["traces"][0]
    ids = list(trace["prompt_ids"]) + list(trace["response_ids"])
    request = json.dumps({
        "model": model, "prompt": ids, "max_tokens": 0,
        "echo": True, "logprobs": 1, "add_special_tokens": False,
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(
            f"{base_url}/v1/completions", data=request,
            headers={"Content-Type": "application/json"}), timeout=300) as r:
        reply = json.load(r)
    lps = reply["choices"][0]["logprobs"]["token_logprobs"]
    result = {"session": body.get("session_id"), "n_ids": len(ids),
              "token_logprobs": lps}
    json.dump(result, open(out_path, "w"))
    print(f"probe: {len(ids)} ids -> {sum(1 for v in lps if v is not None)} "
          f"teacher-forced logprobs -> {out_path}")
    return result


def compare(a_path: str, b_path: str) -> dict:
    a = json.load(open(a_path))["token_logprobs"]
    b = json.load(open(b_path))["token_logprobs"]
    assert len(a) == len(b), (len(a), len(b))
    deltas = [abs(x - y) for x, y in zip(a, b)
              if x is not None and y is not None]
    nonzero = sum(1 for d in deltas if d != 0.0)
    print(f"compare: positions={len(deltas)} mean|Δ|={sum(deltas)/len(deltas):.6f} "
          f"max|Δ|={max(deltas):.6f} nonzero={nonzero}/{len(deltas)}")
    return {"positions": len(deltas), "mean_abs": sum(deltas) / len(deltas),
            "max_abs": max(deltas), "nonzero": nonzero}


if __name__ == "__main__":
    {"probe": lambda: probe(*sys.argv[2:]),
     "compare": lambda: compare(*sys.argv[2:])}[sys.argv[1]]()
