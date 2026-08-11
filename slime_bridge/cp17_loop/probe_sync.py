#!/usr/bin/env python3
"""CP-17 sync probe — teacher-forced logprobs on a fixed token stream.

The proof design rests on a CP-09' measurement: the engine's replay path is
bit-deterministic (replay-vs-replay mean|Δ| exactly 0.000000 on this
estate), so ANY nonzero per-position delta between a probe taken before the
weight sync and one taken after attributes entirely to the weight change.
The mechanism is CP-09''s replay leg verbatim: `/v1/completions` with a
token-id prompt, `max_tokens: 0, echo: true, logprobs: 1,
add_special_tokens: false`.

  probe:    probe_sync.py probe <collected.json> <out.json> [base_url]
  compare:  probe_sync.py compare <a.json> <b.json>
"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"


def probe(body_path: str, out_path: str, base_url: str = BASE) -> None:
    body = json.loads(open(body_path).read())
    trace = body["trajectory"]["traces"][0]
    ids = list(trace["prompt_ids"]) + list(trace["response_ids"])
    request = json.dumps({
        "model": "Qwen/Qwen3-0.6B", "prompt": ids, "max_tokens": 0,
        "echo": True, "logprobs": 1, "add_special_tokens": False,
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(
            f"{base_url}/v1/completions", data=request,
            headers={"Content-Type": "application/json"}), timeout=300) as r:
        reply = json.load(r)
    lps = reply["choices"][0]["logprobs"]["token_logprobs"]
    json.dump({"session": body.get("session_id"), "n_ids": len(ids),
               "token_logprobs": lps}, open(out_path, "w"))
    print(f"probe: {len(ids)} ids -> {sum(1 for v in lps if v is not None)} "
          f"teacher-forced logprobs -> {out_path}")


def compare(a_path: str, b_path: str) -> None:
    a = json.load(open(a_path))["token_logprobs"]
    b = json.load(open(b_path))["token_logprobs"]
    assert len(a) == len(b), (len(a), len(b))
    deltas = [abs(x - y) for x, y in zip(a, b)
              if x is not None and y is not None]
    nonzero = sum(1 for d in deltas if d != 0.0)
    print(f"compare: positions={len(deltas)} mean|Δ|={sum(deltas)/len(deltas):.6f} "
          f"max|Δ|={max(deltas):.6f} nonzero={nonzero}/{len(deltas)}")


if __name__ == "__main__":
    {"probe": lambda: probe(*sys.argv[2:]),
     "compare": lambda: compare(*sys.argv[2:])}[sys.argv[1]]()
