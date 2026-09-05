# ADR-0005 — entropy and KL are not funded in CP-87

## Context

CP-82 found that the advertised entropy coefficient did not change the
actual loss, while KL failed because the batch had no `ref_log_prob`.
CP-86 added early refusals before collection and before worker allocation.
CP-87 is the separate funding decision required by the audit's phase 5,
alongside B18's investigation of the interrupted 192-row optimizer call.

The memory cost is material. With Qwen3's vocabulary of 151,936, one
32,768-position float32 vocabulary buffer is 18.546875 GiB, before other
live logits, softmax and gradient temporaries. This is an allocation
estimate, not a claim that dynamic batching pads the whole collection to
one width. F-13 records two H200 OOMs from full-vocabulary entropy.
KL would require a frozen reference-policy forward pass: roughly 1.2 GB
of BF16 weights alone for a nominal 0.6B model, plus activations and
workspace. The safe implementation's peak memory and runtime are unmeasured.
The audit prices real controls at 1–3 engineer-days plus GPU validation.

## Decision

The operator declines funding for both controls in CP-87. Nonzero entropy
and KL remain explicitly unsupported and refuse before collection or CUDA
allocation. B09's misleading advertised behavior is closed by refusal.
Any implementation requires a later phase's decision and its own booked
GPU window; no control is enabled during the B18 investigation.

## Consequence

The experiment retains the measured baseline: entropy zero, KL disabled,
the persistent optimizer, global Dr.GRPO advantages and correction weights,
both masks, dynamic microbatches and the 32,768-token budget. Successful
training followed by probe-proven sync would prove execution and measured
weight movement; it would not
establish a safe training policy or remove the observed format-copying risk.
Citation reward and corpus design remain outside this decision.
