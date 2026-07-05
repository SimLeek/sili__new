# Peaked Induction Head Initialisation -- Backburner

## What it was

An attempt to initialise Wk/Wv at 5x scale so the attention softmax starts
peaked on same-token history positions rather than uniform.

## Why it didn't work

Q = h_out @ Wq stays at ~0.02 scale regardless of K/V scale.
Score = Q @ K^T / sqrt(d_k) ~ 0.001-0.01 even with 5x Wk.
Softmax only peaks meaningfully at score magnitude >=2.

## What would work

Rotary position bias (RoPE-style) initialised to peak at exactly period-1
offset. This needs:
1. Hidden state to encode position (learned via RNN)
2. Or: direct positional sinusoid added to K/Q before scoring

Not implemented: requires knowing the task period ahead of time and is
more complex than the main integration test needs.

## When to revisit

Once the gen_toy_mistral -> FoldedLayer pipeline produces a pre-trained model
with meaningful hidden state representations. At that point, attention already
operates on an informative hidden state and peaked init becomes much more useful.
