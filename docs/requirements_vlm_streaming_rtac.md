# Requirements: VLM pipeline, streaming conversion, RTAC curiosity agent

```
        .-~~~-.
  .- ~ ~-(       )_ _        the model is bigger than the RAM,
 /                     ~ -.   so we eat it one layer at a time
|                           \
 \                         .'
   ~- . _____________ . -~
```

Target use case: a 2D vision agent operating desktop apps (FreeCAD, KiCad) --
enclosure design, mask repair for PCB fab acceptance, accidental-antenna
checks. Requirements that follow from this: BASE model (no chat loop;
continual operation with occasional instructions), VISION (mandatory),
converts on the same 32-96 GB box that runs it (streaming conversion).

Division of labor: this doc + initial code = Fable. Edge cases, verification
against real downloads, and test hardening = follow-up sessions.

---

## 1. Target model: Mistral-Small-3.1-24B-Base-2503

`mistralai/Mistral-Small-3.1-24B-Base-2503`, arch `Mistral3ForConditionalGeneration`
(HF transformers `mistral3`): Pixtral vision tower + Mistral text + projector.
Apache 2.0. ~48 GB bf16 on disk; fits the 32-96 GB constraint only via
streaming conversion (Section 3). NOTE: the gghfez/Mistral-Small-24B-Base-2501
mirror the vllm tag surfaced is Small 3.0 = TEXT-ONLY; do not use it. 2503 is
the vision base.

### 1.1 Component dims (real / toy)

| component | param            | real 24B | toy  |
|-----------|------------------|----------|------|
| text      | hidden            | 5120     | 32   |
| text      | layers            | 40       | 12   |
| text      | heads / kv        | 32 / 8   | 4/2  |
| text      | head_dim          | 128      | 16   |
| text      | q out = h*hd      | 4096     | 64   |
| text      | intermediate      | 32768    | 64   |
| text      | vocab (tekken)    | 131072   | 64   |
| vision    | hidden            | 1024     | 16   |
| vision    | layers            | 24       | 6    |
| vision    | heads / head_dim  | 16 / 64  | 2/8  |
| vision    | intermediate      | 4096     | 32   |
| vision    | patch (conv k=s)  | 14       | 4    |
| projector | spatial_merge     | 2        | 2    |

**The 3.1 quirk the old toy missed:** text `heads*head_dim = 4096 != hidden
5120`. q_proj is (4096, 5120), k/v (1024, 5120), o_proj (5120, 4096). The toy
MUST reproduce shape asymmetry (toy: q (64,32), k/v (32,32), o (32,64)) or the
conversion never exercises the non-square fold path.

### 1.2 Tensor name schema (HF format) -- VERIFIED against the real model

**Status: CONFIRMED.** Fetched
`mistralai/Mistral-Small-3.1-24B-Base-2503/resolve/main/model.safetensors.index.json`
and `.../config.json` directly (July 2026). This replaces the two-vintage
guess this section originally contained -- one of those two guessed schemas
was right; the other (which had been the toy's DEFAULT) had no evidence for
it at all and has been removed entirely, not kept as a flagged alternate.

Verified schema (NO "model." top-level wrapper -- language_model.*,
vision_tower.*, multi_modal_projector.* are all top-level keys):
```
language_model.model.embed_tokens.weight
language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight
language_model.model.layers.{i}.{input,post_attention}_layernorm.weight
language_model.model.norm.weight
language_model.lm_head.weight
vision_tower.patch_conv.weight                  # 4-D! (1024,3,14,14), kept dense
vision_tower.ln_pre.weight
vision_tower.transformer.layers.{i}.attention.{q,k,v,o}_proj.weight
vision_tower.transformer.layers.{i}.attention_norm.weight
vision_tower.transformer.layers.{i}.feed_forward.{gate,up,down}_proj.weight
vision_tower.transformer.layers.{i}.ffn_norm.weight
multi_modal_projector.norm.weight
multi_modal_projector.patch_merger.merging_layer.weight
multi_modal_projector.linear_1.weight
multi_modal_projector.linear_2.weight
```
40 language layers (0-39), 24 vision layers (0-23). Layer suffix sets diffed
exactly against the real weight_map -- match confirmed for language layer 0,
vision layer 0, and all top-level keys (embed/norm/lm_head/projector/patch_conv/
ln_pre). `gen_toy_mistral_vlm.py`'s schema now matches this exactly and is the
ONLY schema the module produces. The previously-default "model."-wrapped guess
that had no evidence behind it was removed entirely (not kept behind a flag)
-- vision is a hard requirement for the target use case, so a speculative
schema with no known real target was dead weight rather than defensive
coverage.

**Numeric dimensions -- also VERIFIED, no corrections needed:** every value in
the real/toy table (section 1.1) matched config.json exactly on first fetch:
text hidden_size=5120, num_hidden_layers=40, num_attention_heads=32,
num_key_value_heads=8, head_dim=128, intermediate_size=32768, vocab_size=131072;
vision hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
head_dim=64, intermediate_size=4096, patch_size=14; spatial_merge_size=2;
projector: gelu activation, bias=false. The heads*head_dim != hidden quirk
(32*128=4096 vs hidden 5120) that motivated the toy's asymmetric q_proj is
confirmed real, not a misremembering.

**embed_tokens / lm_head tied-weight question (checklist item, now answered):**
`language_model.model.embed_tokens.weight` and `language_model.lm_head.weight`
appear as two SEPARATE keys in weight_map, pointing to DIFFERENT shard files
(00001 and 00010 respectively). Whether or not they are numerically tied at
the config level, they are materialized as two distinct on-disk tensors in
this checkpoint. Consequence: streaming_sparsify needs NO special-casing for
this -- both get their own manifest entry and get pruned independently like
any other 2-D tensor. "Dedup" is only a disk-space/redundant-compute question
(pruning both separately when they might be identical), not a correctness
requirement. Downgraded from checklist item to a note; no code change needed.

### 1.3 Toy generator: `tests/unit/python/gen_toy_mistral_vlm.py`

- Emits BOTH block families so `detect_repeated_block_groups` must find TWO
  groups (12 language layers AND 6 vision layers). Acceptance: exactly 2
  groups; per-suffix `fold_one_suffix` produces correct dims for all 7 text
  suffixes AND all 7 vision suffixes.
- 4-D `patch_conv.weight` present -- pruning must reshape to (out, -1) before
  CSR (torch `to_sparse_csr` is 2-D only). Record original shape.
- `--legacy-prefix` flag emits the legacy vintage; detection tests run BOTH.
- Same forced-zero-fraction ground-truth JSON pattern as the old generator.
- Old `gen_toy_mistral.py` stays (regression tests reference it) with a
  header note pointing here.

Edge cases for follow-up: tied embeddings (embed_tokens vs lm_head),
`_parse_block_key` matching `vision_tower.transformer.layers.N.` (dots in
prefix before `layers`), bias-bearing variants (3.1 projector is bias-free;
other VLMs are not).

---

## 2. Runtime wiring (image -> folded VLM -> heads)

Order per step (mirrors Mistral3ForConditionalGeneration.forward):
```
image -> patch_conv (4x4 conv == unfold+linear) -> ln_pre
      -> vision blocks (folded, attn collapse o(v(x)) single-token)
      -> projector: norm -> patch_merger(2x2 merge) -> linear_1 -> GELU -> linear_2
      -> language blocks (folded)  -> heads (recon / action / value)
```
Initial code in test_mandelbrot_rl keeps the simplified single-vector path
(mean-pooled patches). Follow-up: per-patch sequence through vision fold,
2x2 spatial merge implemented as constant gather matmul.

---

## 3. Streaming (layer-by-layer) conversion -- TOP PRIORITY

Rationale: convert on the machine that runs the model. 24B bf16 = ~48 GB; a
whole-state-dict load is already marginal at 96 GB and impossible at 32.

Mechanism: `safetensors.safe_open(shard, framework='pt')` loads ONE tensor at
a time; multi-shard via `model.safetensors.index.json` weight_map.

### 3.1 Memory budget (why per-suffix works)

Peak(phase 1, sparsify) = one dense tensor + its CSR. Largest 24B tensor:
down_proj (5120, 32768) = 335 MB bf16 / 671 MB fp32. Trivial.

Peak(phase 2, per-suffix fold) = all CSR folds of ONE suffix. down_proj x40 at
density d: 40 * 168M * d * 8 B (fp32 val + int32 col) ~= 54 GB * d.
d=0.10 -> 5.4 GB (fits 32 GB). d=0.50 -> 27 GB (does NOT fit 32 GB).
**Requirement:** `--no-stack` fallback emits per-layer descriptors (n_folds=1
each) when projected suffix nnz exceeds a `--mem-budget`; estimator sums nnz
from phase-1 manifest before loading anything.

### 3.2 API (initial code: `sili/conversion/streaming_prune.py`)

```
streaming_sparsify(model_dir, out_dir, threshold=None, dtype=fp32)
  # phase 1: per-tensor prune->CSR->save out_dir/tensors/{name}.pt
  # manifest.json: {name: {shape, orig_shape, nnz, layout, shard}}
  # resumable: skips names already present in manifest
streaming_fold_suffix(out_dir, prefix, suffix, mem_budget_gb=8)
  # phase 2: sequential load of one suffix -> stack_csr_vertical
  # returns FoldedBlockDescriptor (or list of n_folds=1 descriptors if over budget)
```
Handles: bf16->fp32, 4-D reshape (orig_shape recorded), 1-D kept raw-dense,
both prefix vintages (regex accepts `(model\.)?(language_model|vision_tower)`).

Follow-up edge cases: gated-repo auth flow, resume-after-crash fsck of partial
.pt, tied-weight dedup, MoE (see TODO -- expert-merge into one sparse layer;
CSR sparsity subsumes MoE routing sparsity; NOT initial scope).

---

## 4. RTAC-based curiosity agent (rtrl/rtac.py pattern)

Source: rmst/rtrl `rtrl/rtac.py` (NeurIPS 2019 Real-Time RL). The load-bearing
lines, verbatim:

```python
# shared trunk, ONE combined loss, one optimizer:
loss_total = self.loss_alpha * loss_actor + (1 - self.loss_alpha) * loss_critic   # alpha=0.2
# real-time input includes the PREVIOUS action:
_, next_value_target, _ = self.model_target((next_obs[0], new_actions.detach()))
# SAC-style value target with entropy, PopArt-normalized:
value_target = (1.-terminals)*self.discount*self.outputnorm_target.unnormalize(next_value_target)
value_target += self.reward_scale * rewards
value_target -= self.entropy_scale * new_actions_log_prob.detach()
loss_critic  = sum(mse_loss(v, value_target) for v in values)
loss_actor   = -(1.-terminals)*discount*unnormalize(next_value) + entropy_scale*log_prob
```

### 4.1 Mapping to sili (what the combined loss contains)

Per Simleek's sketch -- prediction loss concat energy loss feed the critic:

- **reward** r_t = w_r * recon_mse_t + w_e * energy_aux_t  (intrinsic:
  novelty + homeostatic pressure). Both scaled to comparable magnitude.
  OPEN CHOICE (flagged): the two scalars may ALSO be critic INPUT features
  (value conditioned on current loss levels) -- both readings implemented
  behind `--critic-sees-losses`.
- **real-time state**: critic/policy input = [h_out ; onehot(prev_action)]
  (rtac's `(obs, action)` pairing).
- **combined loss** via `sili.tensor.combine_losses` (Section 5):
  `alpha*L_actor + (1-alpha)*L_critic + recon-injection + aux_h*w + aux_o*w`
  -- ONE backward, one traversal (sili multi-root backward double-counts
  shared subgraphs; this is why combine_losses is now a standard function).
- **discrete-action adaptation** (rtac is continuous/rsample; we have 7
  actions): actor term = A2C policy gradient with advantage
  delta = r + gamma*V_targ(h_t,a_t) - V(h_{t-1},a_{t-1}), plus entropy bonus
  entropy_scale*H(pi). Pathwise gradient does not apply to discrete sampling.
- **target value net**: EMA copy of value head (rtac target_update).

### 4.2 Initial code simplifications (follow-up items)

Implemented in `--agent rtac`: value head + EMA target, prev-action
real-time input, one-step TD on stored (h_{t-1} as constant leaf), entropy
bonus, combined loss through combine_losses, PopArt output normalization
(sili/rl_utils.py, verified by test_popart.py), double critic
(reduce(torch.min) equivalent, both heads trained toward the same target).

Still deferred (see section 7 checklist for the concrete technical blockers,
not just a list): critic gradient THROUGH the core trunk (needs either a
deferred-SGD-apply refactor or a duplicate forward pass at the critic step --
both are real control-flow changes, not a small patch), replay memory + batch
(needs a transition buffer, a batch-axis audit of EnergyDynamics and every
other single-sample op in this file, and a decision on recompute-vs-cache for
the curiosity reward).

Expectation setting: policy differentiation will still likely need the long
local run (150k steps); RTAC gives the critic machinery so the advantage is
no longer near-zero-variance raw reward.

---

## 5. `combine_losses` (sili/tensor.py) -- standard multi-loss backward

```python
combine_losses((tensor, grad_array),   # inject d(total)/d(tensor) = grad_array
               (scalar_tensor, 0.05),  # weighted scalar loss term
               scalar_tensor)          # bare scalar term, weight 1
# -> returns scalar Tensor; call .backward() ONCE.
```
Replaces the hand-rolled proxy in test_mandelbrot_rl; every multi-loss caller
(RTAC, energy+task, future SISLDO training loops) uses this.

---

## 6. Display + long-run

`--display` on test_mandelbrot_rl: cv2 window, view | reconstruction
side-by-side, 8x nearest upscale, waitKey(1); graceful no-cv2 fallback (warn
once, continue headless). Long-run perf command (local):
```
python3 -m tests.integration.test_mandelbrot_rl --compare --core zero \
    --steps 150000 --timeout 3600 --report-every 500 --display
```

## 7. Follow-up checklist (smaller models)

- [x] **DONE**: verified Section 1.2 names against the real 2503 index.json
      and config.json directly. Schema fixed (toy's default was wrong, see
      section 1.2); dimensions needed no fix. gen_toy_mistral_vlm.py now
      matches the real weight_map exactly for both block families and all
      top-level keys.
- [x] **FIXED**: prefix-collision bug found by the new toy, both vintages.
      Three functions in rnn_fold.py all filtered on bare layer INDEX only:
      `detect_repeated_block_groups` (merged block_params across families ->
      groups sized [6,6] instead of [12]+[6]), `fold_block_group` (collected
      suffixes from the WRONG family -> KeyError trying to look up a vision
      suffix under the language prefix), and `fold_sparse_payload`'s
      `_group_prefix` reverse-lookup + removal loop (nondeterministic prefix
      guess + risk of removing the wrong family's tensors). All three now
      filter/key on (prefix, index). detect_repeated_block_groups' return
      type changed List[List[int]] -> List[Tuple[str, List[int]]] (prefix is
      now returned directly, no more reverse lookup needed).
      Permanent regression test: tests/integration/test_multifamily_fold.py
      (both prefix vintages, asserts group sizes/prefixes, fold_sparse_payload
      end-to-end, lossless nnz, zero leftover per-layer keys).
- [ ] (moved to later-todo, see refactoring_todo.md: patch_conv is
      602K params, small in every dimension -- Simleek's prior sparsification
      attempts found small vision tensors don't sparsify well. Kept DENSE for
      now in both the toy and streaming_prune.py; sparsify-when-it-actually-
      helps is deferred alongside MoE expert-merge.)
- [x] **DONE**: `--no-stack` over-budget fallback + nnz estimator test.
      tests/integration/test_streaming_prune.py runs streaming_sparsify
      against a REAL on-disk sharded safetensors checkpoint (3 shards +
      index.json, via safetensors.torch.save_file), then verifies
      estimate_suffix_bytes and both branches of streaming_fold_suffix
      (large budget -> one stacked descriptor, tiny budget -> per-layer
      list) via FoldedLayer forward passes from both descriptor forms.
- [x] **DONE**: resume/fsck of interrupted streaming_sparsify.
      Found and fixed a real gap: the resume check only tested
      os.path.exists(), which passes for a truncated file left by a process
      killed mid-torch.save (crash, OOM-kill) -- such a file would be
      silently trusted forever and fail much later, at streaming_fold_suffix
      time, with a far less obvious error. Added _tensor_file_ok(), which
      calls torch.load() before trusting an existing file; a load failure
      forces regeneration. test_streaming_prune.py covers both the missing-
      file case (which a bare exists() check would already handle) and the
      corrupted-but-present case (which specifically justifies having a real
      fsck at all, not just an existence check).
- [x] **DONE**: PopArt. sili/rl_utils.py implements it generally (weight-
      like arrays scale-only, bias-like arrays scale+shift), verified by a
      dedicated rescale-invariance test (tests/integration/test_popart.py)
      that checks the DENORMALIZED prediction is unchanged by a stats update
      alone -- the property that makes PopArt correct rather than just a
      target normalizer. Wired into test_mandelbrot_rl.py's rtac agent: the
      TD target is computed in raw units, then popart.update_and_rescale()
      normalizes it for the critic loss AND rescales all four value-head
      weight arrays (both online critics, both target nets) in the same call
      so nothing is left inconsistent by the stats update itself.
- [x] **DONE**: double critic. Two independently-initialized value heads
      (Wv_h/Wv_a and Wv_h2/Wv_a2, each with its own EMA target net), advantage
      and TD bootstrap both use min(v1,v2) per rtrl/rtac.py's
      reduce(torch.min, next_value_target) pattern. Both heads trained toward
      the same TD target every step.
- [ ] (moved to later-todo, see refactoring_todo.md: critic-through-trunk --
      needs a control-flow refactor to keep the previous step's autograd
      graph alive, not a small patch)
- [ ] (moved to later-todo, see refactoring_todo.md: replay buffer --
      needs a transition store plus a batch-axis audit of every op in the
      RL training loop, not a small patch)
- [ ] (moved to later-todo, see refactoring_todo.md: per-patch vision
      sequence + 2x2 spatial merge -- a real architecture change from the
      current mean-pooled-patch simplification)
- [x] **RESOLVED** (see section 1.2): embed_tokens/lm_head are two separate
      on-disk tensors in different shards in the real checkpoint. No dedup
      logic needed in streaming_sparsify -- both prune independently like
      any other 2-D tensor. Not a correctness requirement, just a possible
      future disk-space optimization if ever revisited.
- [ ] MoE expert-merge design note -> implementation (later-TODO)
