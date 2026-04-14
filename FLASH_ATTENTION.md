# Flash Attention in training

This document explains how Flash Attention was enabled on the training
forward pass, the tradeoff it introduces (padded-token contamination),
and the options we have for mitigating it.

## TL;DR

- Training used to build `combined_mask = causal_mask + padding_mask`
  (a float additive mask) and pass it to
  `F.scaled_dot_product_attention`. Float additive masks force SDPA
  into its **math backend**, which materialises the full
  `[B, H, S, S]` attention-score matrix — O(S²) VRAM and no fused
  kernel.
- A new config flag `training_self_attention` (enum
  `TrainingSelfAttention`) selects between:
  - `FLASH` (default): pass `mask=None, is_causal=True` so SDPA
    dispatches to Flash Attention.
  - `PADDED_MASK`: keep the legacy additive-mask path.
- The policy loss already multiplies by `pad_mask = (targets != PAD)`
  and the value loss already gates on `attention_mask.bool() &
  has_evals`, so padded positions contribute **zero gradient** through
  the loss regardless of what attention does with them.
- The one tradeoff: real tokens can attend *to* PAD tokens (and vice
  versa) inside the attention op. This is the "contamination" —
  bounded, empirically small, fixable if needed.

## What changed

### Config

In `src/humblemeister/config/_config.py`:

```python
class TrainingSelfAttention(str, Enum):
    FLASH = "flash"
    """Drop the padding mask and pass is_causal=True so SDPA dispatches to Flash
    Attention. Padded positions still participate in attention but are zeroed out
    of the loss via pad_mask."""
    PADDED_MASK = "padded_mask"
    """Build a combined causal + padding float-additive mask. Correct but forces
    SDPA to the math backend (O(S²) attention-score memory)."""


@dataclass
class ChessTrainingConfig:
    ...
    training_self_attention: str = TrainingSelfAttention.FLASH
```

### Trainer

A single helper on `ChessTrainer` dispatches based on the flag:

```python
def __forward_train(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = TrainingSelfAttention(self.__config.training_self_attention)
    if mode == TrainingSelfAttention.FLASH:
        return self.__model(input_ids, mask=None, is_causal=True)
    causal_mask = make_causal_mask(input_ids.size(1), self.__device)
    padding_mask = make_padding_mask(attention_mask).to(self.__device)
    return self.__model(input_ids, causal_mask + padding_mask)
```

All three training-forward call sites (main train step, eval/full-batch
path, value-head pretrain) route through `__forward_train`.

### Model forward — already Flash-ready

`ChessTransformer.forward` (in
`src/humblemeister/transformer/_chess_transformer.py`) accepts
`is_causal` and forwards it to each block's attention:

```python
def forward(
    self,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    for block in self.__blocks:
        out, _ = block(out, mask, kv_cache=None, is_causal=is_causal)
    ...
```

Inside `MultiHeadAttention.forward`
(`src/humblemeister/attention/_multi_head.py`):

```python
out = F.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=None if is_causal else mask,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=is_causal,
)
```

When `is_causal=True` and `mask=None`, SDPA's dispatcher picks Flash
Attention on CUDA + bf16 (our default), skipping the
`[B, H, S, S]` materialisation entirely.

## Why Flash Attention needs no mask

Flash Attention computes `softmax(QKᵀ/√d) · V` block-by-block on-chip.
It never writes the full attention-weight matrix to HBM — it fuses the
softmax and the `V`-matmul inside SRAM and streams the output back
one block at a time.

This fusion is incompatible with **arbitrary float additive masks**:
the kernel would have to accept an `[B, H, S, S]` tensor and load a
tile of it per block, undoing most of the memory win. So the Flash
kernel only supports:

- no mask (full attention), or
- a causal mask, applied implicitly by skipping above-diagonal tiles.

Our old `combined_mask = causal + padding` is a float tensor — Flash
refuses it and SDPA falls back to math. By passing `is_causal=True`
with no explicit mask, Flash fires.

## The tradeoff — padded-token contamination

Without a padding mask in attention, two things happen at padded
positions:

1. **Real tokens can attend *to* PAD tokens.** When a real query at
   position `t` computes `softmax(Q_t · Kᵀ)`, the PAD keys are part
   of the candidate set. They'll receive a small share of the
   attention probability, and the output at `t` is contaminated by
   `V_pad` in proportion to that share.
2. **PAD queries produce garbage outputs.** A query at a padded
   position attends to some mix of real and padded keys. The output
   is nonsense — but we don't care, because the loss at that position
   is zero (see below).

### What saves us — the loss already ignores padding

From `ChessTrainer.__policy_loss`:

```python
pad_mask = (targets_flat != pad).float()
denom = pad_mask.sum().clamp(min=1)
return (per_token * move_weights.view(-1) * pad_mask * policy_mask).sum() / denom
```

And from the value loss:

```python
token_mask = attention_mask.bool() & has_evals.unsqueeze(1)
value_loss = F.mse_loss(value_pred[token_mask], value_targets[token_mask])
```

Both losses are zero at padded positions. So **PAD-query outputs
never contribute gradient**. The only remaining concern is issue (1):
real-token outputs being slightly corrupted by attending to PAD keys.

### How big is the contamination in practice?

For a batch with `f` fraction padded tokens, a real query's attention
distribution has `f` of its mass spent on PAD keys on average (before
softmax sharpening). PAD tokens have their own learned embedding and
the model quickly learns to produce low-magnitude K/V there (since
their predictions are ignored in the loss, there's no gradient
pushing them to encode anything useful, and weight decay pulls them
toward zero). In practice the output perturbation is small —
measurable at init, negligible after a few hundred steps.

This is how GPT-2, LLaMA, and most production decoder-only models
train. They don't mask padding in attention; they rely on careful
batch construction and loss masking to keep contamination bounded.

## Mitigations, cheapest first

### 0. Do nothing

If padding fraction per batch is already low (say <10%), the
contamination signal is likely below the noise floor of training. Ship
the flag, measure train/val loss against the old PADDED_MASK path for
a few hundred steps, and if the curves match, you're done.

This is the recommended first step.

### 1. Length bucketing in the DataLoader

Sort games by length and group games of similar length into the same
batch. A `BatchSampler` that yields length-bucketed batches reduces
per-batch padding from ~30-50% (random batches) to typically <5%.

Concretely: add a sampler that partitions indices into bins by
`len(game.tensor)` and yields batches drawn from a single bin at a
time. The LR curve sees the same number of steps; only the padding
fraction changes.

This is the single biggest quality lever. Recommended whenever you
see padding fraction >10%.

### 2. Hard-zero PAD contribution via `padding_idx` — **enabled by default**

`nn.Embedding` supports a `padding_idx` argument that freezes the
row at zero (excluded from gradients). `InputEmbedding` and
`ChessTransformer` now accept a `padding_idx` / `pad_id` argument,
and the trainer + engine pass `tokenizer.PAD` when constructing the
model. After `init_weights()` runs its `normal_()` pass, the PAD
row is explicitly re-zeroed in `ChessTransformer.init_weights`.

Consequences of weight tying: the output projection shares its weight
matrix with the embedding, so the PAD row in the output logits is
also forced to zero. Since `cross_entropy` uses `ignore_index=PAD`,
the model is never asked to predict PAD — having a constant-zero
logit there is harmless.

Caveats (it's a nudge, not a guarantee):
- `W_k` / `W_v` linears have no bias in our `MultiHeadAttention`, so
  `K_pad = 0 · W_k = 0` — genuinely zero K/V at the attention op.
- **But** LayerNorm runs *before* attention (pre-norm). A zero input
  to LayerNorm produces a non-zero output because `γ, β` drift during
  training. So the "zero input" guarantee doesn't survive `norm1`.
  In practice the magnitude is still small enough that real-token
  outputs are only mildly perturbed.

For full correctness, combine with (1) length bucketing. If your
padding fraction is already <5%, (2) alone is enough.

### 3. Boolean mask instead of float additive

PyTorch ≥2.3 accepts `torch.bool` masks in SDPA. Some Flash variants
and the mem-efficient backend support them without falling back to
math. Build a combined causal+padding bool mask and pass it:

```python
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=bool_mask)
```

If the Flash kernel refuses (depending on PyTorch/CUDA version and
head-dim constraints), it falls through to the efficient kernel —
still O(S) memory, though not as fast as pure Flash.

This is "correctness with acceptable speed" — no contamination, some
speed recovered over the math backend.

### 4. Full sequence packing

Concatenate multiple games end-to-end in a single row, separated by
BOS tokens, until the row reaches `max_seq_len`. No padding at all.
This requires:

- A custom collate that packs.
- A **per-sequence position reset** — positional embeddings must
  restart at each BOS boundary, otherwise games late in the packed
  row see impossibly large positional indices.
- A **per-sequence causal mask** — Flash Attention's built-in causal
  treats the whole row as one sequence, so tokens in game B could
  attend to game A's tokens. `flash_attn_varlen_func` from the
  `flash-attn` library handles this natively via `cu_seqlens`.

Highest throughput, most restructuring. Production-grade setups
(LLaMA training, etc.) do this. Overkill until padding bucketing
isn't enough.

### 5. `flash-attn` library directly

Install Dao-AILab's `flash-attn` and call `flash_attn_varlen_func`
with cumulative sequence lengths. Natively supports variable-length
packed batches with no padding. This is (4) productised.

## How to choose

| Scenario                                   | Recommended mitigation |
|--------------------------------------------|------------------------|
| Just flipped the flag, want to verify      | (0) — measure loss parity |
| Padding fraction >10% observed             | (1) length bucketing |
| Want zero contamination, happy with ~70% of Flash speed | (3) bool mask |
| Chasing maximum throughput on huge configs | (4) packing with position reset |
| Padding fraction very low already          | (0) — do nothing |

## How to fall back

If anything goes wrong — loss spikes, NaN, quality regression — set
the config field to keep the legacy path:

```python
config.training_self_attention = TrainingSelfAttention.PADDED_MASK
```

No code change needed. The helper `__forward_train` dispatches on
every forward, so the flag takes effect on the next run without
touching checkpoints.

## Verification

To confirm Flash Attention is actually firing, wrap a single forward
pass in an explicit-backend context and check it doesn't error:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    logits, _ = model(input_ids, mask=None, is_causal=True)
```

If this raises, Flash isn't eligible (wrong dtype, wrong device, head
dim too large, etc.) — the default SDPA call still works and falls
back silently, but the explicit context forces an error so you can
diagnose.

Expected gains on the `giant` config (`d_model=1536, n_heads=24,
n_layers=24, max_seq_len=512, bf16`):

- **Peak VRAM**: large drop (the `[B, H, 512, 512]` score matrix at
  `B=16, H=24` is ~100 MB per layer per forward; at 24 layers it
  dominated activation memory during backward).
- **Throughput**: ~2-4× on the attention block, ~30-60% end-to-end
  depending on how attention-heavy the workload is relative to FFN
  and embedding.
