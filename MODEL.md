# HumbleMeister — Model & Training Documentation

## Overview

HumbleMeister is a transformer-based chess engine that treats chess as a **language modelling problem**. A game of chess is represented as a sequence of UCI move tokens, and the model is trained to predict the next move given all previous moves — identical to next-token prediction in a language model. There is no board evaluation function, no tree search, and no hand-crafted features. The model learns chess entirely from sequences of moves.

In addition to move prediction, the model has a **value head** that predicts the Stockfish evaluation of each board position, expressed as a number in `[-1, 1]` from White's perspective. This dual-output design supports both supervised learning from human games and reinforcement learning from self-play.

---

## Tokenization

**File:** `src/humblemeister/data/_tokenizer.py`

### Vocabulary

A game is encoded as a flat sequence of integer token IDs:

```
[BOS, move_1, move_2, ..., move_N, EOS]
```

The vocabulary contains:

| Token | ID | Description |
|-------|----|-------------|
| `<PAD>` | 0 | Padding (ignored in loss) |
| `<BOS>` | 1 | Beginning of sequence |
| `<EOS>` | 2 | End of sequence |
| UCI moves | 3… | All legal UCI moves including promotions |

All possible moves are enumerated at construction time by iterating over every pair of squares on the board, filtering out same-square moves, and adding promotion variants for pawn moves that reach the back rank (with 4 promotion pieces: queen, rook, bishop, knight). This produces a vocabulary of approximately **3,000 tokens**.

### Versioning & `vocab_hash`

`ChessTokenizer.VERSION` is bumped whenever the tokenizer output semantics
change (not for pure code refactors). `ChessTokenizer.vocab_hash()` returns a
short SHA-256 of the sorted `(uci, id)` pairs — a stable fingerprint of the
current vocabulary.

The hash is stamped into every shard written by `ChessGameBank.save()` and
checked on `load()`. Shards whose hash doesn't match the current tokenizer
are rejected; shards written before the stamp existed are accepted with a
warning so older corpora can still be used.

### Legal Move Masking

During inference, illegal moves are masked out before sampling. `get_legal_mask` returns an additive logit mask of shape `[vocab_size]`:
- `0.0` for legal moves
- `-inf` for all other tokens

Adding this mask to the raw logits before softmax drives the probability of illegal moves to exactly zero, guaranteeing that the engine never outputs an illegal move.

---

## Model Architecture

**Files:** `src/humblemeister/transformer/`, `src/humblemeister/attention/`, `src/humblemeister/embedding/`

### High-Level Structure

```
Input token IDs  [batch, seq_len]
        │
        ▼
┌─────────────────────┐
│   InputEmbedding    │  token embedding × √d_model + learned positional encoding
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  TransformerBlock   │  × n_layers
│  ┌───────────────┐  │
│  │  LayerNorm    │  │  pre-norm
│  │  MHA          │  │  multi-head self-attention (Flash Attention)
│  │  Residual     │  │
│  │  LayerNorm    │  │  pre-norm
│  │  FeedForward  │  │  Linear → GELU → Linear
│  │  Residual     │  │
│  └───────────────┘  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    LayerNorm        │  final normalisation
└─────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌──────────────┐         ┌───────────────────────┐
│ Output head  │         │    Value head         │
│ Linear       │         │ Linear(d→d/2)         │
│ (weight-tied)│         │ GELU                  │
│ [batch,seq,V]│         │ Linear(d/2→1)         │
└──────────────┘         │ Tanh                  │
  policy logits          │ [batch, seq]  ∈[-1,1] │
                         └───────────────────────┘
                           position evaluation
```

### Input Embedding

`InputEmbedding` combines two components:
1. **Token embedding** — `nn.Embedding(vocab_size, d_model)`, scaled by `√d_model` following the original transformer paper. This keeps embedding magnitudes compatible with the positional encoding values.
2. **Learned positional encoding** — `nn.Embedding(max_seq_len, d_model)`, one learned vector per position up to `max_seq_len=512`. Unlike sinusoidal positional encoding (which is also implemented but not used by default), learned encodings can adapt to the specific sequential structure of chess games.

### Transformer Block (Pre-Norm)

Each block uses **pre-norm** (LayerNorm before attention and feed-forward, not after). This differs from the original "post-norm" transformer and is standard in modern LLMs because it provides more stable gradients in deep networks.

```
x → LayerNorm → MultiHeadAttention → dropout → + x  (residual)
  → LayerNorm → FeedForward        → dropout → + x  (residual)
```

### Multi-Head Attention

`MultiHeadAttention` uses fused projections (single `W_q`, `W_k`, `W_v` matrix for all heads) and delegates the actual attention computation to `F.scaled_dot_product_attention` — PyTorch's implementation of **Flash Attention**, which never materialises the full `[batch, heads, seq, seq]` attention matrix in VRAM. This is critical for long chess game sequences.

During training, a **combined causal + padding mask** is applied:
- **Causal mask**: prevents position `t` from attending to positions `> t` (autoregressive constraint)
- **Padding mask**: prevents attending to `PAD` tokens in shorter sequences within a batch

```
combined_mask = causal_mask + padding_mask
```

Both masks are additive logit masks (`0.0` or `-inf`), compatible with Flash Attention's `attn_mask` argument.

### Feed-Forward Network

Two-layer MLP with GELU activation:
```
x → Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
```

With `d_ff = 4 × d_model` following standard transformer convention.

### Policy Head (Weight-Tied)

The output projection `Linear(d_model, vocab_size, bias=False)` shares its weight matrix with the token embedding layer:

```python
self.__output.weight = self.__input_embedding.token_embedding.embedding.weight
```

This **weight tying** means the same matrix is used to both embed tokens into `d_model` space (at the input) and to project `d_model` back to vocabulary logits (at the output). It reduces parameters significantly and is standard practice in language models.

### Value Head

A small 2-layer MLP (Linear → GELU → Linear → Tanh) sitting on top of the transformer's final hidden states:

```
hidden [batch, seq, d_model]
    → Linear(d_model, d_model // 2)
    → GELU
    → Linear(d_model // 2, 1)
    → Tanh
    → squeeze → [batch, seq]   ∈ [-1, 1]
```

Output is in `[-1, 1]` from White's perspective at each position in the sequence. The Tanh ensures the output is bounded. This head is trained separately from the policy head and uses Stockfish centipawn evaluations as targets.

### Weight Initialisation

All parameters with `dim > 1` are initialised with:
```
Normal(mean=0.0, std=0.02)
```

Output projection matrices (`W_o`) are additionally scaled down by `1 / √(2 × n_layers)` following GPT-2 practice, to prevent the residual stream from growing too large in deep networks.

### Model Configurations

| Config | d_model | n_heads | n_layers | d_ff | Parameters |
|--------|---------|---------|----------|------|------------|
| tiny | 128 | 4 | 4 | 512 | ~5M |
| small | 256 | 4 | 4 | 1024 | ~18M |
| medium | 512 | 8 | 8 | 2048 | ~85M |
| large | 768 | 12 | 12 | 3072 | ~270M |
| huge | 1024 | 16 | 16 | 4096 | ~500M |
| **giant** | **1536** | **24** | **24** | **6144** | **~687M** |
| uber | 2048 | 32 | 32 | 8192 | ~1.5B |

---

## Data Pipeline

**Files:** `src/humblemeister/data/`

### Game Bank

`ChessGameBank` manages the corpus of training games. It loads games from PGN archives (`.zip` or `.7z`), stores them as lists of `chess.Move` objects, and supports shuffled streaming. Each bank record contains:
- `moves` — list of `chess.Move`
- `outcome` — `1.0` (white win), `0.0` (black loss), `0.5` (draw)
- `move_weights` — per-move Stockfish advantage weights (computed offline)
- `value_evals` — per-position Stockfish evaluations (computed offline)

### Offline Stockfish Evaluation

Before training, the entire game bank is evaluated with Stockfish at `depth=5`. This is done in parallel using forked worker processes (24 workers by default), each owning one persistent Stockfish process.

#### `convert_games()` — end-to-end offline pipeline

`ChessGameBank.convert_games(source_path, ...)` is the batch pipeline used to
build a game bank from raw archives. It:

1. Discovers every `.pgn`, `.zip`, and `.7z` under `source_path`.
2. Spawns a process pool (one worker per file) with `fork` context.
3. Each worker:
   - Parses its file (ELO-filtering optional) to a list of games.
   - Owns its own `chess.engine.SimpleEngine` process — no evaluator is
     shared across workers.
   - Computes per-position `value_evals` and per-move `move_weights` using
     the same formulas as the online evaluator.
   - Writes shards directly to disk using the input filename as the shard
     stem (e.g. `lichess_2023_01.pgn` → `lichess_2023_01_0000.pt`,
     `lichess_2023_01_0001.pt`, …) so shards from different workers
     cannot collide without any central coordination.
4. The parent process only receives `(stem, n_games, n_shards)` tuples,
   not the games themselves — memory stays constant regardless of dataset
   size.
5. After all workers finish, the parent renames per-file shards to
   sequential `shard_XXXX.pt` and writes a `meta.json`.

This is how the 7M-game ELO 2200+ corpus is prepared for the 688M model.

#### `fill_value_evals()` — backfill for legacy shards

Early shards were produced before the value head existed and therefore have
no `value_evals` field. `ChessGameBank.fill_value_evals(path, n_workers)`
iterates those shards, runs Stockfish on any games with missing evals, and
rewrites the shard in place. Games that already have evals are skipped.
After the backfill the in-memory bank is reloaded from disk. This avoids
re-running the full `convert_games` pipeline when only the value signal is
missing.

#### Move Weights

For each game, all `N+1` board positions (before + after each move) are evaluated. Per-move advantage is computed as:

```
adv[t] = -eval[t] - eval[t-1]
```

Where `eval[t]` is centipawns from the side-to-move's perspective at position `t`. The negation of `eval[t]` converts from the opponent's perspective (post-move) back to the mover's perspective.

Advantages are Z-scored within the game (mean 0, std 1), then passed through softmax with temperature, and scaled so the mean weight is `1.0`:

```
weights = softmax(adv / temperature) × N
```

This produces a per-move importance weight. Better moves get higher weights, worse moves get lower weights, and the average weight across a game is always `1.0` so the overall loss scale is preserved.

#### Value Evaluations

Per-position Stockfish evaluations are converted to White's perspective and normalised to `[-1, 1]` using tanh:

```
white_pov[t] = eval[t]  if t is even (White to move)
             = -eval[t] if t is odd  (Black to move)

value_eval[t] = tanh(white_pov[t] / 400.0)
```

The divisor `400` is chosen so that a 1-pawn advantage (~100cp) maps to `tanh(0.25) ≈ 0.24`, a 4-pawn advantage maps to `tanh(1.0) ≈ 0.76`, and a mate (capped at ±2000cp) maps to `tanh(5.0) ≈ 1.0`.

### Dataset & Collation

`ChessDataset` holds a list of `GameRecord` objects for the current epoch. `collate()` pads variable-length sequences and performs the sequence shift for autoregressive training:

```
game_tensor  = [BOS, m1, m2, ..., mN, EOS]
input_ids    = [BOS, m1, m2, ..., mN]       ← fed into model
targets      = [     m1, m2, ..., mN, EOS]  ← prediction targets
```

Move weights and value evaluations are aligned with `input_ids` and padded with zeros. A `has_value_evals` boolean mask per game tracks which games have real Stockfish evaluations vs. zero fallback, to correctly gate the value loss.

Bank games longer than `config.max_moves` (default 500) are skipped during
epoch assembly in `__generate_from_bank`. This caps sequence length for the
positional encoding (`max_seq_len = 512`) and trims the long tail of
extreme games.

---

## Training

**File:** `src/humblemeister/trainer/_trainer.py`

### Overview

Training proceeds in two phases:

```
Phase 1: Supervised learning on bank games (no self-play)
         ↓
Value head pretraining (frozen transformer body)
         ↓
Phase 2: Joint policy + value training, with optional self-play curriculum
```

### Epoch Structure

Each epoch consists of two steps:

```
1. generate_games(epoch)   → fills dataset with N games
2. train_on_games(epoch)   → one pass over the dataset
```

### Batch Assembly (Length Bucketing)

Training batches are assembled by `LengthBucketBatchSampler`
(`src/humblemeister/data/_bucket_sampler.py`), which guarantees every
game inside a GPU batch has the **exact same sequence length** — so the
collator pads nothing. This is what allows Flash Attention to run
without the padded-token contamination described in
[FLASH_ATTENTION.md](FLASH_ATTENTION.md).

Per epoch, the sampler:

1. Draws `n_games` indices uniformly without replacement from the full
   corpus. Every game has equal probability of being included in the
   epoch's sample regardless of its length.
2. Groups the drawn indices by `len(game.tensor)`.
3. For each length-group, yields `⌊n_L / batch_size⌋` full batches plus
   one tail batch of size `n_L mod batch_size` (if non-zero). Tail
   batches are single-length too — smaller than `batch_size` but zero
   padding.
4. Shuffles the full batch list so the trainer doesn't see all-short
   then all-long batches in sequence.

Determinism: the RNG stream is `bucket_sampler_seed + epoch`, so
`(seed, epoch)` fully determines the epoch's sample and batch order.
Resume needs no extra checkpoint state — passing the current epoch
number reconstructs the sampler exactly.

The sampler is controlled by two config flags:

- `batch_length_sampling` (enum, default `BUCKETED`) — set to `RANDOM`
  to bypass bucketing and fall back to `shuffle=True`. The `BUCKETED`
  setting only activates when `training_self_attention == FLASH`; under
  `PADDED_MASK` the padding mask already handles mixed-length batches
  correctly, so the flag is ignored.
- `bucket_sampler_seed` (int, default `0`) — seed base for per-epoch
  sampling. Change across runs if you want different random subsets of
  the corpus visited in each training run.

Trade-offs:
- **Pro**: zero intra-batch padding → Flash Attention runs on real
  tokens only, no contamination; VRAM for the attention op matches
  exactly the real sequence length.
- **Con**: GPU batch size is no longer fixed. Long-tail length classes
  with few games per epoch produce small tail batches (e.g. 1–10 games)
  that under-utilise the GPU. If tail overhead becomes significant, a
  ±k tail-merging fallback can be added as a follow-up — see mitigation
  (1) in [FLASH_ATTENTION.md](FLASH_ATTENTION.md).

### Loss Function

#### Policy Loss

Weighted cross-entropy, with per-sample label smoothing that depends on
whether the game came from the bank or from self-play:

```python
# ADVANTAGE_WEIGHTED (default): bank smoothed, self-play unsmoothed
smooth = F.cross_entropy(logits_flat, targets_flat,
                         ignore_index=pad, label_smoothing=0.1, reduction="none")
raw    = F.cross_entropy(logits_flat, targets_flat,
                         ignore_index=pad, label_smoothing=0.0, reduction="none")
per_token = torch.where(is_sp_flat, raw, smooth)

pad_mask = (targets_flat != pad).float()
denom = pad_mask.sum().clamp(min=1)
loss  = (per_token * move_weights.view(-1) * pad_mask * policy_mask).sum() / denom
```

Bank games get `label_smoothing=0.1` (human moves are plausible but not
uniquely optimal); self-play games get `label_smoothing=0.0` because the
advantage weight already handles move-quality weighting and smoothing on
top of a blunder would bias the gradient toward the bad move.

A second mode (`VALUE_ONLY`) zeros out self-play policy contributions
entirely so only the value head trains on them.

**Full details, rationale, and branching logic** for the two modes
(`VALUE_ONLY` vs `ADVANTAGE_WEIGHTED`) are documented separately in
[SELF_PLAY_LOSS.md](SELF_PLAY_LOSS.md).

#### Value Loss

MSE between the value head output and Stockfish evaluations, computed
only on positions that have real evaluations (games with `value_evals`):

```python
token_mask = attention_mask.bool() & has_evals.unsqueeze(1)
value_loss = F.mse_loss(value_pred[token_mask], value_targets[token_mask])
```

Both bank and self-play games contribute to the value loss: self-play
games always have evals (produced by the same Stockfish pipeline that
builds their `move_weights`). Bank games contribute iff they went
through the evaluator.

#### Combined Loss

```
loss = policy_loss + value_loss_weight × value_loss
```

With `value_loss_weight = 0.5` by default. This combined loss backpropagates through both the value head and the transformer body, causing the representations to evolve to support both objectives simultaneously.

#### Outcome Weighting (Self-Play Only)

When self-play games are present in the dataset, the combined loss is scaled by the game outcome:

```
scaled_loss = loss × (1.0 + outcome_scale × outcome_weights)
```

Where:
- `outcome_weights = mean(outcomes × 2 - 1)` maps wins/draws/losses to `[+1, 0, -1]`
- `outcome_scale` ramps from 0 to 1 over `outcome_warmup` steps

This reinforces patterns from winning self-play games and suppresses patterns from losing games. It only applies to self-play games because bank games are worth learning from equally regardless of outcome.

### Mixed Precision (bf16)

When `config.bf16=True` (default) and the device is CUDA, the forward
and loss computations run inside `torch.autocast("cuda",
dtype=torch.bfloat16)`. bfloat16 halves activation memory vs fp32 and
— unlike fp16 — has the full fp32 exponent range, so no `GradScaler` is
needed. Optimizer states and master weights remain fp32.

### Optimiser & Schedule

- **Optimiser**: AdamW with `weight_decay=0.1`, `grad_clip=1.0`
- **Learning rate**: Cosine annealing with linear warmup, parameterised
  by `start_epoch` so the curve can be rebuilt at any resume point.

```python
def lr_lambda(step: int) -> float:
    epoch = start_epoch + step
    if epoch < warmup_epochs:
        return epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
```

`n_epochs` here is the run's `end_epoch`, which is stored on the
checkpoint so that `resume()` can reconstruct the exact same cosine
curve without replaying the scheduler step-by-step.

### `run()` vs `resume()`

Training has two entry points with complementary responsibilities:

- `trainer.run(max_epochs=..., ...)` — **always starts fresh**. Clears
  the checkpoints directory, builds a cosine schedule over
  `[0, end_epoch]`, and trains from epoch 0.
- `trainer.resume(start_epoch=None, end_epoch=None, max_epochs=None, ...)`
  — loads the latest checkpoint, derives the current epoch from it, and
  rebuilds the scheduler anchored at that epoch so the LR curve is
  exactly what it would have been had training never stopped. Each
  argument optionally overrides the value persisted in the checkpoint:
  - `start_epoch` — override the resume epoch.
  - `end_epoch` — override the target end epoch (LR will re-fit this
    new horizon).
  - `max_epochs` — alternative to `end_epoch`, specifies how many more
    epochs to run from `start_epoch`.

There is no resume-from-run(): `run()` is deliberately one-shot.

### Value Head Pretraining

Before self-play begins, the value head is pretrained independently:

1. Freeze all transformer parameters
2. Train only the value head parameters (`~1.18M` for giant) with AdamW `lr=1e-3` (default, overridable)
3. Each epoch loads `n_games` from the bank, runs the full frozen forward pass, computes MSE loss against `value_evals`
4. Restore all parameters to trainable
5. Save checkpoint

This gives the value head a meaningful starting point before it needs to provide signal for self-play outcome weighting. Without pretraining, the value head would be random at the start of self-play, producing noisy reinforcement signals.

Expected loss trajectory:
- **Naive baseline** (predict 0 always): MSE ≈ 0.15–0.25 (variance of target distribution)
- **After pretraining** on frozen transformer: plateaus around 0.13–0.14
- **During joint training**: continues improving as representations evolve, targeting < 0.1

### Self-Play Curriculum

Self-play is introduced gradually:

```
Epoch 0              self_play_start_epoch         +ramp_epochs
 │                          │                           │
 ▼                          ▼                           ▼
[──── 100% bank games ──────][── ramp 0%→30% ──────────][── 30% self-play ──→]
```

At each epoch, the dataset is filled with:
- `n_bank = n_games × (1 - ratio)` games from the bank
- `n_self_play = n_games × ratio` games from self-play

The ratio can also be overridden directly in `run()` via `self_play_min` / `self_play_max`, which linearly interpolates over the current run's epoch range, or disabled entirely with `disable_selfplay=True`.

### Streaming Training (large models)

For the larger configs (`giant`, `uber`) the full epoch dataset does
not fit in VRAM alongside the model + optimizer state. When
`config.streaming=True`, each epoch is broken into chunks of
`streaming_chunk_size` games that are generated → evaluated → consumed
→ discarded one at a time:

```python
self.__optimizer.zero_grad()
for b_this, sp_this in chunks:
    self.__dataset.clear()
    # generate b_this bank + sp_this self-play games (eval mode)
    # evaluate with Stockfish
    # forward + backward; do NOT step yet
    ...
self.__optimizer.step()  # single step per epoch, across all chunks
```

Key properties:
- Gradients accumulate across all chunks; there is exactly one
  `optimizer.step()` per epoch regardless of how many chunks an epoch
  is split into.
- Peak memory holds at most `streaming_chunk_size` games + model +
  optimizer state. The entire pretraining corpus never needs to be
  materialised at once.
- Self-play games within a chunk are evaluated synchronously before the
  next chunk starts — no async pipeline.

Self-play-only chunk generation additionally respects
`self_play_max_moves` (hard draw cap), and uses
`self_play_blunder_threshold` as the value-gap mask that filters
candidate moves before the policy sample — obvious blunders (by the
value head's own evaluation) are excluded before sampling. See
[MOVE_PICKING.md](MOVE_PICKING.md).

#### Heap fragmentation mitigation

The per-epoch alloc/free churn (generate games → build dataset → discard)
causes glibc's `malloc` to accumulate unreturned arenas: RSS grows
monotonically even though Python objects are being freed. Every 50
epochs the trainer runs:

```python
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
```

`malloc_trim(0)` asks glibc to return freed-but-cached pages to the OS.
Without this, multi-day training runs drift into swap on memory-tight
hosts. The call is wrapped in a `try/except` so non-Linux / non-glibc
environments silently skip it.

### Self-Play Generation

Two backends are available:

- **`SelfPlayGPU`** (default, used by the giant config) — batched GPU
  generation with a single shared model. Games run in lockstep inside a
  GPU batch, with per-game KV caches and finished games dropped from the
  batch by row.
- **`SelfPlayCPU`** — multi-process CPU generation. Workers fork from a
  saved checkpoint, each loads its own `ChessModel` on CPU, and plays its
  share of games via `ChessGame`. Useful when the GPU is saturated by
  training or when scaling CPU cores is easier than VRAM.

Both share the same Stockfish post-processing path (`AsyncBatchEvaluator`)
for move-weight computation and, where applicable, outcome determination
at `max_moves`.

Games are generated in batches (`self_play_batch_size`). The batch size on the last batch is capped to exactly the number of games still needed, so self-play never overshoots the configured ratio.

#### Full Forward Recompute (default, `self_play_kv_cache=False`)

Each active game is processed independently via `sample_move`. A full forward pass is run over the complete token sequence `[BOS, m1, ..., m_{t-1}]` to obtain logits at the last position:

```
Step t (per game):
  logits, _ = model.forward([BOS, m1, ..., m_{t-1}], causal_mask)
  next_logit = logits[0, -1]  ← last position only
```

This is O(n²) compute per game (full attention over the growing sequence) but uses a flat and predictable amount of VRAM regardless of sequence length. Games are processed sequentially, not batched together.

#### KV Cache (optional, `self_play_kv_cache=True`)

Each game maintains a persistent `KVCache` across moves. `sample_move_kv_cache` is called with the current cache and the number of tokens already cached, so only the new tokens since the last call are processed:

```
Step t (per game):
  move, cache = sample_move_kv_cache(..., cache=cache, cache_tokens=prev_len)
  # internally: only processes move_history[prev_len:] through generate_step
```

This is O(new tokens) per step — typically O(1) after the first call — but VRAM grows linearly with sequence length × n_layers × 2 (K+V). For the giant model (24 layers, d_model=1536), this can be significant for long games.

#### Stockfish Evaluation of Self-Play Games

After generation, self-play games are evaluated by `AsyncBatchEvaluator`, a pool of 24 persistent worker processes each running a Stockfish instance. Workers are forked once at the start of `__generate_self_play` and kept alive for the duration — Stockfish startup cost is paid only once. Jobs (batch file paths) are sent over pipes; workers evaluate the batch in-place and signal completion with `OK\n`.

Generation and evaluation overlap: while the GPU generates the next batch, workers evaluate the previous batch concurrently.

### TensorBoard Metrics

The following scalars are logged every `log_every` epochs:

| Scalar | Description |
|--------|-------------|
| `loss/train` | Combined policy + value loss |
| `loss/value` | Value head MSE loss |
| `games/avg_length` | Mean game length in moves |
| `games/win_rate` | Fraction of White wins |
| `games/draw_rate` | Fraction of draws |
| `games/loss_rate` | Fraction of Black wins |
| `curriculum/self_play` | Configured self-play ratio this epoch |
| `lr` | Current learning rate |
| `perf/epoch_s` | Wall time per epoch (seconds) |
| `perf/eval_ms_per_game` | Mean Stockfish eval time per bank game (ms) |
| `self_play/gen_time_s` | GPU time spent generating self-play games |
| `self_play/eval_time_s` | Time blocked on Stockfish workers (submit + drain) |
| `self_play/games_generated` | Raw games produced by the generator |
| `self_play/games_included` | Games that passed filtering and entered the dataset |

The gap between `self_play/games_generated` and `self_play/games_included` reflects games dropped during Stockfish evaluation or due to invalid move sequences.

### Checkpointing

Checkpoints are saved:
- Every `checkpoint_every=50` epochs (scheduled)
- At the end of value head pretraining
- At the end of each `run()` call
- Immediately when `LossBreakthroughDetector` fires (see below)

Each checkpoint contains:
- `model_state` — full model state dict including value head
- `optimizer_state` — Adam moment estimates
- `epoch` — last completed epoch
- `end_epoch` — the target end epoch of the run that produced this
  checkpoint (used by `resume()` to rebuild the matching LR curve)
- `loss` — combined loss at that epoch

Note: `scheduler_state` is **not** stored. The scheduler is
reconstructed deterministically from `epoch` + `end_epoch` +
`warmup_epochs` + base `lr`, which avoids the subtle bugs that arise
from replaying `LambdaLR`'s internal step counter across runs with
different horizons.

The last `keep_last_n=20` checkpoints are retained; older ones are pruned automatically. Checkpoints are saved in safetensors format for the final exported model.

Loading is tolerant of missing keys (e.g. loading a pre-value-head checkpoint into a model with a value head) via `strict=False`, initialising missing parameters from the already-computed random initialisation.

#### Loss Breakthrough Detection

`LossBreakthroughDetector` monitors training loss across two rolling windows:
- **Old window** (16 epochs): baseline mean and MQD (standard deviation)
- **Recent window** (4 epochs): current trend mean

A breakthrough is triggered when `recent_mean < old_mean - 3 × old_MQD` — i.e. the recent trend has durably shifted below the baseline by more than 3 standard deviations. This catches genuine grokking events (sustained drops) rather than transient spikes, and triggers an immediate out-of-schedule checkpoint.

---

## Inference

**File:** `src/humblemeister/_engine.py`

Inference is split into two classes so that a single set of weights can serve
many concurrent games (e.g. the Gradio app running N parallel sessions):

- **`ChessModel`** — holds the `ChessTransformer` weights, the `ChessTokenizer`
  and the device. It exposes a single `sample(...)` method that runs one
  inference step. One `ChessModel` per model artifact, shared across all games.
- **`ChessGame`** — one per game. Owns the `chess.Board`, the move-history
  token list, the player colour, a per-game `KVCache`, and the sampling
  hyperparameters (`temperature`, `blunder_threshold`, `is_self_play`,
  `use_kv_cache`). Delegates the actual forward pass to the shared
  `ChessModel`.

`ChessGame.sample_move()` dispatches to one of two sampling paths depending on
the constructor flag `use_kv_cache`:

#### Without KV cache (`use_kv_cache=False`, default)

Calls `sample_move`, which runs a full forward pass over the complete move history on every turn:

1. Enumerate `board.legal_moves` and encode each as a token ID
2. Run `model.forward([BOS, m1, ..., m_{t-1}])` — full sequence forward pass
3. Extract logits at the last position; slice to legal token IDs only
4. Batched forward over all resulting positions for value-head scores
5. Mask blunders by value gap, then pick (argmax for play, sample for self-play)

#### With KV cache (`use_kv_cache=True`)

Calls `sample_move_kv_cache` with a persistent `KVCache` stored on the
`ChessGame`. The cache is initialised on `start_game()` / `reset()` and
accumulates across moves:

- Only tokens added since the previous call are processed — O(1) per turn once warmed up
- The cache is reset at the start of each new game

#### Value-Gap Masking

Both sampling paths score every legal move via the value head — a single batched `generate_step` over all legal moves (sharing the same cache prefix), with the sign flipped for Black to move so scores are always from the side-to-move perspective.

Candidate moves whose value score is more than `blunder_threshold` below the best candidate are then excluded via a boolean mask before the policy head picks. In tanh-value units, `0.15 ≈ 60 cp` (strict, good for play against a human), `0.25 ≈ 100 cp` (default for self-play). See [MOVE_PICKING.md](MOVE_PICKING.md) for the full design.

Picking depends on the `is_self_play` flag:

- `is_self_play=False` (human play) — argmax of policy among survivors. Deterministic.
- `is_self_play=True` (self-play generation) — sample from softmax(policy) among survivors. Preserves exploration while excluding blunders.

Temperature controls policy sharpness: `< 1.0` concentrates mass on the top policy moves, `> 1.0` flattens. Only affects the stochastic (`is_self_play=True`) path — argmax is insensitive to temperature.

Models are loaded via `ChessModel.load(path, device)`, which auto-detects
whether `path` is a safetensors directory (produced by `save_model`) or a
`.pt` checkpoint.

---

## Appendix: Key Hyperparameters (Giant Config)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 1536 | Hidden dimension |
| `n_heads` | 24 | Attention heads |
| `n_layers` | 24 | Transformer blocks |
| `d_ff` | 6144 | Feed-forward inner dim |
| `max_seq_len` | 512 | Max game length in tokens |
| `dropout` | 0.1 | Dropout rate |
| `train_batch_size` | 64 | Games per gradient step |
| `n_games` | 16384 | Games loaded per epoch |
| `max_moves` | 500 | Bank games longer than this are skipped during epoch assembly |
| `lr` | 3e-4 | Peak learning rate |
| `warmup_epochs` | 50 | LR warmup duration |
| `label_smoothing` | 0.1 | Cross-entropy smoothing |
| `weight_decay` | 0.1 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient norm clipping |
| `value_loss_weight` | 0.5 | Value vs policy loss ratio |
| `self_play_start_epoch` | 1000 | When self-play begins |
| `self_play_max_ratio` | 0.30 | Max self-play fraction |
| `self_play_batch_size` | 256 | Games generated in parallel per SelfPlayGPU batch |
| `self_play_loss_mode` | `advantage_weighted` | See [SELF_PLAY_LOSS.md](SELF_PLAY_LOSS.md) |
| `self_play_blunder_threshold` | 0.25 | Value-gap mask for candidate moves during self-play (see [MOVE_PICKING.md](MOVE_PICKING.md)) |
| `self_play_max_moves` | 84 | Hard draw cap for self-play games |
| `self_play_kv_cache` | True | Use KV cache during self-play generation |
| `streaming` | True | Chunked epoch, one optimizer step per epoch |
| `streaming_chunk_size` | 3072 | Games per streaming chunk |
| `bf16` | True | bfloat16 mixed precision |
| `stockfish_depth` | 5 | Stockfish search depth |
| `stockfish_workers` | 24 | Parallel Stockfish processes |
| `advantage_temperature` | 1.0 | Sharpness of per-move weight distribution |
| `outcome_warmup` | 20000 | Steps to ramp outcome weighting |
| `keep_last_n` | 20 | Number of recent checkpoints to retain |
| `training_self_attention` | `flash` | Flash Attention path (see [FLASH_ATTENTION.md](FLASH_ATTENTION.md)); `padded_mask` falls back to the legacy additive-mask path |
| `batch_length_sampling` | `bucketed` | Batch assembly strategy; `bucketed` yields same-length batches (zero padding), `random` uses standard shuffled loader. Only active when `training_self_attention = flash`. |
| `bucket_sampler_seed` | 0 | Seed base for the per-epoch uniform sample used by bucketed batching; combined with epoch number |
