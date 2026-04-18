# Move picking — blending policy and value heads

This document describes how the inference layer selects a legal move
given the transformer's two output heads (policy logits and value
score), why the previous additive-blend scheme was replaced, and the
split between self-play and human-play picking strategies.

## TL;DR

- The old scheme blended additively: `policy_logit + value_weight *
  tanh_value`. This mixes an **unbounded** logit with a **[-1, 1]**
  value score — the two signals live on incompatible scales, and the
  result is sensitive to arbitrary choices of `value_weight`.
- We replace it with **value-gap masking**: the value head acts as an
  eligibility filter (candidate moves whose post-move evaluation is
  more than `blunder_threshold` worse than the best candidate are
  excluded), and the policy head chooses among the survivors.
- Two variants of `pick_move`:
  - `pick_move_play` — deterministic. Mask by value gap, then
    `argmax(policy)` among survivors. Used for games against a human.
  - `pick_move_selfplay` — stochastic. Mask by value gap, then sample
    from `softmax(policy)` among survivors. Used during self-play data
    generation to preserve exploration.
- `value_weight` is removed from the inference path entirely.

## Background — why the previous blend was wrong

The model has two output heads sharing a backbone:

- **Policy head**: produces unbounded logits over the full token
  vocabulary. `policy_weights[i]` is the logit for the i-th legal
  move's token.
- **Value head**: produces a scalar in `[-1, 1]` via `tanh`, roughly
  `tanh(centipawns / 400)` — a White-POV evaluation of the resulting
  position after the candidate move.

The old `pick_move` blended them linearly:

```python
final_scores = policy_weights + value_weight * value_scores
probabilities = F.softmax(final_scores, dim=0)
idx = torch.multinomial(probabilities, 1)
```

Two concrete problems:

1. **Scale mismatch.** Policy logits routinely span 10+ units between
   top and bottom candidates; value scores differ by at most 2
   (worst-case `-1 → 1`). So the policy always dominates unless
   `value_weight` is cranked to ~5+ — at which point the blend starts
   overriding sensible policy preferences too. There's no principled
   value of `value_weight` that balances the two.
2. **Tactical blindness.** Observed in self-play: the model misses
   hanging pieces (e.g. doesn't capture a hanging queen), and plays
   near-uniformly at random in many positions. The value head knows
   the queen is hanging — its score for `QxQ` is hugely better than
   the alternatives — but that signal gets swamped by noisy policy
   logits after the additive blend.

Training loss context: policy cross-entropy ≈ 2.5 (perplexity ~12,
i.e. genuinely noisy); value loss ≈ 0.01 (very sharp). The value
head is the stronger signal and should be doing more work.

## The new scheme — mate-in-1 short-circuit, then value-gap masking

Before either the value or policy head is consulted, both pickers scan
legal moves for a **mate-in-1** and return it immediately if found:

```python
def _find_mate_in_one(board: chess.Board, legal_moves: list[chess.Move]) -> int | None:
    for i, move in enumerate(legal_moves):
        if not board.gives_check(move):
            continue
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return i
    return None
```

Why this is necessary despite a "sharp" value head:

- A mate-in-1 results in a position with value ≈ +1 from the mover's
  perspective. But the value head is trained on real games, and
  terminal mate positions are rare — it may emit something like 0.5
  rather than ~1.0 for a mate.
- Another "just winning" move (e.g. one that wins the queen but doesn't
  mate) can land in the same value range, so mate and non-mate both
  survive the value-gap mask.
- Among survivors the pick is by POLICY. Policy is noisy (train CE ≈
  2.5), and nothing in it specifically rewards mate over "obviously
  winning". Observed in practice: the engine skipped a mate-in-1 in
  favour of another winning move.

The short-circuit fixes this directly — a mate-in-1 is always the
right move, the `board.gives_check(m)` filter keeps the cost near-zero
(most legal moves don't give check, so we skip the push/pop), and
finding the mate is cheap compared to a forward pass.

No extension to mate-in-2+ — that requires search, which is out of
scope. Mate-in-1 is the one case where we have a ground-truth
shortcut and the value/policy heads visibly fail.

## Value-gap masking

When no mate-in-1 exists, both picker variants apply the same eligibility
filter:

```python
def _mask_by_value_gap(
    value_scores: torch.Tensor,     # [n_legal], in [-1, 1], side-to-move POV
    blunder_threshold: float,
) -> torch.Tensor:
    """
    Keep moves whose value is within `blunder_threshold` of the best candidate.
    Returns a boolean mask of shape [n_legal]; at least one entry is True.
    """
    best = value_scores.max()
    return value_scores >= (best - blunder_threshold)
```

Why value-gap specifically (as opposed to softmax-rank or top-K):

- **Chess-interpretable.** `blunder_threshold = 0.2` corresponds to a
  value drop of ~80 cp via `tanh(cp/400)` — i.e. "reject moves that
  are more than ~80 cp worse than the best candidate". This is the
  natural unit in which chess engines reason about blunders.
- **Position-invariant.** Top-K-by-count is fragile in low-legal-move
  positions (K=1 out of 4 legal moves is very different from K=10 out
  of 40). A gap threshold behaves sensibly regardless of the legal
  move count.
- **Softmax is a no-op for ranking.** Softmaxing the value scores
  before thresholding doesn't change the order, and makes the
  threshold depend on the temperature, the legal-move count, and the
  score spread — all irrelevant to what "blunder" means.

### `pick_move_play` — deterministic, for human play

```python
def pick_move_play(
    board: chess.Board,
    legal_moves: list[chess.Move],
    policy_weights: torch.Tensor,
    value_scores: torch.Tensor,
    blunder_threshold: float,
) -> tuple[int, chess.Move]:
    mate_idx = _find_mate_in_one(board, legal_moves)
    if mate_idx is not None:
        return mate_idx, legal_moves[mate_idx]
    mask = _mask_by_value_gap(value_scores, blunder_threshold)
    masked_policy = policy_weights.masked_fill(~mask, float("-inf"))
    idx = int(masked_policy.argmax().item())
    return idx, legal_moves[idx]
```

Mate-first, then pure argmax among value-gap survivors. Repeat-position
deterministic. Good for playing against a human — the engine should be
consistent, avoid obviously dumb moves, and never miss a mate-in-1.

### `pick_move_selfplay` — stochastic, for data generation

```python
def pick_move_selfplay(
    board: chess.Board,
    legal_moves: list[chess.Move],
    policy_weights: torch.Tensor,
    value_scores: torch.Tensor,
    blunder_threshold: float,
) -> tuple[int, chess.Move]:
    mate_idx = _find_mate_in_one(board, legal_moves)
    if mate_idx is not None:
        return mate_idx, legal_moves[mate_idx]
    mask = _mask_by_value_gap(value_scores, blunder_threshold)
    masked_policy = policy_weights.masked_fill(~mask, float("-inf"))
    probabilities = F.softmax(masked_policy, dim=0)
    idx = int(torch.multinomial(probabilities, num_samples=1).item())
    return idx, legal_moves[idx]
```

Same mate short-circuit and mask, but sample from the policy
distribution restricted to survivors. Preserves exploration within the
space of non-blunders. The value head prunes catastrophes; the policy
head steers among the plausible options.

### Fallback when `value_scores is None`

Both variants degrade gracefully to policy-only when no value scores
are available (e.g. because `value_weight == 0.0` used to skip the
batched value forward pass). In practice we always have value scores
now — the engine always runs the value-scoring forward pass — but the
contract is:

- `pick_move_play` → `argmax(policy_weights)`
- `pick_move_selfplay` → sample from `softmax(policy_weights)`

## What's removed

**`value_weight` is gone.** It's dropped from:

- `pick_move` signature(s) in `src/humblemeister/inference/_move_sampler.py`
- `sample_move` / `sample_move_kv_cache` parameters
- `ChessGame.__init__` in `src/humblemeister/_engine.py`
- The Gradio app (`app.py`)

Replaced by `blunder_threshold`, which is conceptually cleaner:
the value head's job is "veto blunders", not "nudge the policy".

## Configuration

New field on `ChessTrainingConfig` in `src/humblemeister/config/_config.py`:

```python
self_play_blunder_threshold: float = 0.25
"""Value-gap threshold for masking candidate moves during self-play.
Moves whose value score is more than this much below the best candidate
are excluded before policy sampling. In tanh-value units: 0.25 ≈ 100cp."""
```

Defaults:

- **Self-play**: `blunder_threshold = 0.25` (~100cp). Loose enough to
  preserve learning signal from "slightly suboptimal but plausible"
  moves; strict enough to stop the model reinforcing hanging-piece
  blunders.
- **Human play**: the Gradio app exposes `blunder_threshold` as a
  float ≥ 0 input. A lower default (e.g. `0.15` ≈ 60cp) is more
  conservative — the engine refuses to trade off even small amounts
  of eval against policy preference.

### Tuning guide

| `blunder_threshold` | tanh-value gap | Approx. centipawns | Behavior                                     |
|---------------------|----------------|--------------------|----------------------------------------------|
| 0.05                | 0.05           | ~20 cp             | Nearly pure value-argmax; policy barely matters |
| 0.15                | 0.15           | ~60 cp             | Conservative play; rejects even small concessions |
| 0.25                | 0.25           | ~100 cp            | Good self-play default; allows exploration    |
| 0.50                | 0.50           | ~220 cp            | Only excludes outright blunders              |
| ∞                   | —              | —                  | No value masking; back to policy-only         |

## How this interacts with self-play training

The self-play loss (documented in [SELF_PLAY_LOSS.md](SELF_PLAY_LOSS.md))
already weights self-play moves by Stockfish-computed advantages, so
bad moves contribute near-zero gradient. What changes with the new
picker:

- **Fewer blunders in generated games.** The value-head mask prevents
  the picker from ever *choosing* a move with a big value gap. So
  Stockfish advantages see games that already have fewer catastrophic
  errors.
- **More learning signal per game.** Games end in checkmate or draw
  more often (rather than being decided by early hung pieces), so the
  advantage signal is distributed over more tactically relevant
  moves.
- **Exploration preserved.** Sampling-within-survivors means the
  training distribution still covers a range of plausible moves —
  self-play doesn't collapse to the same game being played over and
  over.

This is the setup the advantage-weighted self-play objective was
designed for: "most moves are plausible; a few are brilliant; the
model learns to concentrate weight on the brilliant ones." Without
the value-head mask, "most moves are plausible" wasn't actually true
at the current policy loss (~2.5 CE, perplexity ~12).

## Why not also change training?

The policy head is trained on cross-entropy against human moves (bank)
and advantage-weighted self-play moves. That objective is
well-understood and the value-head mask only affects *inference* —
which positions end up in the self-play dataset, not how gradients
flow through the policy head.

If the mask ends up being too strict and starves the policy head of
diverse training data, we'd loosen `blunder_threshold`. If it's too
loose and self-play games still contain blunders, tighten it. No
training-loop changes either way.

## Files touched

- `src/humblemeister/inference/_move_sampler.py` — new `pick_move_play`,
  `pick_move_selfplay`, `_mask_by_value_gap`, `_find_mate_in_one`;
  `sample_move` and `sample_move_kv_cache` lose `value_weight`, gain
  `blunder_threshold` and `is_self_play`. Both pickers take `board` so
  the mate short-circuit can run before consulting either head.
- `src/humblemeister/_engine.py` — `ChessGame` drops `value_weight`,
  adds `blunder_threshold` and `is_self_play`. `ChessModel.load` now
  filters unknown config fields so older checkpoints still load.
- `src/humblemeister/config/_config.py` — new
  `self_play_blunder_threshold` field on `ChessTrainingConfig`
  (replaces `self_play_value_weight`).
- `src/humblemeister/trainer/_self_play_gpu.py` (and CPU equivalent) —
  pass `config.self_play_blunder_threshold` through to the sampler. The
  CPU worker sets `is_self_play=True` on the `ChessGame` it constructs.
- `src/humblemeister/trainer/_trainer.py` — `run()` / `resume()` /
  `__train_loop` rename `self_play_value_weight` →
  `self_play_blunder_threshold`.
- `app.py` — replace the `value_weight` Gradio input with a
  `blunder_threshold` input, same validation shape (float ≥ 0).
