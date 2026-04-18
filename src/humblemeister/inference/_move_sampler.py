from __future__ import annotations

import chess
import torch
import torch.nn.functional as F

from humblemeister.attention import KVCache
from humblemeister.data import ChessTokenizer
from humblemeister.transformer import ChessTransformer


def _mask_by_value_gap(
    value_scores: torch.Tensor,
    blunder_threshold: float,
) -> torch.Tensor:
    """
    Return a boolean mask keeping moves within `blunder_threshold` of the best candidate.

    Args:
        value_scores:      [n_legal], side-to-move POV values in [-1, 1].
        blunder_threshold: max allowed gap below the best candidate.

    Returns:
        [n_legal] bool tensor; at least one entry is True (the best candidate).
    """
    best = value_scores.max()
    return value_scores >= (best - blunder_threshold)


def _find_mate_in_one(board: chess.Board, legal_moves: list[chess.Move]) -> int | None:
    """Return the index of the first mate-in-1 in `legal_moves`, or None.

    Short-circuits the model: a mate-in-1 is always the correct move, and the
    value head can be noisy enough that mate and 'just winning' get scored
    similarly — leaving the pick up to the policy, which may prefer the
    non-mate.
    """
    for i, move in enumerate(legal_moves):
        if not board.gives_check(move):
            continue
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return i
    return None


def pick_move_play(
    board: chess.Board,
    legal_moves: list[chess.Move],
    policy_weights: torch.Tensor,
    value_scores: torch.Tensor | None,
    blunder_threshold: float,
) -> tuple[int, chess.Move]:
    """
    Deterministic picker — mate-in-1 short-circuit, else value-gap mask + argmax
    over policy among survivors.

    When value_scores is None, degrades to pure argmax over policy_weights
    (still with the mate-in-1 short-circuit).
    """
    mate_idx = _find_mate_in_one(board, legal_moves)
    if mate_idx is not None:
        return mate_idx, legal_moves[mate_idx]

    if value_scores is None:
        idx = int(policy_weights.argmax().item())
        return idx, legal_moves[idx]

    mask = _mask_by_value_gap(value_scores, blunder_threshold)
    masked_policy = policy_weights.masked_fill(~mask, float("-inf"))
    idx = int(masked_policy.argmax().item())
    return idx, legal_moves[idx]


def pick_move_selfplay(
    board: chess.Board,
    legal_moves: list[chess.Move],
    policy_weights: torch.Tensor,
    value_scores: torch.Tensor | None,
    blunder_threshold: float,
) -> tuple[int, chess.Move]:
    """
    Stochastic picker — mate-in-1 short-circuit, else value-gap mask + sample
    from softmax(policy) among survivors.

    When value_scores is None, degrades to sampling from softmax(policy_weights)
    (still with the mate-in-1 short-circuit).
    """
    mate_idx = _find_mate_in_one(board, legal_moves)
    if mate_idx is not None:
        return mate_idx, legal_moves[mate_idx]

    if value_scores is None:
        masked_policy = policy_weights
    else:
        mask = _mask_by_value_gap(value_scores, blunder_threshold)
        masked_policy = policy_weights.masked_fill(~mask, float("-inf"))

    probabilities = F.softmax(masked_policy, dim=0)
    if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
        idx = int(torch.randint(len(legal_moves), (1,)).item())
    else:
        idx = int(torch.multinomial(probabilities, num_samples=1).item())
    return idx, legal_moves[idx]


def _pick(
    board: chess.Board,
    legal_moves: list[chess.Move],
    policy_weights: torch.Tensor,
    value_scores: torch.Tensor | None,
    blunder_threshold: float,
    is_self_play: bool,
) -> tuple[int, chess.Move]:
    if is_self_play:
        return pick_move_selfplay(
            board, legal_moves, policy_weights, value_scores, blunder_threshold
        )
    return pick_move_play(board, legal_moves, policy_weights, value_scores, blunder_threshold)


def sample_move(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    board: chess.Board,
    move_history: list[int],
    device: torch.device,
    temperature: float = 1.0,
    blunder_threshold: float = 0.25,
    is_self_play: bool = False,
    bf16: bool = True,
) -> chess.Move:
    """
    Sample the next move given the current board state and move history.

    Uses a full forward pass (no KV cache) over the complete token sequence,
    plus a batched full recompute over all resulting positions for the value
    head. See MOVE_PICKING.md for the two-stage mask-then-pick scheme.

    Args:
        model:             ChessTransformer in eval mode
        tokenizer:         ChessTokenizer
        board:             current board position
        move_history:      token IDs so far, including BOS
        device:            torch device
        temperature:       softmax temperature for policy sampling (1.0 = unmodified)
        blunder_threshold: max allowed value gap from best candidate; wider = more
                           permissive. In tanh-value units (0.25 ≈ 100 cp).
        is_self_play:      True → sample from policy among survivors (exploration);
                           False → argmax among survivors (deterministic play).
        bf16:              whether to use bfloat16 autocast
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("no legal moves available")

    # ------------------------------------------------------------------ #
    #  Step 1: policy logits at the current position                     #
    # ------------------------------------------------------------------ #
    input_ids = torch.tensor([move_history], dtype=torch.long, device=device)  # [1, seq_len]

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            logits, _ = model(input_ids, is_causal=True)

    policy_logits = logits[0, -1].clone() / temperature  # [vocab_size]
    del logits, _

    # ------------------------------------------------------------------ #
    #  Step 2: legal move token IDs and their policy scores              #
    # ------------------------------------------------------------------ #
    legal_token_ids = torch.tensor(
        [tokenizer.encode_move(m) for m in legal_moves],
        dtype=torch.long,
        device=device,
    )  # [n_legal]

    legal_policy = policy_logits[legal_token_ids]  # [n_legal]

    # ------------------------------------------------------------------ #
    #  Step 3: value scores for each resulting position                  #
    # ------------------------------------------------------------------ #
    value_input = torch.tensor(
        [move_history + [int(tid.item())] for tid in legal_token_ids],
        dtype=torch.long,
        device=device,
    )  # [n_legal, seq_len+1]

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            value_logits, value_preds = model(value_input, is_causal=True)
    del value_logits, value_input
    value_scores = value_preds[:, -1].clone()  # [n_legal], White POV
    del value_preds

    # flip sign for Black to move — model always evaluates from White's perspective,
    # but we want to reward moves that are good for the side to move
    if board.turn == chess.BLACK:
        value_scores = -value_scores

    # ------------------------------------------------------------------ #
    #  Step 4: mask blunders and pick                                    #
    # ------------------------------------------------------------------ #
    _, result = _pick(
        board, legal_moves, legal_policy, value_scores, blunder_threshold, is_self_play
    )
    return result


def sample_move_kv_cache(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    board: chess.Board,
    move_history: list[int],
    device: torch.device,
    temperature: float = 1.0,
    blunder_threshold: float = 0.25,
    is_self_play: bool = False,
    bf16: bool = True,
    cache: KVCache | None = None,
    cache_tokens: int = 0,
) -> tuple[chess.Move, KVCache]:
    """
    Sample the next move using KV caching.

    Tokens in move_history[:cache_tokens] are assumed to already be present in
    `cache`; only move_history[cache_tokens:] are processed to update it.
    When cache is None / cache_tokens is 0, the full history is prefilled from
    scratch. Pass the returned KVCache back on subsequent calls to make each
    step O(new tokens) rather than O(full history).

    Value scores are produced via a single batched generate_step over all legal
    move tokens at once, using an expanded copy of the prefilled cache. See
    MOVE_PICKING.md for the mask-then-pick scheme.

    Args:
        model:             ChessTransformer in eval mode
        tokenizer:         ChessTokenizer
        board:             current board position
        move_history:      all token IDs so far, including BOS
        device:            torch device
        temperature:       softmax temperature for policy sampling (1.0 = unmodified)
        blunder_threshold: max allowed value gap from best candidate; wider = more
                           permissive. In tanh-value units (0.25 ≈ 100 cp).
        is_self_play:      True → sample from policy among survivors (exploration);
                           False → argmax among survivors (deterministic play).
        bf16:              whether to use bfloat16 autocast
        cache:             existing KVCache covering move_history[:cache_tokens];
                           None means start from an empty cache.
        cache_tokens:      number of tokens from move_history already in cache.

    Returns:
        (move, updated_cache) — the sampled legal move and the KVCache updated
        to cover all of move_history.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("no legal moves available")
    if not move_history:
        raise ValueError("move_history must contain at least the BOS token")

    # ------------------------------------------------------------------ #
    #  Step 1: process any tokens not yet in the cache                   #
    # ------------------------------------------------------------------ #
    if cache is None:
        cache = KVCache()
        cache_tokens = 0

    tokens_to_process = move_history[cache_tokens:]
    if not tokens_to_process:
        raise ValueError("no new tokens to process — cache_tokens equals len(move_history)")

    logits: torch.Tensor | None = None
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            for token_id in tokens_to_process:
                token = torch.tensor([[token_id]], dtype=torch.long, device=device)  # [1, 1]
                logits, _, cache = model.generate_step(token, cache)
            if logits is None:
                raise RuntimeError("Unexpected: no tokens were processed")
            policy_logits = logits[0, 0] / temperature  # [vocab_size]

    # ------------------------------------------------------------------ #
    #  Step 2: legal move token IDs and their policy scores              #
    # ------------------------------------------------------------------ #
    legal_token_ids = torch.tensor(
        [tokenizer.encode_move(m) for m in legal_moves],
        dtype=torch.long,
        device=device,
    )  # [n_legal]

    policy_weights = policy_logits[legal_token_ids]  # [n_legal]

    # ------------------------------------------------------------------ #
    #  Step 3: value scores via one batched generate_step                #
    # ------------------------------------------------------------------ #
    n_legal = len(legal_moves)

    # expand the prefilled cache from batch=1 to batch=n_legal
    expanded_layers = [
        type(layer)(
            k=layer.k.expand(n_legal, -1, -1, -1).contiguous(),
            v=layer.v.expand(n_legal, -1, -1, -1).contiguous(),
        )
        for layer in cache.layers
    ]
    expanded_cache = KVCache(layers=expanded_layers)

    move_tokens = legal_token_ids.unsqueeze(1)  # [n_legal, 1]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            _, value_preds, _ = model.generate_step(move_tokens, expanded_cache)
    value_scores = value_preds.squeeze(1)  # [n_legal], White POV

    if board.turn == chess.BLACK:
        value_scores = -value_scores

    # ------------------------------------------------------------------ #
    #  Step 4: mask blunders and pick                                    #
    # ------------------------------------------------------------------ #
    _, result = _pick(
        board, legal_moves, policy_weights, value_scores, blunder_threshold, is_self_play
    )
    return result, cache
