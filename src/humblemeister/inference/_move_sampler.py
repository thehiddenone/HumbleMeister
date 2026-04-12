from __future__ import annotations

import chess
import torch
import torch.nn.functional as F

from humblemeister.attention import KVCache
from humblemeister.data import ChessTokenizer
from humblemeister.transformer import ChessTransformer


def sample_move(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    board: chess.Board,
    move_history: list[int],
    device: torch.device,
    temperature: float = 1.0,
    value_weight: float = 0.0,
    bf16: bool = True,
) -> chess.Move:
    """
    Sample the next move given the current board state and move history.

    Uses a full forward pass (no KV cache) over the complete token sequence.
    When value_weight > 0, each legal move is additionally scored by running
    a batched forward pass over all resulting positions and blending the
    value head output with the policy logits.

    Args:
        model:         ChessTransformer in eval mode
        tokenizer:     ChessTokenizer
        board:         current board position
        move_history:  token IDs so far, including BOS
        device:        torch device
        temperature:   softmax temperature for policy sampling (1.0 = unmodified)
        value_weight:  λ — weight of value head signal relative to policy logits.
                       0.0 = pure policy, higher values shift toward value guidance.
        bf16:          whether to use bfloat16 autocast

    Returns:
        A legal chess.Move sampled from the blended distribution.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("no legal moves available")

    # ------------------------------------------------------------------ #
    #  Step 1: policy logits at the current position                      #
    # ------------------------------------------------------------------ #
    input_ids = torch.tensor([move_history], dtype=torch.long, device=device)  # [1, seq_len]

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            logits, _ = model(input_ids, is_causal=True)

    policy_logits = logits[0, -1].clone() / temperature  # [vocab_size]
    del logits, _

    # ------------------------------------------------------------------ #
    #  Step 2: legal move token IDs and their policy scores               #
    # ------------------------------------------------------------------ #
    legal_token_ids = torch.tensor(
        [tokenizer.encode_move(m) for m in legal_moves],
        dtype=torch.long,
        device=device,
    )  # [n_legal]

    legal_policy = policy_logits[legal_token_ids]  # [n_legal]

    # ------------------------------------------------------------------ #
    #  Step 3: value scores for each resulting position (when λ > 0)      #
    # ------------------------------------------------------------------ #
    if value_weight > 0.0:
        # all sequences are the same length: move_history + one legal move token
        value_input = torch.tensor(
            [move_history + [int(tid.item())] for tid in legal_token_ids],
            dtype=torch.long,
            device=device,
        )  # [n_legal, seq_len+1]

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
                value_logits, value_preds = model(value_input, is_causal=True)
        del value_logits, value_input
        # value_preds: [n_legal, seq_len+1] — take the last position
        value_scores = value_preds[:, -1].clone()
        del value_preds
        # [n_legal], each in [-1, 1] from White's perspective

        # flip sign for Black to move — model always evaluates from White's perspective,
        # but we want to reward moves that are good for the side to move
        if board.turn == chess.BLACK:
            value_scores = -value_scores

        final_scores = legal_policy + value_weight * value_scores
    else:
        final_scores = legal_policy

    # ------------------------------------------------------------------ #
    #  Step 4: sample from the distribution over legal moves              #
    # ------------------------------------------------------------------ #
    probs = F.softmax(final_scores, dim=0)

    if torch.isnan(probs).any() or torch.isinf(probs).any():
        # fall back to uniform over legal moves
        idx = int(torch.randint(len(legal_moves), (1,)).item())
    else:
        idx = int(torch.multinomial(probs, num_samples=1).item())

    return legal_moves[idx]


def sample_move_kv_cache(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    board: chess.Board,
    move_history: list[int],
    device: torch.device,
    temperature: float = 1.0,
    value_weight: float = 0.0,
    bf16: bool = True,
    cache: KVCache | None = None,
    cache_tokens: int = 0,
) -> tuple[chess.Move, KVCache]:
    """
    Sample the next move using KV caching.

    Tokens in move_history[:cache_tokens] are assumed to already be present in
    `cache`; only move_history[cache_tokens:] are processed to update it.
    When cache is None / cache_tokens is 0, the full history is prefilled from
    scratch.  Pass the returned KVCache back on subsequent calls to make each
    step O(new tokens) rather than O(full history).

    Policy logits are read from the last processed token — no extra forward
    pass needed.  When value_weight > 0, a single batched generate_step scores
    all legal move resulting positions simultaneously using the shared cache.

    Args:
        model:         ChessTransformer in eval mode
        tokenizer:     ChessTokenizer
        board:         current board position
        move_history:  all token IDs so far, including BOS
        device:        torch device
        temperature:   softmax temperature for policy sampling (1.0 = unmodified)
        value_weight:  λ — weight of value head signal relative to policy logits.
                       0.0 = pure policy, higher values shift toward value guidance.
        bf16:          whether to use bfloat16 autocast
        cache:         existing KVCache covering move_history[:cache_tokens];
                       None means start from an empty cache.
        cache_tokens:  number of tokens from move_history already in cache.

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
    #  Step 1: process any tokens not yet in the cache                    #
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
            # logits: [1, 1, vocab_size] — distribution over the next token
            policy_logits = logits[0, 0] / temperature  # [vocab_size]

    # ------------------------------------------------------------------ #
    #  Step 2: legal move token IDs and their policy scores               #
    # ------------------------------------------------------------------ #
    legal_token_ids = torch.tensor(
        [tokenizer.encode_move(m) for m in legal_moves],
        dtype=torch.long,
        device=device,
    )  # [n_legal]

    legal_policy = policy_logits[legal_token_ids]  # [n_legal]

    # ------------------------------------------------------------------ #
    #  Step 3: value scores via one batched generate_step (when λ > 0)   #
    # ------------------------------------------------------------------ #
    if value_weight > 0.0:
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

        # one generate_step over all legal move tokens at once
        move_tokens = legal_token_ids.unsqueeze(1)  # [n_legal, 1]
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
                _, value_preds, _ = model.generate_step(move_tokens, expanded_cache)
        # value_preds: [n_legal, 1] → squeeze to [n_legal]
        value_scores = value_preds.squeeze(1)

        if board.turn == chess.BLACK:
            value_scores = -value_scores

        final_scores = legal_policy + value_weight * value_scores
    else:
        final_scores = legal_policy

    # ------------------------------------------------------------------ #
    #  Step 4: sample from the distribution over legal moves              #
    # ------------------------------------------------------------------ #
    probs = F.softmax(final_scores, dim=0)

    if torch.isnan(probs).any() or torch.isinf(probs).any():
        idx = int(torch.randint(len(legal_moves), (1,)).item())
    else:
        idx = int(torch.multinomial(probs, num_samples=1).item())

    return legal_moves[idx], cache
