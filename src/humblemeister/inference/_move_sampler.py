from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import chess

from humblemeister.attention import make_causal_mask, make_padding_mask
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
    input_ids = torch.tensor(
        [move_history], dtype=torch.long, device=device
    )  # [1, seq_len]

    with torch.no_grad():
        causal_mask = make_causal_mask(input_ids.size(1), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits, _ = model(input_ids, causal_mask)

    policy_logits = logits[0, -1] / temperature  # [vocab_size]

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
        # build one sequence per legal move: move_history + [move_token]
        sequences = [
            torch.tensor(move_history + [tid.item()], dtype=torch.long, device=device)
            for tid in legal_token_ids
        ]
        # pad to same length — all sequences differ by at most 1 token here,
        # but pad_sequence handles the general case cleanly
        padded = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.PAD)
        # [n_legal, seq_len+1]

        attention_mask = (padded != tokenizer.PAD)
        padding_mask = make_padding_mask(attention_mask).to(device)
        causal_mask_v = make_causal_mask(padded.size(1), device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                _, value_preds = model(padded, causal_mask_v + padding_mask)
            # value_preds: [n_legal, seq_len+1]
            # extract value at the last real token position for each sequence
            seq_lens = attention_mask.sum(dim=1) - 1  # [n_legal]
            value_scores = value_preds[torch.arange(len(legal_moves), device=device), seq_lens]
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
