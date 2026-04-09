from __future__ import annotations

from typing import Optional

import chess
import torch

NEG_INF = float("-inf")


class ChessTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    SPECIAL_TOKENS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}

    __move_to_id: dict[str, int]
    __id_to_move: dict[int, str]
    __vocab_size: int

    def __init__(self) -> None:
        self.__move_to_id = {}
        self.__id_to_move = {}

        idx = len(self.SPECIAL_TOKENS)

        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                if from_sq == to_sq:
                    continue

                uci = chess.Move(from_sq, to_sq).uci()
                if uci not in self.__move_to_id:
                    self.__move_to_id[uci] = idx
                    self.__id_to_move[idx] = uci
                    idx += 1

                # only rank-7→rank-8 (white) or rank-2→rank-1 (black) can promote,
                # and only if the file distance is at most 1 (straight or diagonal capture)
                from_rank = chess.square_rank(from_sq)
                to_rank = chess.square_rank(to_sq)
                from_file = chess.square_file(from_sq)
                to_file = chess.square_file(to_sq)
                is_promotion_square = abs(from_file - to_file) <= 1 and (
                    (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0)
                )
                if is_promotion_square:
                    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        uci_promo = chess.Move(from_sq, to_sq, promotion=promo).uci()
                        if uci_promo not in self.__move_to_id:
                            self.__move_to_id[uci_promo] = idx
                            self.__id_to_move[idx] = uci_promo
                            idx += 1

        self.__vocab_size = idx

    @property
    def vocab_size(self) -> int:
        return self.__vocab_size

    def encode_move(self, move: chess.Move) -> int:
        return self.__move_to_id[move.uci()]

    def decode_move(self, token_id: int) -> Optional[chess.Move]:
        if token_id in (self.PAD, self.BOS, self.EOS):
            return None
        return chess.Move.from_uci(self.__id_to_move[token_id])

    def encode_game(self, moves: list[chess.Move]) -> list[int]:
        return [self.BOS] + [self.encode_move(m) for m in moves] + [self.EOS]

    def encode_game_tensor(self, moves: list[chess.Move]) -> torch.Tensor:
        return torch.tensor(self.encode_game(moves), dtype=torch.long)

    def decode_game(self, token_ids: list[int]) -> list[chess.Move]:
        return [m for t in token_ids if (m := self.decode_move(t)) is not None]

    def get_legal_mask(self, board: chess.Board) -> torch.Tensor:
        """Returns a logit mask tensor — 0.0 for legal moves, -inf for illegal."""
        mask = torch.full((self.__vocab_size,), NEG_INF)
        for move in board.legal_moves:
            mask[self.encode_move(move)] = 0.0
        return mask
