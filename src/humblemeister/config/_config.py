from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class SelfPlayLossMode(str, Enum):
    VALUE_ONLY = "value_only"
    """Zero the policy loss on self-play games; only the value head trains on them."""
    ADVANTAGE_WEIGHTED = "advantage_weighted"
    """Policy gradient weighted by per-move Stockfish advantages (no label smoothing)."""


class TrainingSelfAttention(str, Enum):
    FLASH = "flash"
    """Drop the padding mask and pass is_causal=True so SDPA dispatches to Flash
    Attention. Padded positions still participate in attention but are zeroed out
    of the loss via pad_mask."""
    PADDED_MASK = "padded_mask"
    """Build a combined causal + padding float-additive mask. Correct but forces
    SDPA to the math backend (O(S²) attention-score memory)."""


class BatchLengthSampling(str, Enum):
    BUCKETED = "bucketed"
    """Sample n_games games uniformly at random from the corpus, group them by
    exact sequence length, and emit batches where every game shares the same
    length. Eliminates intra-batch padding entirely, which is what Flash
    Attention needs to run without padded-token contamination. Only active when
    training_self_attention == FLASH; ignored otherwise (the PADDED_MASK path
    already handles arbitrary-length batches correctly via the padding mask)."""
    RANDOM = "random"
    """Standard shuffle=True DataLoader — each batch is a random draw from the
    corpus, padded to the longest game in the batch. Use for parity testing or
    when debugging bucketing behaviour."""


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ChessTrainingConfig:
    model_name: str

    # model architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    train_batch_size: int = 256

    max_seq_len: int = 512
    dropout: float = 0.1
    bf16: bool = True  # bfloat16 mixed precision — halves activation memory, no GradScaler needed
    training_self_attention: str = TrainingSelfAttention.FLASH
    """How attention is invoked during training forward passes. FLASH enables the
    Flash Attention kernel (recommended); PADDED_MASK preserves the legacy
    additive-mask path."""
    batch_length_sampling: str = BatchLengthSampling.BUCKETED
    """How batches are assembled. Only takes effect when training_self_attention
    is FLASH. Under PADDED_MASK this flag is ignored and the trainer uses
    standard random shuffling (the padding mask makes bucketing unnecessary)."""
    bucket_sampler_seed: int = 0
    """Seed base for the per-epoch uniform sample used by BUCKETED sampling.
    Combined with the epoch number so each epoch draws a different subset
    deterministically."""

    # training
    n_games: int = 512  # total games loaded per epoch
    n_epochs: int = 2000
    lr: float = 3e-4
    device: str = _get_device()

    max_moves: int = 500
    warmup_epochs: int = 50
    outcome_warmup: int = 20000

    # self-play curriculum — ramps in gradually after self_play_start_epoch
    self_play_start_epoch: int = 1000  # all bank games before this
    self_play_ramp_epochs: int = 400  # epochs to ramp from 0% → max ratio
    self_play_max_ratio: float = 0.3  # cap at 30% self-play games
    self_play_batch_size: int = 8  # games generated in parallel per SelfPlayGPU batch
    self_play_workers: int = 1  # number of CPU processes for SelfPlay generation

    # stockfish evaluation
    use_stockfish: bool = True
    stockfish_path: str = "stockfish"
    stockfish_depth: int = 5  # depth 5 ≈ 1-5ms/position
    stockfish_workers: int = 24  # parallel engine processes for board evaluation
    advantage_temperature: float = 1.0  # sharpness of per-move weight distribution
    value_loss_weight: float = 0.5  # relative weight of value loss vs policy loss
    self_play_kv_cache: bool = True  # use KV cache during self-play generation
    self_play_blunder_threshold: float = 0.25
    """Value-gap threshold for masking candidate moves during self-play (tanh-value units;
    0.25 ≈ 100 cp). Moves whose value score is more than this below the best candidate are
    excluded before the policy sample. See MOVE_PICKING.md."""
    self_play_max_moves: int = 84  # hard draw cap for self-play games
    self_play_loss_mode: str = (
        SelfPlayLossMode.ADVANTAGE_WEIGHTED
    )  # how self-play games contribute to the loss
    streaming: bool = False  # stream games in chunks instead of generating all at once
    streaming_chunk_size: int = 64  # games per streaming chunk (generation + grad accumulation)

    # checkpointing
    checkpoint_dir: str = "env/checkpoints"
    checkpoint_every: int = 50  # save every N epochs
    keep_last_n: int = 20  # keep only the last N checkpoints

    # logging
    log_dir: str = "env/logs"
    log_every: int = 1

    @classmethod
    def tiny(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 128
        result.n_heads = 4
        result.n_layers = 4
        result.d_ff = 512
        result.train_batch_size = 512
        return result

    @classmethod
    def small(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 256
        result.n_heads = 4
        result.n_layers = 4
        result.d_ff = 1024
        result.train_batch_size = 512
        return result

    @classmethod
    def medium(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 512
        result.n_heads = 8
        result.n_layers = 8
        result.d_ff = 2048
        result.train_batch_size = 256
        result.n_games = 1024
        return result

    @classmethod
    def large(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 768
        result.n_heads = 12
        result.n_layers = 12
        result.d_ff = 3072
        result.train_batch_size = 64
        result.n_games = 2048
        return result

    @classmethod
    def huge(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 1024
        result.n_heads = 16
        result.n_layers = 16
        result.d_ff = 4096
        result.train_batch_size = 64
        result.n_games = 2048
        return result

    @classmethod
    def giant(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 1536
        result.n_heads = 24
        result.n_layers = 24
        result.d_ff = 6144
        result.train_batch_size = 64
        result.self_play_kv_cache = True
        result.streaming = True
        result.streaming_chunk_size = 3072
        result.self_play_batch_size = 256
        result.n_games = 16384
        return result

    @classmethod
    def uber(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 2048
        result.n_heads = 32
        result.n_layers = 32
        result.d_ff = 8192
        result.train_batch_size = 2
        result.self_play_kv_cache = True
        result.streaming = True
        result.streaming_chunk_size = 96
        result.self_play_batch_size = 96
        result.n_games = 8192
        return result

    def save(self, file_path: str | Path) -> None:
        with Path(file_path).open(mode="w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def from_file(cls, file_path: str | Path) -> ChessTrainingConfig:
        with Path(file_path).open() as f:
            d = json.load(f)
            if not isinstance(d, dict):
                raise TypeError(
                    f"Cannot load ChessTrainingConfig: expected dict, got {type(d).__name__}"
                )
            if "model_name" not in d:
                raise ValueError("Cannot load ChessTrainingConfig: model_name is not defined")
            model_name = d["model_name"]
            del d["model_name"]
            return cls(model_name, **d)
