from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from ._dataset import ChessDataset


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Yields batches where every game shares the same sequence length.

    Per epoch:
      1. Draw `n_games` indices uniformly without replacement from the dataset.
      2. Group the drawn indices by `len(game.tensor)`.
      3. For each length-group, emit `floor(n_L / batch_size)` full batches plus
         one tail batch of size `n_L mod batch_size` (if non-zero). Every batch
         contains games of exactly one length, so the collator pads nothing.
      4. Shuffle the full batch list before iterating.

    Determinism: the RNG stream is `seed + epoch`, so `(seed, epoch)` fully
    determines which games are drawn and in what order. No state is carried
    across epochs — call `set_epoch(e)` before each iteration.
    """

    __dataset: ChessDataset
    __batch_size: int
    __n_games: int
    __seed: int
    __epoch: int
    __lengths: list[int]
    __cached_batches: list[list[int]] | None

    def __init__(
        self,
        dataset: ChessDataset,
        batch_size: int,
        n_games: int,
        seed: int,
        epoch: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if n_games <= 0:
            raise ValueError(f"n_games must be positive, got {n_games}")
        self.__dataset = dataset
        self.__batch_size = batch_size
        self.__n_games = min(n_games, len(dataset))
        self.__seed = seed
        self.__epoch = epoch
        self.__lengths = [len(g.tensor) for g in dataset.games]
        self.__cached_batches = None

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.__epoch:
            self.__epoch = epoch
            self.__cached_batches = None

    def __build_batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.__seed + self.__epoch)
        n_total = len(self.__dataset)

        # uniform sample without replacement
        perm = torch.randperm(n_total, generator=generator).tolist()
        sampled = perm[: self.__n_games]

        # group by exact length
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in sampled:
            groups[self.__lengths[idx]].append(idx)

        batches: list[list[int]] = []
        for length in sorted(groups.keys()):
            group = groups[length]
            # shuffle within group so the same length class doesn't always see
            # the same game ordering across epochs
            order = torch.randperm(len(group), generator=generator).tolist()
            shuffled = [group[i] for i in order]
            for start in range(0, len(shuffled), self.__batch_size):
                batches.append(shuffled[start : start + self.__batch_size])

        # shuffle batch order so the trainer doesn't see all-short-then-all-long
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[i] for i in batch_order]

    def __iter__(self) -> Iterator[list[int]]:
        if self.__cached_batches is None:
            self.__cached_batches = self.__build_batches()
        yield from self.__cached_batches
        # next iteration rebuilds — protects against accidental stale reuse
        self.__cached_batches = None

    def __len__(self) -> int:
        if self.__cached_batches is None:
            self.__cached_batches = self.__build_batches()
        return len(self.__cached_batches)
