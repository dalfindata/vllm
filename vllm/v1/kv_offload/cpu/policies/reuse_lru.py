# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class ReuseLRUCachePolicy(CachePolicy):
    """Evict the coldest idle blocks first, with LRU as a tiebreaker.

    This policy keeps the simple OrderedDict layout of LRU but ranks eviction
    candidates by ``BlockStatus.reuse_score`` first. When multiple candidates
    have the same score, the oldest block is evicted first.
    """

    def __init__(self, cache_capacity: int):
        # cache_capacity unused but kept for a uniform constructor
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    @override
    def clear(self) -> None:
        self.blocks.clear()

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[int, int, OffloadKey, BlockStatus]] = []
        for order, (key, block) in enumerate(self.blocks.items()):
            if block.ref_cnt == 0 and key not in protected:
                candidates.append((block.reuse_score, order, key, block))

        if len(candidates) < n:
            return None

        selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:n]
        evicted: list[tuple[OffloadKey, BlockStatus]] = []
        for _, _, key, block in selected:
            del self.blocks[key]
            evicted.append((key, block))
        return evicted
