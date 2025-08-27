import random
import asyncio
import json
import affine as af
from collections import deque
from typing import Any, Deque, List, Optional, Dict

class R2BufferedDataset:
    def __init__(
        self,
        dataset_name: str,
        total_size: int = 0,
        buffer_size: int = 100,
        max_batch: int = 10,
        seed: Optional[int] = None,
        config: str = "default",
        split: str = "train",
    ):
        self.dataset_name   = dataset_name
        self.config         = config
        self.split          = split
        self.buffer_size    = buffer_size
        self.max_batch      = max_batch
        self._rng           = random.Random(seed)

        self._buffer: Deque[Any] = deque()
        self._lock   = asyncio.Lock()
        self._fill_task = None

        # Postgres scan state
        self._db_offset: int = 0
        self.total_size = total_size

    async def _read_next_rows(self, desired: int) -> list[Any]:
        # Try reading from current offset; if empty and offset > 0, wrap to 0 once
        rows: List[Any] = await af.select_dataset_rows(
            dataset_name=self.dataset_name,
            config=self.config,
            split=self.split,
            limit=desired,
            offset=self._db_offset,
            include_index=False,
        )
        if not rows and self._db_offset:
            self._db_offset = 0
            rows = await af.select_dataset_rows(
                dataset_name=self.dataset_name,
                config=self.config,
                split=self.split,
                limit=desired,
                offset=self._db_offset,
                include_index=False,
            )
        self._db_offset += len(rows)
        return rows

    async def _fill_buffer(self) -> None:
        af.logger.trace("Starting DB buffer fill")
        while len(self._buffer) < self.buffer_size:
            desired = self.max_batch if self.max_batch else (self.buffer_size - len(self._buffer))
            rows = await self._read_next_rows(desired)
            if not rows:
                break
            for item in rows:
                self._buffer.append(item)
        af.logger.trace("DB buffer fill complete")

    async def get(self) -> Any:
        async with self._lock:
            if not self._fill_task or self._fill_task.done():
                self._fill_task = asyncio.create_task(self._fill_buffer())
            if not self._buffer:
                await self._fill_task
            item = self._buffer.popleft()
            if self._fill_task.done():
                self._fill_task = asyncio.create_task(self._fill_buffer())
            return item

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.get()
