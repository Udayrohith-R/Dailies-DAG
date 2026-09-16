"""Asyncio/Trio frame scheduler: streams frames through paged attention without blocking."""
from __future__ import annotations
import asyncio
from typing import List

async def generate_branch(attn, hidden: "torch.Tensor", block_table: List[int],
                          manager, layer_id: int, start_frame: int, queue: asyncio.Queue):
    import torch
    with torch.no_grad():
        out = attn(hidden, block_table=block_table, cache_manager=manager,
                   layer_id=layer_id, start_frame=start_frame)
    await queue.put(out)
    return out
