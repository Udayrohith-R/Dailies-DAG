"""Branchable EditDAG: each branch owns a block_table forked at branch_frame_idx."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class DagNode:
    id: str
    parent: Optional[str]
    branch_frame_idx: int
    block_table: List[int]
    edit: dict = field(default_factory=dict)

class EditDAG:
    def __init__(self, manager, root_table: List[int]):
        self.manager = manager
        self.nodes: Dict[str, DagNode] = {"main": DagNode("main", None, 0, root_table)}
        self._lock = asyncio.Lock()

    async def branch(self, parent_id: str, child_id: str, branch_frame_idx: int,
                     edit: Optional[dict] = None, tail_frames: int = 0) -> DagNode:
        async with self._lock:
            parent = self.nodes[parent_id]
            child_table = self.manager.fork(parent.block_table, branch_frame_idx, tail_frames)
            node = DagNode(child_id, parent_id, branch_frame_idx, child_table, edit or {})
            self.nodes[child_id] = node
            return node

    async def free(self, node_id: str) -> None:
        async with self._lock:
            node = self.nodes.pop(node_id)
            self.manager.free_seq(node.block_table)
