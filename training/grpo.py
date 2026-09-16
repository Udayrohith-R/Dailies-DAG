"""GRPO / SRPO-Guard loop stub: reward relight success, penalize depth/flow drift."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GuardConfig:
    w_light: float = 1.0
    lambda_depth: float = 2.0
    lambda_flow: float = 2.0

def guard_loss(luminance_gain: float, depth_drift: float, flow_drift: float,
               cfg: GuardConfig = GuardConfig()) -> float:
    return -cfg.w_light * luminance_gain + cfg.lambda_depth * depth_drift + cfg.lambda_flow * flow_drift

def grpo_step(policy, rollouts: list, cfg: GuardConfig = GuardConfig()):
    """Single GRPO step over branched rollouts. Full TRL wiring is phase-3 work."""
    losses = [guard_loss(r["gain"], r["depth_drift"], r["flow_drift"], cfg) for r in rollouts]
    base = sum(losses) / max(len(losses), 1)
    advantages = [base - l for l in losses]  # group-relative
    return {"loss": base, "advantages": advantages}
