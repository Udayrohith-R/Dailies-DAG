"""Stage-Relative GRPO guardrails (SRPO-Guard)."""
from dailies_dag.rl.srpo_guard import (
    DepthRewardModel,
    FlowRewardModel,
    GuardConfig,
    StageSchedule,
    compute_rewards,
    depth_loss,
    flow_loss,
    grpo_step,
    luminance_shift_score,
    pareto_filter,
    score_rollouts,
    stage_weight,
)

__all__ = [
    "DepthRewardModel",
    "FlowRewardModel",
    "GuardConfig",
    "StageSchedule",
    "compute_rewards",
    "depth_loss",
    "flow_loss",
    "grpo_step",
    "luminance_shift_score",
    "pareto_filter",
    "score_rollouts",
    "stage_weight",
]
