"""Activation steering: fused Triton kernel + Wan pre-hooks."""
from dailies_dag.steering.hook import (
    SteeringHandle,
    install_steering_pre_hook,
    remove_steering_pre_hook,
)
from dailies_dag.steering.triton_steer_kernel import (
    HAS_TRITON,
    is_triton_available,
    steer_activations,
)

__all__ = [
    "HAS_TRITON",
    "SteeringHandle",
    "install_steering_pre_hook",
    "is_triton_available",
    "remove_steering_pre_hook",
    "steer_activations",
]
