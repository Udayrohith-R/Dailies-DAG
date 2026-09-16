"""Single-clip LoRA overfit driver (stub: wire to diffusers Wan/LTX pipeline)."""
from __future__ import annotations

def overfit_clip(pipeline, clip_latents, steps: int = 500, lr: float = 1e-5):
    """Overfit LoRA adapters to ONE clip. Full training loop is phase-1b work."""
    raise NotImplementedError("Wire to your Wan2.1-1.3B / LTX pipeline with PEFT LoRA.")
