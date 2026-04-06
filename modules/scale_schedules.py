"""modules/scale_schedules.py — Timestep-dependent concept scale schedules for CFG decomposition."""
import math

VALID_SCHEDULES = ("constant", "linear", "cosine", "step")


def get_concept_scale(
    t: float,
    schedule: str = "constant",
    scale_high: float = 3.0,
    scale_low: float = 1.5,
    step_cutoff: float = 0.3,
    base_scale: float = 2.0,
) -> float:
    """Return the concept scale for a normalized timestep t in [0.0, 1.0].

    t=1.0 corresponds to the start of denoising (high noise).
    t=0.0 corresponds to the end of denoising (low noise / clean image).

    Args:
        t: Normalized timestep in [0.0, 1.0]. 1.0 = high noise, 0.0 = clean.
        schedule: One of "constant", "linear", "cosine", "step".
        scale_high: Scale value at t=1.0 (high noise end).
        scale_low: Scale value at t=0.0 (low noise end).
        step_cutoff: Threshold for "step" schedule (returns scale_high when t > cutoff).
        base_scale: Fixed scale returned by "constant" schedule.

    Returns:
        Concept scale as a float.
    """
    if schedule == "constant":
        return base_scale
    elif schedule == "linear":
        return scale_high * t + scale_low * (1 - t)
    elif schedule == "cosine":
        return scale_low + 0.5 * (scale_high - scale_low) * (1 + math.cos(math.pi * (1 - t)))
    elif schedule == "step":
        return scale_high if t > step_cutoff else scale_low
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Valid: {VALID_SCHEDULES}")
