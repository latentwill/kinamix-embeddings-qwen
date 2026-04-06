"""
modules/adaptive_cfg.py — Three-sensor adaptive gating for CFG-aware losses.

Uses magnitude, direction variance, and attention probe signals to
self-regulate DMag and CDA loss weights during DSCI training.

Sensor 1 (Magnitude): ||d|| where d = v_full - v_text. Gates DMag.
Sensor 2 (Direction Variance): 1 - mean(cosine_sim) from CDA buffer. Gates CDA.
Sensor 3 (Attention Probe): probe_mass and probe_entropy from DiT hooks. Gates both.
"""
from __future__ import annotations


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


class AdaptiveCFGGating:
    """Compute adaptive DMag and CDA weights from three sensor signals.

    Args:
        dmag_base: Maximum DMag weight (used at cold start).
        cda_base: Maximum CDA weight (used at cold start).
        target_norm: Concept direction magnitude at which DMag fully tapers.
        mass_target: Probe mass at which attention gate fully tapers.
        max_entropy: Theoretical max entropy (log(seq_len)). Used to normalize.
        variance_ceiling: Variance value that maps to full CDA. Default 0.5.
        floor_frac: Minimum fraction of base weights. Default 0.1 (10%).
    """

    def __init__(
        self,
        dmag_base: float,
        cda_base: float,
        target_norm: float,
        mass_target: float,
        max_entropy: float,
        variance_ceiling: float = 0.5,
        floor_frac: float = 0.1,
    ) -> None:
        self.dmag_base = dmag_base
        self.cda_base = cda_base
        self.target_norm = target_norm
        self.mass_target = mass_target
        self.max_entropy = max_entropy
        self.variance_ceiling = variance_ceiling
        self.floor_frac = floor_frac
        self.last_metrics: dict[str, float] = {}

    def compute(
        self,
        mag: float,
        direction_variance: float,
        probe_mass: float,
        probe_entropy: float,
    ) -> tuple[float, float]:
        """Compute adaptive effective weights from current sensor readings.

        Args:
            mag: Current concept direction magnitude ||d||.
            direction_variance: From DirectionBuffer.direction_variance().
            probe_mass: Mean attention mass on concept tokens (from AttentionCollector).
            probe_entropy: Mean attention entropy (from AttentionCollector).

        Returns:
            (dmag_effective, cda_effective) — clamped to [base*floor, base].
        """
        floor = self.floor_frac

        # Sensor 1: Magnitude gate → DMag
        mag_gate = _clamp(1.0 - (mag / self.target_norm), lo=floor, hi=1.0)

        # Sensor 2: Direction variance gate → CDA
        var_gate = _clamp(direction_variance / self.variance_ceiling, lo=floor, hi=1.0)

        # Sensor 3a: Attention mass gate → DMag
        attn_mass_gate = _clamp(1.0 - (probe_mass / self.mass_target), lo=floor, hi=1.0)

        # Sensor 3b: Attention entropy gate → CDA
        entropy_gate = _clamp(probe_entropy / self.max_entropy, lo=floor, hi=1.0)

        # Combined effective weights
        dmag_eff = _clamp(
            self.dmag_base * mag_gate * attn_mass_gate,
            lo=self.dmag_base * floor,
            hi=self.dmag_base,
        )
        cda_eff = _clamp(
            self.cda_base * var_gate * entropy_gate,
            lo=self.cda_base * floor,
            hi=self.cda_base,
        )

        self.last_metrics = {
            "mag": mag,
            "direction_variance": direction_variance,
            "probe_mass": probe_mass,
            "probe_entropy": probe_entropy,
            "mag_gate": mag_gate,
            "var_gate": var_gate,
            "attn_mass_gate": attn_mass_gate,
            "entropy_gate": entropy_gate,
            "dmag_effective": dmag_eff,
            "cda_effective": cda_eff,
        }

        return dmag_eff, cda_eff
