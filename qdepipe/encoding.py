"""Encoding spec — how a (scaled) scalar becomes a quantum rotation angle.

This is the data-engineering decision that is *specific to quantum models*, and
the one the literature treats as a throwaway hyperparameter. We make it explicit
and sweepable (axes 7-9). The classical ESN ignores `EncodingSpec` entirely; the
QRC models consume it in their `featurize`.

Theory (Schuld–Sweke–Meyer 2021): the encoding fixes which Fourier frequencies
the reservoir can represent. The *range* matters for injectivity — if the scaled
data spans more than one period of the angle map, distinct inputs alias to the
same feature. With data in [0,1]:
  * range=pi   -> angles in [0, π]   (injective, half a period)
  * range=2pi  -> angles in [0, 2π]  (full period; 0 and 1 collide for cos)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncodingSpec:
    """Quantum input-encoding configuration (axes 7-9)."""

    family: str = "angle"      # "angle" | "amplitude" | "basis" | "reupload"
    input_scaling: float = 1.0  # multiplies the scaled value before the range map
    angle_range: float = 3.141592653589793  # π by default (injective on [0,1])
    reuploads: int = 1          # times the datum is re-injected (axis 9)

    def to_angle(self, value: float) -> float:
        """Map a scaled scalar (typically in [0,1]) to a rotation angle."""
        return self.input_scaling * self.angle_range * float(value)


# convenient presets for sweeps
ANGLE_PI = EncodingSpec(family="angle", angle_range=3.141592653589793)
ANGLE_2PI = EncodingSpec(family="angle", angle_range=6.283185307179586)
