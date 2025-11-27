"""
Utility subpackage for autonomous_parking.

Contains small, reusable helpers such as geometry utilities
and compatibility wrappers.
"""

from .geometry import wrap_to_pi, angle_diff, euclidean_distance, squared_distance

__all__ = [
    "wrap_to_pi",
    "angle_diff",
    "euclidean_distance",
    "squared_distance",
]
