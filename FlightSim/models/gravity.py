"""
models/gravity.py
=================
Model grawitacji.

Zaimplementowane:
  - Stałe g (g = 9.80665 m/s²) — domyślne
  - g(h): grawitacja jako funkcja wysokości (sferyczna Ziemia)

Zaślepka:
  - WGS84: pełny model grawitacji z rozwinięciem sferyczno-harmonicznym
    (WGS84 GRS80), istotny dla trajektorii dalekiego zasięgu

Dla rakiet 70mm zasięgu kilku km: g(h) vs g_const różnica < 0.01%.
"""

import numpy as np
from typing import Protocol


G0 = 9.80665   # standardowe przyspieszenie grawitacyjne [m/s²]
R_EARTH = 6371000.0  # promień Ziemi (sferyczny) [m]


class GravityModel(Protocol):
    """Interfejs modelu grawitacji."""
    def g(self, h: float) -> float:
        """Przyspieszenie grawitacyjne na wysokości h [m/s²]."""
        ...


class ConstantGravity:
    """Stałe g = G0. Wystarczające dla zasięgów < 100 km."""

    def g(self, h: float) -> float:
        return G0

    def __repr__(self) -> str:
        return f"ConstantGravity(g={G0} m/s²)"


class SphericalGravity:
    """
    Grawitacja z korekcją wysokości: g(h) = G0 * (R/(R+h))².
    Różnica vs stałe g: ~0.03% przy h = 1 km, ~0.3% przy h = 10 km.
    """

    def g(self, h: float) -> float:
        return G0 * (R_EARTH / (R_EARTH + max(0.0, h))) ** 2

    def __repr__(self) -> str:
        return "SphericalGravity(inverse-square law)"


class WGS84Gravity:
    """
    ZAŚLEPKA: Grawitacja WGS84 z rozwinięciem sferyczno-harmonicznym.
    Do implementacji przy rozszerzeniu na dalekie zasięgi.
    Wymaga szerokości geograficznej φ.
    """

    def g(self, h: float) -> float:
        raise NotImplementedError(
            "Model WGS84 grawitacji nie jest jeszcze zaimplementowany. "
            "Użyj ConstantGravity lub SphericalGravity."
        )


def create_gravity(model: str = "constant") -> GravityModel:
    """
    Fabryka modelu grawitacji.

    Parameters
    ----------
    model : str
        "constant"  → stałe g (domyślne)
        "spherical" → g(h) z poprawką wysokości
        "wgs84"     → zaślepka WGS84
    """
    if model == "constant":
        return ConstantGravity()
    elif model == "spherical":
        return SphericalGravity()
    elif model == "wgs84":
        return WGS84Gravity()
    else:
        raise ValueError(f"Nieznany model grawitacji: '{model}'.")
