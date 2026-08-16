"""
control/derivatives.py
======================
Pochodne sterowania C*_delta(alpha, Mach) — skutecznosc powierzchni sterowych,
oraz wzorce wychylen paneli uzywane przy generowaniu sweepa DATCOM.

Wzorce sa TUTAJ, a nie w generatorze, zeby generator decków i efektor liczacy
momenty korzystaly z tej samej definicji "co znaczy wychylenie w kanale pitch".
Rozjazd miedzy nimi bylby cichym bledem znaku.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Wzorce wychylen dla ukladu krzyzowego 4 paneli przy PHIF = 0/90/180/270.
# Mnozniki wychylenia zadanego w danym kanale.
#   pitch: para paneli w plaszczyznie pitch, przeciwne znaki
#   yaw:   para paneli w plaszczyznie yaw,  przeciwne znaki
#   roll:  wszystkie panele zgodnie
# UWAGA: przypisanie indeksu panelu do plaszczyzny to KONWENCJA. Weryfikuje
# ja procedura znakowa (patrz docstring AeroSurfaceEffector) — jesli test
# znaku nie przechodzi, poprawiamy TE tablice, a nie znak w efektorze.
PANEL_PATTERNS: Dict[str, List[float]] = {
    "d_pitch": [0.0, +1.0, 0.0, -1.0],
    "d_yaw":   [+1.0, 0.0, -1.0, 0.0],
    "d_roll":  [+1.0, +1.0, +1.0, +1.0],
}


@dataclass
class ControlDerivTable:
    """
    Pochodne sterowania w funkcji (alpha, Mach), wszystkie NA RADIAN.

    Przeliczenie deg->rad robimy raz przy budowie tablicy, zeby goracy tor
    (kazdy stopien RK45) nie konwertowal jednostek.

    Interpolator budowany JEDEN RAZ w __post_init__ — swiadomie inaczej niz
    TableAero._interp, ktory tworzy RegularGridInterpolator przy KAZDYM
    wywolaniu (models/aerodynamics.py:333-344). Tutaj to byloby kosztowne.
    """
    alpha_rad: np.ndarray          # (n_alpha,)
    mach:      np.ndarray          # (n_mach,)
    Cm_delta:  np.ndarray          # (n_alpha, n_mach) [1/rad] pitch
    Cn_delta:  np.ndarray          # yaw
    Cl_delta:  np.ndarray          # roll
    CN_delta:  np.ndarray          # sila normalna
    CY_delta:  np.ndarray          # sila boczna
    xcg_ref:   float = 0.0
    S_ref:     float = 0.0
    d_ref:     float = 0.0
    delta_ref_deg: float = 5.0
    # Diagnostyka dopasowania liniowego (R^2 najgorszej komorki itp.)
    fit_info:  dict = field(default_factory=dict)

    _interp: Optional[dict] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._build()

    def _build(self):
        grid = (np.asarray(self.alpha_rad, float), np.asarray(self.mach, float))
        self._interp = {
            name: RegularGridInterpolator(
                grid, np.asarray(getattr(self, name), float),
                method="linear", bounds_error=False, fill_value=None)
            for name in ("Cm_delta", "Cn_delta", "Cl_delta",
                         "CN_delta", "CY_delta")
        }

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_interp"] = None      # interpolatory nie sa picklowalne stabilnie
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build()

    def interp(self, alpha_rad: float, mach: float) -> Dict[str, float]:
        """Wszystkie pochodne w jednym punkcie (alpha [rad], Mach)."""
        if self._interp is None:
            self._build()
        pt = np.array([[float(alpha_rad), float(mach)]])
        return {k: float(v(pt)[0]) for k, v in self._interp.items()}


def fit_derivative_from_sweep(delta_deg: np.ndarray,
                              coeff: np.ndarray,
                              linear_range_deg: float = 6.0):
    """
    Dopasowuje pochodna d(coeff)/d(delta) [1/rad] ze sweepa wychylen.

    Sweep (a nie pojedyncza roznica skonczona) pozwala ZWERYFIKOWAC zalozenie
    liniowosci zamiast je przyjmowac: dopasowanie robimy tylko w zakresie
    +/- linear_range_deg, a jako diagnostyke zwracamy R^2 oraz maksymalne
    odchylenie od prostej na CALYM sweepie.

    Zwraca (slope_per_rad, info_dict).
    """
    d = np.asarray(delta_deg, float)
    c = np.asarray(coeff, float)
    mask = np.abs(d) <= linear_range_deg + 1e-9
    if mask.sum() < 2:
        mask = np.ones_like(d, dtype=bool)

    # Regresja przez punkty w zakresie liniowym; wyraz wolny = wartosc przy 0
    A = np.vstack([d[mask], np.ones(mask.sum())]).T
    slope_per_deg, intercept = np.linalg.lstsq(A, c[mask], rcond=None)[0]

    pred = slope_per_deg * d[mask] + intercept
    ss_res = float(np.sum((c[mask] - pred) ** 2))
    ss_tot = float(np.sum((c[mask] - np.mean(c[mask])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    pred_all = slope_per_deg * d + intercept
    max_dev = float(np.max(np.abs(c - pred_all))) if len(d) else 0.0

    info = {"r2": r2, "max_dev_full_sweep": max_dev,
            "intercept": float(intercept), "n_fit": int(mask.sum())}
    return float(slope_per_deg) * (180.0 / np.pi), info
