"""
models/force_logger.py
======================
Logger sił i momentów — zapisuje szczegółowe dane z każdego kroku
całkowania do pliku CSV.

Kolumny CSV:
  t                    — czas [s]
  --- Stan ---
  x, y, z              — pozycja w LF [m]
  u, v, w              — prędkości body frame [m/s]
  speed                — prędkość całkowita [m/s]
  alpha_deg            — kąt natarcia [°]
  beta_deg             — kąt ślizgu [°]
  mach                 — liczba Macha [-]
  q_dyn                — ciśnienie dynamiczne [Pa]
  psi_deg, theta_deg, phi_deg — kąty Eulera LF [°]
  --- Masa ---
  mass                 — masa chwilowa [kg]
  xcg                  — pozycja xcg [m]
  --- Współczynniki aero ---
  CN, CA, Cm           — współczynniki [-]
  xcp                  — środek parcia [m od nosa]
  Cmq_eff              — efektywne Cmq [-]
  --- Siły składowe [N] ---
  FA_x, FA_z           — aero body frame
  F_thrust             — ciąg
  Fg_x, Fg_y, Fg_z     — grawitacja body frame
  FX, FY, FZ           — sumy sił body frame
  --- Momenty składowe [N·m] ---
  MA_pitch             — moment aero pochylania
  MA_roll              — moment aero toczenia
  MA_yaw               — moment aero odchylania
  M_thrust_pitch       — moment ciągu pochylania
  M_thrust_yaw         — moment ciągu odchylania
  MY, MZ, MX           — sumy momentów body frame
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np


class ForceLogger:
    """
    Logger sił i momentów działających na rakietę.

    Użycie:
        logger = ForceLogger("rocket_70mm_baseline", enabled=True)
        # Przekaż do ForceModel6DOF
        fm = ForceModel6DOF(..., logger=logger)
        # Po symulacji
        logger.close()
        print(logger.filepath)
    """

    COLUMNS = [
        "t",
        # Stan
        "x", "y", "z",
        "u", "v", "w", "speed",
        "alpha_deg", "beta_deg", "mach", "q_dyn",
        "psi_deg", "theta_deg", "phi_deg",
        # Masa
        "mass", "xcg",
        # Współczynniki aero
        "CN", "CA", "Cm", "xcp", "Cmq_eff", "Clp_eff", "CYB_eff",
        # Siły składowe [N]
        "FA_x", "FA_z",
        "FA_y_aero",       # siła boczna od CYβ
        "FA_y_magnus",     # siła Magnusa
        "F_ctrl",          # siła sterowania (δ=0 na razie)
        "F_thrust",
        "Fg_x", "Fg_y", "Fg_z",
        "FX", "FY", "FZ",
        # Momenty składowe [N·m]
        "MA_pitch", "MA_yaw",
        "MA_roll_cant",    # moment toczący od zaklinowania
        "MA_roll_damp",    # tłumienie toczenia Clp
        "MA_roll",         # suma momentów toczących
        "M_ctrl",          # moment sterowania (δ=0 na razie)
        "M_thrust_pitch", "M_thrust_yaw",
        "MX", "MY", "MZ",
    ]

    def __init__(
        self,
        case_name: str = "simulation",
        log_dir:   str | Path = "logs",
        enabled:   bool = True,
        timestamp: Optional[str] = None,
    ):
        self.enabled = enabled
        self.filepath = None
        self._file = None
        self._writer = None

        if not enabled:
            return

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.filepath = log_dir / f"{case_name}_{timestamp}.csv"
        self._file   = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        print(f"[ForceLogger] Logowanie do: {self.filepath}")

    def log(self, row: dict):
        """Zapisz jeden wiersz danych."""
        if not self.enabled or self._writer is None:
            return
        self._writer.writerow([row.get(c, 0.0) for c in self.COLUMNS])

    def close(self):
        """Zamknij plik CSV."""
        if self._file is not None:
            self._file.close()
            self._file = None
            if self.enabled:
                print(f"[ForceLogger] Zapisano: {self.filepath}")

    def __del__(self):
        self.close()
