"""
models/launcher.py
==================
Model wyrzutni (szyny startowej).

Przez pierwsze L_rail metrów (mierzone wzdłuż osi X_body od momentu
zapłonu) rakieta jest mechanicznie związana z szyną. Szyna ogranicza:
  - ruch poprzeczny (v=0, w=0)
  - obroty (p=0, qr=0, r=0)
  - pochodne tych wielkości (zerowane po obliczeniu)

Rakieta może się poruszać wyłącznie wzdłuż osi X_body (u).

Po opuszczeniu szyny (przebyta droga >= L_rail) rakieta leci swobodnie.

Droga wzdłuż szyny liczona jest jako całka z u(t) od momentu zapłonu.
Przy u > 0 (rakieta jedzie do przodu) droga rośnie. Jeśli rakieta
cofnęłaby się (u < 0) droga nie maleje — fizycznie szyna trzyma jedną
stronę.

Parametry
---------
L_rail : float
    Długość szyny startowej [m]. 0 = brak szyny.
t_ignition : float
    Czas zapłonu [s]. Przed zapłonem rakieta stoi nieruchomo.
"""

from dataclasses import dataclass


@dataclass
class LauncherConfig:
    """
    Konfiguracja wyrzutni.

    Parametry
    ---------
    L_rail : float
        Długość szyny [m]. 0 = brak ograniczenia (lot swobodny od t=0).
    t_ignition : float
        Czas zapłonu [s]. Synchronizowany z MassModel.t_ignition.
    """
    L_rail:     float = 0.0
    t_ignition: float = 0.0


class RailLauncher:
    """
    Model szyny startowej.

    Śledzi przebytą drogę wzdłuż szyny i informuje solver czy rakieta
    jest jeszcze na szynie.

    Użycie w force_model6:
        if launcher.on_rail(distance):
            # zeruj v, w, p, qr, r i ich pochodne
    """

    def __init__(self, config: LauncherConfig):
        self.L_rail     = float(config.L_rail)
        self.t_ignition = float(config.t_ignition)

    def on_rail(self, rail_distance: float) -> bool:
        """
        Czy rakieta jest jeszcze na szynie?

        Parameters
        ----------
        rail_distance : float
            Droga przebyta wzdłuż szyny od chwili zapłonu [m].
            Liczona jako całka z max(u, 0) dt w force_model.
        """
        if self.L_rail <= 0.0:
            return False
        return rail_distance < self.L_rail

    def is_disabled(self) -> bool:
        """True gdy szyna jest wyłączona (L_rail=0)."""
        return self.L_rail <= 0.0
