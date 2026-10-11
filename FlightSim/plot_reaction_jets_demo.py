"""
plot_reaction_jets_demo.py
==========================
Demonstracja sterowania REACTION JETS na modelu demonstracyjnym 70mm
(control/demo_model.py — statycznie stabilna aero, bez ciagu glownego).

Silniki z configurations/rocket_70mm_reaction_jets.yaml, komendy z
harmonogramu testowego (ScheduleJetCommander):

    t = 3.00 s  R1E1           -> impuls "nos w gore"   (silnik na +Z)
    t = 3.00 s  R1E1 ponownie  -> odrzucone (silnik zuzyty)
    t = 5.00 s  R2E3 + R2E4    -> para z rzedu 2 (przesuniecie 22.5 deg)
    t = 7.00 s  R1E5           -> impuls "nos w dol"    (silnik na -Z)

Uzycie:
    python plot_reaction_jets_demo.py            # zapis do results/
    python plot_reaction_jets_demo.py --show     # pokaz wykres
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from control import ScheduleJetCommander, build_reaction_jet_system
from control.demo_model import fly_demo
from datcom_io.config_reader import load_config

CFG = ROOT / "configurations" / "rocket_70mm_reaction_jets.yaml"
OUT = ROOT / "results" / "reaction_jets_demo.png"
T_MAX = 10.0

SCHEDULE = [
    (3.0, "R1E1"),
    (3.0, "R1E1"),              # celowo: test odrzucenia zuzytego silnika
    (5.0, ["R2E3", "R2E4"]),
    (7.0, "R1E5"),
]


def main():
    cfg = load_config(CFG)
    jets = build_reaction_jet_system(cfg, commander=ScheduleJetCommander(SCHEDULE))
    eng = cfg.reaction_jets.engine
    print("=" * 70)
    print(f"REACTION JETS — {cfg.name}: {jets.n_engines} silnikow, "
          f"{eng.thrust:.0f} N x {eng.burn_time * 1e3:.1f} ms "
          f"(impuls {jets.profile.impulse:.3f} N*s), opoznienie "
          f"{eng.ignition_delay * 1e3:.1f} ms")
    print("=" * 70)

    r0 = fly_demo(None, t_max=T_MAX)
    res = fly_demo(jets, t_max=T_MAX, dt_output=0.001)

    print("\nLog komend:")
    for e in res.jet_events:
        extra = (f"zaplon {e.t_ign:.4f} s, koniec {e.t_end:.4f} s"
                 if e.status == "fired" else "")
        print(f"  t={e.t_cmd:6.3f} s  {e.engine:5s}  {e.status:15s} {extra}")

    # Ciag z gestej siatki wokol impulsow — impuls 10 ms jest wezszy niz
    # krok wyjscia, wiec do wykresu ciagu probkujemy osobno.
    t_thr = np.unique(np.concatenate(
        [np.linspace(e.t_ign - 0.005, e.t_end + 0.005, 200)
         for e in res.jet_events if e.status == "fired"] + [res.t]))
    thr = jets.thrust_history(t_thr)

    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"Reaction jets — {cfg.name} (model demo 70mm)",
                 fontsize=13, fontweight="bold")
    for k in np.nonzero(thr.max(axis=0) > 0)[0]:
        ax[0].plot(t_thr, thr[:, k], label=jets.layout.names[k])
    ax[0].set_ylabel("ciag [N]")
    ax[0].legend(loc="upper right", ncol=4, fontsize=8)

    for name, lbl, col in [("qr", "q (pitch)", "tab:blue"),
                           ("r", "r (yaw)", "tab:orange"),
                           ("p", "p (roll)", "tab:green")]:
        ax[1].plot(res.t, np.degrees(getattr(res, name)), color=col, label=lbl)
        ax[1].plot(r0.t, np.degrees(getattr(r0, name)), color=col, ls=":", lw=0.8)
    ax[1].set_ylabel("predkosc katowa [deg/s]")
    ax[1].legend(loc="upper right", fontsize=8)

    ax[2].plot(res.t, np.degrees(res.alpha), label="alpha")
    ax[2].plot(res.t, np.degrees(res.beta), label="beta")
    ax[2].set_ylabel("kat [deg]")
    ax[2].legend(loc="upper right", fontsize=8)

    ax[3].plot(res.t, np.degrees(np.unwrap(res.theta)) - np.interp(
        res.t, r0.t, np.degrees(np.unwrap(r0.theta))), label="d theta")
    ax[3].plot(res.t, np.degrees(np.unwrap(res.psi)) - np.interp(
        res.t, r0.t, np.degrees(np.unwrap(r0.psi))), label="d psi")
    ax[3].set_ylabel("roznica vs bez jetow [deg]")
    ax[3].set_xlabel("t [s]")
    ax[3].legend(loc="upper right", fontsize=8)

    for a in ax:
        a.grid(alpha=0.3)
        for e in res.jet_events:
            if e.status == "fired":
                a.axvline(e.t_ign, color="grey", lw=0.6, ls="--")

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()          # Windows moze trzymac lock na starym pliku
    fig.savefig(OUT, dpi=120)
    print(f"\nWykres: {OUT}")
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
