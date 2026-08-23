"""
plot_control_channels.py
========================
Test macierzowy wszystkich trzech kanalow sterowania.

  kanaly     : pitch (d_pitch), yaw (d_yaw), roll (d_roll)
  wychylenia : 2, 5, 8 deg
  przebieg   : 15 s, impuls 1 s zaczynajacy sie w t = 3 s

Jeden wykres na kanal (3 amplitudy naniesione razem) + wykres zbiorczy.

Wazne przy czytaniu wynikow:
  * S_ref/d_ref efektora bierzemy Z GEOMETRII MODELU (AeroSurfaceEffector
    .from_geometry), a nie z tablicy DATCOM — patrz docstring efektora.
    Inaczej moment sterowania jest ~18x zawyzony wzgledem momentu
    przywracajacego i kazde wychylenie "przewraca" rakiete.
  * Kanal roll dziala na Ixx = 0.0032 kg*m^2, czyli ~200x mniejszym niz
    Iyy/Izz. Ta sama skutecznosc daje wiec duzo wieksze przyspieszenie
    katowe — roll z natury reaguje ostrzej niz pitch/yaw.

Uzycie:
    python plot_control_channels.py                        # zero_order (passthrough)
    python plot_control_channels.py --actuator second_order # serwo 2. rzedu
    python plot_control_channels.py --actuator yaml         # z konfiguracji YAML
    python plot_control_channels.py --show                  # pokaz wykresy
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from control import (AeroSurfaceEffector, ControlSystem, PassthroughActuator,
                     SecondOrderActuator, StepCommander,
                     build_actuator_from_config)
from control.demo_model import build_demo_model, fly_demo
from datcom_io.config_reader import load_config

OUT_DIR = ROOT / "results"
CTRL_DIR = ROOT / "datcom_runs" / "rocket_70mm_canards"
SWEEP = [-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.]

AMPS = [2.0, 5.0, 8.0]
T_STEP, T_DUR, T_MAX = 3.0, 1.0, 15.0

# kanal -> (etykieta, predkosc katowa, kat, nazwa momentu, u_idx, M_axis)
# u_idx  — pozycja kanalu w wektorze wychylen u = (d_pitch, d_yaw, d_roll)
# M_axis — os momentu w ControlWrench.M = (Mx=roll, My=pitch, Mz=yaw)
# To DWA ROZNE indeksy; wczesniej uzywalem jednego do obu, przez co
# raportowany moment byl liczony dla niewlasciwego kanalu i wychodzil zerowy.
CHANNELS = {
    "d_pitch": ("PITCH", "qr", "theta", "My", 0, 1),
    "d_yaw":   ("YAW",   "r",  "psi",   "Mz", 1, 2),
    "d_roll":  ("ROLL",  "p",  "phi",   "Mx", 2, 0),
}


def unwrap_deg(a):
    """Katy Eulera owijaja sie na +/-180 deg — bez tego roznice na koncu
    przebiegu (zwlaszcza dla rolla) sa bez sensu."""
    return np.degrees(np.unwrap(np.asarray(a, float)))


def load_table():
    outs = sorted(glob.glob(str(CTRL_DIR / "datcom_ctrl_*.out")))
    if not outs:
        outs = [str(CTRL_DIR / "datcom_ctrl.out")]
    from control.datcom_control import build_control_derivatives
    return build_control_derivatives(outs, sweep_deg=SWEEP,
                                     linear_range_deg=6.0, verbose=False)


def run_case(table, geom, channel, amp, actuator=None):
    eff = AeroSurfaceEffector.from_geometry(table, geom)
    act = actuator if actuator is not None else PassthroughActuator()
    cs = ControlSystem(StepCommander(T_STEP, amp, channel, duration_s=T_DUR),
                       act, [eff])
    return fly_demo(cs, t_max=T_MAX), act


def actual_deflection_history(t_arr, channel, amp, actuator):
    """Rzeczywiste wychylenie aktuatora wzdluz osi czasu (symulacja ODE)."""
    from control.types import ControlCommand, FlightState
    u_idx = CHANNELS[channel][4]
    cmd_fn = StepCommander(T_STEP, amp, channel, duration_s=T_DUR)

    if actuator.n_states == 0:
        out = np.zeros(len(t_arr))
        for i, t in enumerate(t_arr):
            fs = FlightState(t=t)
            cmd = cmd_fn.command(fs)
            out[i] = float(cmd.u[u_idx])
        return out

    xa = actuator.initial_state().copy()
    out = np.zeros(len(t_arr))
    for i, t in enumerate(t_arr):
        fs = FlightState(t=t)
        cmd = cmd_fn.command(fs)
        out[i] = float(actuator.output(cmd, fs, xa)[u_idx])
        dt = (t_arr[i + 1] - t) if i + 1 < len(t_arr) else 0.01
        dxa = actuator.derivatives(cmd, fs, xa)
        xa = xa + dxa * dt
    return out


def moment_history(table, geom, res, channel, amp):
    """Moment sterowania odtworzony wzdluz trajektorii (do wykresu)."""
    from control.types import FlightState
    from models.atmosphere import create_atmosphere
    eff = AeroSurfaceEffector.from_geometry(table, geom)
    atm = create_atmosphere("ISA")
    u_idx, m_axis = CHANNELS[channel][4], CHANNELS[channel][5]
    t_end = T_STEP + T_DUR
    out = np.zeros(len(res.t))
    for i, t in enumerate(res.t):
        if not (T_STEP <= t < t_end):
            continue
        a = atm.at(-res.z[i])
        spd = float(res.speed[i])
        u = np.zeros(3); u[u_idx] = amp
        fs = FlightState(t=t, alpha=float(res.alpha[i]), beta=float(res.beta[i]),
                         mach=a.mach(spd), q_dyn=0.5 * a.density * spd ** 2,
                         speed=spd)
        out[i] = eff.wrench(u, fs).M[m_axis]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--actuator", choices=["zero_order", "second_order", "yaml"],
                    default="zero_order",
                    help="Typ aktuatora: zero_order (passthrough), second_order, "
                         "lub yaml (wczytaj z rocket_70mm_canards.yaml)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    table = load_table()
    geom = build_demo_model(None).geom

    # Aktuator
    if args.actuator == "yaml":
        cfg = load_config(str(ROOT / "configurations" / "rocket_70mm_canards.yaml"))
        actuator = build_actuator_from_config(cfg.actuator)
        act_label = f"z YAML ({cfg.actuator.type})" if cfg.actuator else "z YAML (domyslny)"
    elif args.actuator == "second_order":
        actuator = SecondOrderActuator(n_channels=3, wn=60.0, zeta=0.7,
                                       rate_limit_deg_s=400.0, pos_limit_deg=15.0)
        act_label = f"second_order (wn={actuator.wn}, zeta={actuator.zeta}, " \
                    f"rate={actuator.rate_limit} deg/s, pos={actuator.pos_limit} deg)"
    else:
        actuator = PassthroughActuator()
        act_label = "zero_order (passthrough)"

    print("=" * 74)
    print(f"TEST KANALOW STEROWANIA — impuls {T_DUR:.0f}s od t={T_STEP:.0f}s, "
          f"przebieg {T_MAX:.0f}s, wychylenia {AMPS} deg")
    print(f"S_ref={geom.S_ref:.6f} m^2  d_ref={geom.d_ref:.4f} m "
          f"(z geometrii modelu, nie z LREF DATCOM)")
    print(f"Aktuator: {act_label}")
    print("=" * 74)

    r_off = fly_demo(None, t_max=T_MAX)
    summary = {}

    for channel, (lbl, rate_at, ang_at, mom_lbl, _ui, _mi) in CHANNELS.items():
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"Kanal {lbl} — impuls {T_DUR:.0f} s od t={T_STEP:.0f} s "
                     f"(aktuator: {args.actuator})", fontsize=13, fontweight="bold")
        colors = ["tab:blue", "tab:orange", "tab:red"]
        rows = []

        for amp, col in zip(AMPS, colors):
            res, _act = run_case(table, geom, channel, amp, actuator)
            mom = moment_history(table, geom, res, channel, amp)
            rate = np.degrees(getattr(res, rate_at))
            ang = unwrap_deg(getattr(res, ang_at))
            ang_off = unwrap_deg(getattr(r_off, ang_at))
            d_rate = rate - np.interp(res.t, r_off.t,
                                      np.degrees(getattr(r_off, rate_at)))

            cmd_trace = np.where((res.t >= T_STEP) &
                                 (res.t < T_STEP + T_DUR), amp, 0.0)
            ax[0, 0].plot(res.t, cmd_trace,
                          color=col, lw=1.6, label=f"{amp:.0f} deg cmd")
            act_trace = actual_deflection_history(res.t, channel, amp, actuator)
            ax[0, 0].plot(res.t, act_trace,
                          color=col, lw=1.2, ls="--", label=f"{amp:.0f} deg act")
            ax[0, 1].plot(res.t, mom, color=col, lw=1.6, label=f"{amp:.0f} deg")
            ax[1, 0].plot(res.t, rate, color=col, lw=1.5, label=f"{amp:.0f} deg")
            ax[1, 1].plot(res.t, ang, color=col, lw=1.5, label=f"{amp:.0f} deg")

            rows.append((amp, float(np.max(np.abs(d_rate))),
                         float(ang[-1] - np.interp(res.t[-1], r_off.t, ang_off)),
                         float(np.max(np.abs(mom))), res.status))

        ax[1, 1].plot(r_off.t, unwrap_deg(getattr(r_off, ang_at)), "k--",
                      lw=1.1, label="bez sterowania")

        for a in ax.ravel():
            a.axvspan(T_STEP, T_STEP + T_DUR, color="tab:blue", alpha=0.10)
            a.grid(alpha=0.3); a.legend(fontsize=8)
        ax[0, 0].set_ylabel("wychylenie [deg]"); ax[0, 0].set_title("Komenda / wychylenie rzeczywiste")
        ax[0, 1].set_ylabel(f"{mom_lbl} [N*m]"); ax[0, 1].set_title("Moment sterowania")
        ax[1, 0].set_ylabel(f"{rate_at} [deg/s]"); ax[1, 0].set_xlabel("czas [s]")
        ax[1, 0].set_title("Predkosc katowa")
        ax[1, 1].set_ylabel(f"{ang_at} [deg]"); ax[1, 1].set_xlabel("czas [s]")
        ax[1, 1].set_title("Kat")
        fig.tight_layout()
        p = OUT_DIR / f"control_channel_{lbl.lower()}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)

        summary[lbl] = rows
        print(f"\n{lbl}:")
        print(f"  {'ampl':>5} {'max|d_rate|':>11} {'d_kat(koniec)':>14} "
              f"{'max|M|':>9}  status")
        for amp, mr, da, mm, st in rows:
            print(f"  {amp:>5.0f} {mr:>9.1f}d/s {da:>12.2f}deg {mm:>8.3f}Nm  {st}")

    # --- zbiorczy: liniowosc odpowiedzi wzgledem amplitudy ---------------- #
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for (lbl, rows), col in zip(summary.items(), ["tab:blue", "tab:green", "tab:red"]):
        a = [r[0] for r in rows]
        ax[0].plot(a, [r[3] for r in rows], "o-", color=col, label=lbl)
        ax[1].plot(a, [abs(r[1]) for r in rows], "o-", color=col, label=lbl)
    ax[0].set_xlabel("wychylenie [deg]"); ax[0].set_ylabel("max |moment| [N*m]")
    ax[0].set_title("Moment sterowania vs wychylenie\n(powinien byc liniowy)")
    ax[1].set_xlabel("wychylenie [deg]"); ax[1].set_ylabel("max |predkosc katowa| [deg/s]")
    ax[1].set_title("Odpowiedz vs wychylenie")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=9)
    fig.tight_layout()
    p = OUT_DIR / "control_channels_summary.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWykresy: {OUT_DIR}/control_channel_{{pitch,yaw,roll}}.png")
    print(f"          {p.name}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
