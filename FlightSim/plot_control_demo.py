"""
plot_control_demo.py
====================
Wizualizacja dzialania modulu sterowania.

Rysuje dwa zestawy wykresow:

  1. ODPOWIEDZ SKOKOWA (zawsze) — pelny lancuch
     command -> actuator -> moment -> 6DOF na skok wychylenia canardow.
     Uzywa syntetycznych pochodnych, dopoki nie ma danych z DATCOM, wiec
     dziala od razu i pokazuje MECHANIKE lancucha.

  2. POCHODNE STEROWANIA (gdy istnieje datcom_ctrl.out) — zmierzone
     C*_delta(alpha, Mach) plus kontrola liniowosci: surowy przebieg
     wspolczynnika wzdluz sweepa wychylen z naniesiona dopasowana prosta.
     Tu wlasnie widac, czy zalozenie liniowosci sie broni.

Uzycie:
    python plot_control_demo.py
    python plot_control_demo.py --amp 4 --tstep 2.5
    python plot_control_demo.py --show          # okno zamiast pliku
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from control import (AeroSurfaceEffector, ControlDerivTable, ControlSystem,
                     PassthroughActuator, StepCommander)
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF
from forces.force_model6 import (ForceModel6DOF, PropulsionConfig6DOF,
                                 RocketGeometry6DOF)
from models.aerodynamics import TableAero
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from models.mass6 import ConstantIxx, MassModel6DOF

def safe_savefig(fig, path, **kwargs):
    p = Path(path)
    if p.exists():
        p.unlink()
    fig.savefig(p, **kwargs)

OUT_DIR = ROOT / "results"
CTRL_DIR = ROOT / "datcom_runs" / "rocket_70mm_canards"


def ctrl_outs():
    """Pliki sweepa: po jednym na kanal, ze zgodnoscia wstecz."""
    o = sorted(CTRL_DIR.glob("datcom_ctrl_*.out"))
    if not o and (CTRL_DIR / "datcom_ctrl.out").exists():
        o = [CTRL_DIR / "datcom_ctrl.out"]
    return o
SWEEP = [-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.]


# --------------------------------------------------------------------------
def synthetic_table(Cm_d=-3.0, Cn_d=+3.0, Cl_d=+0.5, CN_d=+8.0, CY_d=+8.0):
    """
    Pochodne zastepcze, gdy brak danych z DATCOM.

    Cm_delta < 0 przy dodatnim CN_delta jest tu wpisane recznie tylko po to,
    by wykres mial sensowny ksztalt — PRAWDZIWY znak bierze sie z pomiaru
    DATCOM (patrz control/effectors/aero_surface.py).
    """
    alpha = np.deg2rad(np.array([-20., -10., 0., 10., 20.]))
    mach = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    o = np.ones((len(alpha), len(mach)))
    return ControlDerivTable(
        alpha_rad=alpha, mach=mach,
        Cm_delta=Cm_d*o, Cn_delta=Cn_d*o, Cl_delta=Cl_d*o,
        CN_delta=CN_d*o, CY_delta=CY_d*o,
        xcg_ref=0.71, S_ref=0.00385, d_ref=0.070, delta_ref_deg=5.0)


def build_model(control=None):
    mass = MassModel6DOF(m_full=4.2, m_empty=4.19, t_burn=1.0,
                         xcg_full=0.71, xcg_empty=0.71,
                         Iyy_full=0.646, Iyy_empty=0.646,
                         ixx_model=ConstantIxx(0.0032))
    aero = TableAero(
        alpha_table=np.deg2rad(np.array([-20., -10., 0., 10., 20.])),
        mach_table=np.array([0.1, 0.5, 1.0, 2.0, 3.0]),
        CA_table=0.45*np.ones((5, 5)),
        CN_table=np.array([[-4.]*5, [-2.]*5, [0.]*5, [2.]*5, [4.]*5]),
        Cm_table=np.array([[1.2]*5, [0.6]*5, [0.]*5, [-0.6]*5, [-1.2]*5]),
        xcg_ref=0.71)
    return ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"), mass_model=mass, aero_model=aero,
        gravity=create_gravity("constant"),
        geometry=RocketGeometry6DOF.from_diameter(0.070, xcp=0.95),
        propulsion=PropulsionConfig6DOF(thrust=0.0),
        launcher=LauncherConfig(L_rail=0.0), control=control)


def fly(control, t_max=12.0):
    st = State6DOF.initial(elevation_deg=80.0, azimuth_deg=0.0, speed_0=250.0)
    return run_simulation_6dof(build_model(control), st, t_max=t_max,
                               dt_output=0.01, z_ground=-1e9)


# --------------------------------------------------------------------------
def plot_step_response(table, amp, tstep, dur, tag, show):
    r_off = fly(None)
    r_on = fly(ControlSystem(StepCommander(tstep, amp, "d_pitch", duration_s=dur),
                             PassthroughActuator(), [AeroSurfaceEffector(table)]))
    t_end = tstep + dur if dur else float("inf")

    # Odtworz komende i moment wzdluz trajektorii (do wykresu).
    eff = AeroSurfaceEffector(table)
    from control.types import FlightState
    atm = create_atmosphere("ISA")
    d_cmd, m_ctrl = [], []
    for i, t in enumerate(r_on.t):
        d = amp if (tstep <= t < t_end) else 0.0
        a = atm.at(-r_on.z[i])
        spd = float(r_on.speed[i])
        fs = FlightState(t=t, alpha=float(r_on.alpha[i]), mach=a.mach(spd),
                         q_dyn=0.5*a.density*spd**2, speed=spd)
        d_cmd.append(d)
        m_ctrl.append(eff.wrench(np.array([d, 0., 0.]), fs).My)
    d_cmd = np.array(d_cmd); m_ctrl = np.array(m_ctrl)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    win = (f"impuls {amp:+.1f} deg, t={tstep:.1f}..{t_end:.1f}s"
           if dur else f"skok trwaly {amp:+.1f} deg od t={tstep:.1f}s")
    fig.suptitle(f"Modul sterowania — {win} w kanale pitch  [{tag}]",
                 fontsize=13, fontweight="bold")

    def mark(a):
        """Zacieniowany czas trwania komendy — widoczny na kazdym panelu."""
        if dur:
            a.axvspan(tstep, t_end, color="tab:blue", alpha=0.10,
                      label="komenda aktywna")
        else:
            a.axvline(tstep, color="gray", ls=":", lw=1)

    a = ax[0, 0]
    a.plot(r_on.t, d_cmd, "b-", lw=1.6)
    mark(a)
    a.set_ylabel("wychylenie [deg]"); a.set_title("1. Komenda -> serwo (passthrough)")
    a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(r_on.t, m_ctrl, "r-", lw=1.6)
    mark(a); a.axhline(0, color="k", lw=0.6)
    a.set_ylabel("M_ctrl pitch [N*m]"); a.set_title("2. Moment sterowania")
    a.grid(alpha=0.3)

    a = ax[0, 2]
    a.plot(r_off.t, np.degrees(r_off.qr), "k--", lw=1.2, label="bez sterowania")
    a.plot(r_on.t, np.degrees(r_on.qr), "b-", lw=1.6, label="ze sterowaniem")
    mark(a)
    a.set_ylabel("q [deg/s]"); a.set_title("3. Predkosc katowa pitch")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(r_off.t, np.degrees(r_off.theta), "k--", lw=1.2, label="bez sterowania")
    a.plot(r_on.t, np.degrees(r_on.theta), "b-", lw=1.6, label="ze sterowaniem")
    mark(a)
    a.set_xlabel("czas [s]"); a.set_ylabel("theta [deg]")
    a.set_title("4. Kat pochylenia"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(r_off.t, np.degrees(r_off.alpha), "k--", lw=1.2, label="bez sterowania")
    a.plot(r_on.t, np.degrees(r_on.alpha), "b-", lw=1.6, label="ze sterowaniem")
    mark(a)
    a.set_xlabel("czas [s]"); a.set_ylabel("alpha [deg]")
    a.set_title("5. Kat natarcia"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 2]
    a.plot(r_off.x, -r_off.z, "k--", lw=1.2, label="bez sterowania")
    a.plot(r_on.x, -r_on.z, "b-", lw=1.6, label="ze sterowaniem")
    a.set_xlabel("downrange [m]"); a.set_ylabel("wysokosc [m]")
    a.set_title("6. Trajektoria"); a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    p = OUT_DIR / "control_step_response.png"
    safe_savefig(fig, p, dpi=130, bbox_inches="tight")
    print(f"Zapisano: {p}")

    d_theta = np.degrees(r_on.theta[-1] - r_off.theta[-1])
    print(f"  odchylenie theta na koncu: {d_theta:+.2f} deg")
    print(f"  max |M_ctrl|: {np.max(np.abs(m_ctrl)):.2f} N*m")
    if show:
        plt.show()
    return fig


def plot_derivatives(show):
    """Wykresy zmierzonych pochodnych + kontrola liniowosci."""
    from control.datcom_control import build_control_derivatives
    from datcom_io.missile_datcom_reader import parse_missile_datcom_cases
    from control.datcom_control import _coeff_grid, parse_sweep_labels

    outs = ctrl_outs()
    tab = build_control_derivatives(outs, sweep_deg=SWEEP,
                                    linear_range_deg=6.0)
    al = np.degrees(tab.alpha_rad)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Pochodne sterowania zmierzone w DATCOM (sweep -10..+10 deg)",
                 fontsize=13, fontweight="bold")

    for k, (field, lbl) in enumerate([("Cm_delta", "Cm_delta (pitch)"),
                                      ("Cn_delta", "Cn_delta (yaw)"),
                                      ("Cl_delta", "Cl_delta (roll)")]):
        a = ax[k]
        T = getattr(tab, field)
        for j, M in enumerate(tab.mach):
            a.plot(al, T[:, j], marker="o", ms=3, lw=1.2, label=f"M={M:.2f}")
        a.axhline(0, color="k", lw=0.6)
        a.set_xlabel("alpha [deg]"); a.set_ylabel(f"{field} [1/rad]")
        a.set_title(lbl); a.grid(alpha=0.3)
        if k == 0:
            a.legend(fontsize=7, ncol=2)
        info = (tab.fit_info or {}).get(field, {})
        if info:
            a.text(0.02, 0.02, f"R^2(min)={info.get('worst_r2', float('nan')):.4f}",
                   transform=a.transAxes, fontsize=8,
                   bbox=dict(fc="white", alpha=0.7, ec="gray"))

    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    p = OUT_DIR / "control_derivatives.png"
    safe_savefig(fig, p, dpi=130, bbox_inches="tight")
    print(f"Zapisano: {p}")

    # --- kontrola liniowosci: surowy CM wzdluz sweepa + dopasowana prosta --
    groups = parse_missile_datcom_cases(outs[0])
    labels = parse_sweep_labels(outs[0])
    if labels and len(labels) == len(groups) - 1:
        base, sweeps = groups[0], groups[1:]
        alpha0 = np.asarray(base.cases[0].alpha, float)
        mach = np.asarray(sorted(c.mach for c in base.cases), float)
        ia = int(np.argmin(np.abs(alpha0)))          # alpha ~ 0
        im = min(2, len(mach) - 1)

        d_list, c_list = [], []
        for (ch, d), grp in zip(labels, sweeps):
            if ch != "d_pitch":
                continue
            d_list.append(d)
            c_list.append(_coeff_grid(grp, "CM", alpha0, mach)[ia, im])
        if d_list:
            o = np.argsort(d_list)
            d_arr = np.array(d_list)[o]; c_arr = np.array(c_list)[o]
            fig2, a2 = plt.subplots(figsize=(7.5, 5))
            a2.plot(d_arr, c_arr, "bo-", ms=5, lw=1.4, label="CM z DATCOM")
            fit = np.abs(d_arr) <= 6.0
            k, b = np.polyfit(d_arr[fit], c_arr[fit], 1)
            xs = np.linspace(d_arr.min(), d_arr.max(), 50)
            a2.plot(xs, k*xs + b, "r--", lw=1.4,
                    label=f"dopasowanie |d|<=6 deg  ({k:+.5f}/deg)")
            a2.axvspan(-6, 6, color="green", alpha=0.08,
                       label="zakres dopasowania")
            a2.axhline(0, color="k", lw=0.6); a2.axvline(0, color="k", lw=0.6)
            a2.set_xlabel("wychylenie canardow [deg]")
            a2.set_ylabel(f"CM  (alpha={alpha0[ia]:.0f} deg, M={mach[im]:.2f})")
            a2.set_title("Kontrola liniowosci skutecznosci sterowania\n"
                         "odchylenie punktow od prostej poza zielonym pasem "
                         "= nieliniowosc przy duzych wychyleniach")
            a2.legend(fontsize=9); a2.grid(alpha=0.3)
            fig2.tight_layout()
            p2 = OUT_DIR / "control_linearity.png"
            safe_savefig(fig2, p2, dpi=130, bbox_inches="tight")
            print(f"Zapisano: {p2}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amp", type=float, default=2.0, help="amplituda skoku [deg]")
    ap.add_argument("--tstep", type=float, default=3.0, help="poczatek komendy [s]")
    ap.add_argument("--duration", type=float, default=1.0,
                    help="czas trwania komendy [s]; 0 = skok trwaly")
    ap.add_argument("--show", action="store_true", help="pokaz okna zamiast zapisu")
    args = ap.parse_args()

    print("=" * 70)
    print("DEMO MODULU STEROWANIA")
    print("=" * 70)

    outs = ctrl_outs()
    have_datcom = bool(outs)
    if have_datcom:
        print(f"Znaleziono {[o.name for o in outs]} — uzywam ZMIERZONYCH pochodnych.")
        from control.datcom_control import build_control_derivatives
        table = build_control_derivatives(outs, sweep_deg=SWEEP,
                                          linear_range_deg=6.0, verbose=False)
        tag = "pochodne z DATCOM"
    else:
        print("Brak datcom_ctrl_*.out — uzywam pochodnych SYNTETYCZNYCH.")
        print("(uruchom: python run_control_datcom.py ctrl --run)")
        table = synthetic_table()
        tag = "pochodne syntetyczne"

    print("\n1. Odpowiedz na impuls sterowania")
    dur = args.duration if args.duration > 0 else None
    plot_step_response(table, args.amp, args.tstep, dur, tag, args.show)

    print("\n2. Pochodne sterowania")
    if have_datcom:
        plot_derivatives(args.show)
    else:
        print("  pominiete — wymaga datcom_ctrl_*.out")

    print("\n" + "=" * 70)
    print(f"Wykresy w: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
