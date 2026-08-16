"""
tests/test_control_step.py
==========================
Test calego lancucha sterowania: command -> actuator -> moment -> 6DOF.

Nie wymaga DATCOM — uzywa SYNTETYCZNEJ tablicy pochodnych sterowania, wiec
sprawdza mechanike lancucha i konwencje znakow, a nie liczby z DATCOM.

Uruchomienie:  python3 tests/test_control_step.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control import (AeroSurfaceEffector, ConstantCommander, ControlSystem,
                     ControlDerivTable, ControlWrench, PassthroughActuator,
                     StepCommander, ZeroCommander)
from control.types import FlightState
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF
from forces.force_model6 import (ForceModel6DOF, PropulsionConfig6DOF,
                                 RocketGeometry6DOF)
from models.aerodynamics import TableAero
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from models.mass6 import ConstantIxx, MassModel6DOF

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


# --------------------------------------------------------------------------
# Modele pomocnicze
# --------------------------------------------------------------------------
def synthetic_table(Cm_d=+1.0, Cn_d=-1.0, Cl_d=+0.2, CN_d=+2.0, CY_d=+2.0):
    """Stale pochodne sterowania [1/rad] — zeby wynik byl przewidywalny."""
    alpha = np.deg2rad(np.array([-10.0, 0.0, 10.0]))
    mach  = np.array([0.1, 1.0, 3.0])
    ones  = np.ones((len(alpha), len(mach)))
    return ControlDerivTable(
        alpha_rad=alpha, mach=mach,
        Cm_delta=Cm_d * ones, Cn_delta=Cn_d * ones, Cl_delta=Cl_d * ones,
        CN_delta=CN_d * ones, CY_delta=CY_d * ones,
        xcg_ref=0.71, S_ref=0.00385, d_ref=0.070, delta_ref_deg=5.0,
    )


def build_model(control=None):
    """Minimalny, ale realistyczny model 70mm."""
    geom = RocketGeometry6DOF.from_diameter(0.070, xcp=0.95)
    # Masa praktycznie stala (m_empty tuz ponizej m_full, xcg/Iyy identyczne),
    # zeby test dotyczyl sterowania, a nie zmiennosci masy.
    mass = MassModel6DOF(
        m_full=4.2, m_empty=4.19, t_burn=1.0,
        xcg_full=0.71, xcg_empty=0.71,
        Iyy_full=0.646, Iyy_empty=0.646,
        ixx_model=ConstantIxx(0.0032),
    )
    prop = PropulsionConfig6DOF(thrust=0.0)
    return ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"), mass_model=mass,
        aero_model=TableAero(
            alpha_table=np.deg2rad(np.array([-10.0, 0.0, 10.0])),
            mach_table=np.array([0.1, 1.0, 3.0]),
            CA_table=0.4 * np.ones((3, 3)),
            CN_table=np.array([[-2.0]*3, [0.0]*3, [2.0]*3]),
            Cm_table=np.array([[+0.6]*3, [0.0]*3, [-0.6]*3]),  # statycznie stabilny
            xcg_ref=0.71,
        ),
        gravity=create_gravity("constant"), geometry=geom, propulsion=prop,
        launcher=LauncherConfig(L_rail=0.0), control=control,
    )


def fly(control, t_max=8.0):
    fm = build_model(control)
    st = State6DOF.initial(elevation_deg=80.0, azimuth_deg=0.0, speed_0=200.0)
    return run_simulation_6dof(fm, st, t_max=t_max, dt_output=0.02,
                               z_ground=-1e9)   # bez przerywania o grunt


# --------------------------------------------------------------------------
print("=" * 62)
print("TEST LANCUCHA STEROWANIA (command -> actuator -> moment -> 6DOF)")
print("=" * 62)

# --- 1. Bloki osobno ------------------------------------------------------
print("\n1. Bloki lancucha")
fs0 = FlightState(t=0.0, q_dyn=1.0e4, mach=0.5, speed=200.0)
fs5 = FlightState(t=5.0, q_dyn=1.0e4, mach=0.5, speed=200.0)

sc = StepCommander(t_step=3.0, amplitude_deg=2.0, channel="d_pitch")
check("StepCommander: 0 przed skokiem", np.allclose(sc.command(fs0).u, 0.0))
check("StepCommander: amplituda po skoku", sc.command(fs5).get("d_pitch") == 2.0)

act = PassthroughActuator()
cmd = sc.command(fs5)
check("PassthroughActuator: wyjscie == zadanie",
      np.allclose(act.output(cmd, fs5), cmd.u))
check("PassthroughActuator: bez stanow ODE", act.n_states == 0)

eff = AeroSurfaceEffector(synthetic_table())
w0 = eff.wrench(np.zeros(3), fs5)
check("Efektor: zerowe wychylenie -> zerowy wrench",
      np.allclose(w0.M, 0.0) and np.allclose(w0.F, 0.0))
w_noq = eff.wrench(np.array([2.0, 0, 0]), FlightState(t=5.0, q_dyn=0.0))
check("Efektor: q_dyn=0 -> zerowy wrench", np.allclose(w_noq.M, 0.0))

w1 = eff.wrench(np.array([2.0, 0.0, 0.0]), fs5)
w2 = eff.wrench(np.array([4.0, 0.0, 0.0]), fs5)
check("Efektor: moment liniowy wzgledem wychylenia",
      abs(w2.My - 2.0 * w1.My) < 1e-9 * max(1.0, abs(w2.My)))
fs_2q = FlightState(t=5.0, q_dyn=2.0e4, mach=0.5, speed=200.0)
check("Efektor: moment liniowy wzgledem q_dyn",
      abs(eff.wrench(np.array([2.0, 0, 0]), fs_2q).My - 2.0 * w1.My) < 1e-9 * abs(w2.My))

check("ControlWrench.__add__ sumuje", (w1 + w1).My == 2.0 * w1.My)

# --- 2. Konwencja znakow --------------------------------------------------
# Kanaly rozdzielone: kazda os osobno, zeby nie maskowaly sie nawzajem.
print("\n2. Konwencja znakow (na zamrozonym stanie)")
tab = synthetic_table(Cm_d=+1.0, Cn_d=+1.0, Cl_d=+1.0)
eff_s = AeroSurfaceEffector(tab)
for i, (ch, axis, lbl) in enumerate([("d_pitch", 1, "MY"), ("d_yaw", 2, "MZ"),
                                     ("d_roll", 0, "MX")]):
    u = np.zeros(3); u[i] = +1.0
    W = eff_s.wrench(u, fs5)
    check(f"+{ch} -> +{lbl} (dodatnia pochodna)", W.M[axis] > 0.0,
          f"({lbl}={W.M[axis]:.4g})")

# Ten sam znak musi dotrzec do pochodnych stanu przez model sil.
fm = build_model(ControlSystem(
    commander=ConstantCommander({"d_pitch": 2.0}),
    actuator=PassthroughActuator(),
    effectors=[AeroSurfaceEffector(synthetic_table(Cm_d=+1.0))]))
st = State6DOF.initial(elevation_deg=0.0, azimuth_deg=0.0, speed_0=200.0)
x = st.to_numpy(include_rail=True)
dx = fm.derivatives(0.0, x)
check("+d_pitch -> dq/dt > 0 (nos w gore) w modelu sil", dx[11] > 0.0,
      f"(dq/dt={dx[11]:.4g})")

# --- 3. Brak sterowania == stan sprzed zmiany ----------------------------
print("\n3. Neutralnosc: control=None / zerowa komenda")
r_none = fly(None)
r_zero = fly(ControlSystem(ZeroCommander(), PassthroughActuator(),
                           [AeroSurfaceEffector(synthetic_table())]))
r_amp0 = fly(ControlSystem(StepCommander(3.0, 0.0, "d_pitch"),
                           PassthroughActuator(),
                           [AeroSurfaceEffector(synthetic_table())]))
check("control=None vs ZeroCommander: identyczne",
      np.allclose(r_none.theta, r_zero.theta, atol=1e-12))
check("control=None vs amplituda 0: identyczne",
      np.allclose(r_none.theta, r_amp0.theta, atol=1e-12))
check("ControlSystem bez efektorow: zerowy wrench",
      np.allclose(ControlSystem(sc, act, []).compute(fs5).M, 0.0))

# --- 4. Test skokowy end-to-end ------------------------------------------
print("\n4. Skok +2 deg w pitch przy t=3 s")
r_step = fly(ControlSystem(StepCommander(3.0, +2.0, "d_pitch"),
                           PassthroughActuator(),
                           [AeroSurfaceEffector(synthetic_table(Cm_d=+1.0))]))
pre = r_step.t < 2.9
post = r_step.t > 3.1
check("przed skokiem: trajektoria == bez sterowania",
      np.allclose(r_step.theta[pre], r_none.theta[pre], atol=1e-9))
d_theta = np.degrees(r_step.theta[-1] - r_none.theta[-1])
check("po skoku: theta odchylona od przypadku bez sterowania",
      abs(d_theta) > 0.5, f"(dtheta={d_theta:.3f} deg)")
check("po skoku: znak zgodny z +Cm_delta (nos w gore)", d_theta > 0.0,
      f"(dtheta={d_theta:.3f} deg)")
q_pre = float(np.max(np.abs(r_step.qr[pre])))
q_post = float(np.max(np.abs(r_step.qr[post])))
check("po skoku: predkosc katowa pitch rosnie",
      q_post > q_pre, f"(|q| {q_pre:.4g} -> {q_post:.4g} rad/s)")
check("symulacja zakonczona poprawnie", r_step.status == "ok",
      f"(status={r_step.status})")

# --- 5. Odwrocony znak pochodnej -> odwrocona reakcja ---------------------
print("\n5. Kontrola: odwrocenie znaku Cm_delta odwraca reakcje")
r_neg = fly(ControlSystem(StepCommander(3.0, +2.0, "d_pitch"),
                          PassthroughActuator(),
                          [AeroSurfaceEffector(synthetic_table(Cm_d=-1.0))]))
d_theta_neg = np.degrees(r_neg.theta[-1] - r_none.theta[-1])
check("Cm_delta<0 -> theta w przeciwna strone", d_theta_neg < 0.0,
      f"(dtheta={d_theta_neg:.3f} deg)")

print("\n" + "=" * 62)
print(f"  Wynik: {PASS}/{PASS + FAIL} testow zaliczonych")
print("  STATUS: OK" if FAIL == 0 else f"  STATUS: {FAIL} FAIL")
print("=" * 62)
sys.exit(1 if FAIL else 0)
