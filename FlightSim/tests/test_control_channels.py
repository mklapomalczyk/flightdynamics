"""
tests/test_control_channels.py
==============================
Test macierzowy TRZECH kanalow sterowania: pitch, yaw, roll.

  wychylenia : 2, 5, 8 deg
  przebieg   : 15 s, impuls 1 s od t = 3 s

Sprawdza, ze modul nadaje sie do sterowania w kazdej osi, a nie tylko
w pitch (jedyny kanal, ktory mial hak w modelu sil przed ta praca):

  1. wlasciwa OS reaguje na wlasciwy kanal,
  2. sprzezenie skrosne jest male wzgledem osi sterowanej,
  3. moment rosnie LINIOWO z wychyleniem,
  4. moment jest zerowy poza oknem impulsu,
  5. zaden przebieg nie konczy sie koziolkowaniem (status == "ok"),
  6. odpowiedz rosnie monotonicznie z amplituda.

Uzywa ZMIERZONYCH pochodnych z DATCOM, jesli sa; inaczej syntetycznych.

Uruchomienie:  python3 tests/test_control_channels.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import (AeroSurfaceEffector, ControlDerivTable, ControlSystem,
                     PassthroughActuator, StepCommander)
from control.demo_model import build_demo_model, fly_demo
from control.types import FlightState

PASS = FAIL = SKIP = 0
AMPS = [2.0, 5.0, 8.0]
T_STEP, T_DUR, T_MAX = 3.0, 1.0, 15.0
SWEEP = [-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.]

# kanal -> (etykieta, atrybut predkosci katowej, os momentu, pozycja w u)
CHANNELS = {
    "d_pitch": ("PITCH", "qr", 1, 0),
    "d_yaw":   ("YAW",   "r",  2, 1),
    "d_roll":  ("ROLL",  "p",  0, 2),
}
RATE_OF_AXIS = {0: "p", 1: "qr", 2: "r"}


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def load_table():
    d = ROOT / "datcom_runs" / "rocket_70mm_canards"
    outs = sorted(d.glob("datcom_ctrl_*.out")) or (
        [d / "datcom_ctrl.out"] if (d / "datcom_ctrl.out").exists() else [])
    if outs:
        from control.datcom_control import build_control_derivatives
        return build_control_derivatives(outs, sweep_deg=SWEEP,
                                         linear_range_deg=6.0, verbose=False), True
    a = np.deg2rad(np.array([-20., -10., 0., 10., 20.]))
    m = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    o = np.ones((len(a), len(m)))
    return ControlDerivTable(alpha_rad=a, mach=m, Cm_delta=2.2*o, Cn_delta=-2.2*o,
                             Cl_delta=0.08*o, CN_delta=1.0*o, CY_delta=-1.0*o,
                             xcg_ref=0.71, S_ref=0.00385, d_ref=0.070), False


print("=" * 72)
print("TEST KANALOW STEROWANIA — pitch / yaw / roll")
print(f"impuls {T_DUR:.0f}s od t={T_STEP:.0f}s, przebieg {T_MAX:.0f}s, "
      f"wychylenia {AMPS} deg")
print("=" * 72)

table, real = load_table()
print(f"pochodne: {'ZMIERZONE (DATCOM)' if real else 'syntetyczne'}")
geom = build_demo_model(None).geom
# S_ref/d_ref z GEOMETRII modelu, nie z LREF DATCOM — patrz docstring efektora.
eff = AeroSurfaceEffector.from_geometry(table, geom)
print(f"S_ref={geom.S_ref:.6f} m^2  d_ref={geom.d_ref:.4f} m")

# Przebieg odniesienia BEZ sterowania. Rakieta i tak pochyla tor pod
# grawitacja (max|q| ~ 7 deg/s), wiec sprzezenie skrosne MUSI byc liczone
# jako PRZYROST wzgledem tego przebiegu, nie jako wartosc bezwzgledna —
# inaczej naturalna rotacja toru liczy sie jako sprzezenie od sterowania
# i przy malych komendach pozornie je przewyzsza.
r_off = fly_demo(None, t_max=T_MAX)

def induced(res, ax):
    """Maksymalny PRZYROST predkosci katowej wzgledem przebiegu bez sterowania."""
    a = getattr(res, RATE_OF_AXIS[ax])
    b = np.interp(res.t, r_off.t, getattr(r_off, RATE_OF_AXIS[ax]))
    return float(np.max(np.abs(a - b)))


results = {}
for channel, (lbl, rate_at, m_axis, u_idx) in CHANNELS.items():
    print(f"\n{lbl}")
    peak_rate, peak_mom = [], []

    for amp in AMPS:
        cs = ControlSystem(StepCommander(T_STEP, amp, channel, duration_s=T_DUR),
                           PassthroughActuator(), [eff])
        res = fly_demo(cs, t_max=T_MAX)

        # 5. brak koziolkowania
        check(f"{lbl} {amp:.0f} deg: przebieg zakonczony poprawnie",
              res.status == "ok", f"(status={res.status})")

        rates = {ax: induced(res, ax) for ax in (0, 1, 2)}
        peak_rate.append(rates[m_axis])

        # 1. wlasciwa os reaguje najmocniej
        others = max(v for ax, v in rates.items() if ax != m_axis)
        check(f"{lbl} {amp:.0f} deg: sterowana os reaguje najmocniej "
              f"(przyrost wzgl. lotu bez sterowania)",
              rates[m_axis] > others,
              f"(os={np.degrees(rates[m_axis]):.1f} vs inne={np.degrees(others):.1f} deg/s)")

        # 2. sprzezenie skrosne male
        ratio = others / max(rates[m_axis], 1e-12)
        check(f"{lbl} {amp:.0f} deg: sprzezenie skrosne < 40% osi sterowanej",
              ratio < 0.40, f"(sprzezenie={ratio*100:.1f}%)")
        if amp == AMPS[0]:
            print(f"    (przyrosty p/q/r = "
                  f"{np.degrees(rates[0]):.2f} / {np.degrees(rates[1]):.2f} / "
                  f"{np.degrees(rates[2]):.2f} deg/s)")

        # 4. moment tylko w oknie impulsu
        fs = FlightState(t=0.0, q_dyn=1.0e4, mach=0.5, speed=200.0)
        u = np.zeros(3); u[u_idx] = amp
        peak_mom.append(abs(eff.wrench(u, fs).M[m_axis]))
        cmd = StepCommander(T_STEP, amp, channel, duration_s=T_DUR)
        u_out = cmd.command(FlightState(t=T_STEP + T_DUR + 1.0)).u
        check(f"{lbl} {amp:.0f} deg: moment zerowy po impulsie",
              abs(eff.wrench(u_out, fs).M[m_axis]) < 1e-12)

    # 6. monotonicznosc odpowiedzi
    check(f"{lbl}: odpowiedz rosnie z amplituda",
          peak_rate[0] < peak_rate[1] < peak_rate[2],
          f"({[round(np.degrees(v),1) for v in peak_rate]} deg/s)")

    # 3. liniowosc momentu wzgledem wychylenia
    m_arr = np.array(peak_mom)
    a_arr = np.array(AMPS)
    rel = np.max(np.abs(m_arr / a_arr - m_arr[0] / a_arr[0])) / (m_arr[0] / a_arr[0])
    check(f"{lbl}: moment liniowy wzgledem wychylenia (<1%)", rel < 0.01,
          f"(rozrzut {rel*100:.2f}%)")
    print(f"    momenty: {[round(v,4) for v in peak_mom]} N*m  "
          f"predkosci: {[round(np.degrees(v),1) for v in peak_rate]} deg/s")
    results[lbl] = (peak_mom, peak_rate)

# Symetria pitch/yaw — uklad krzyzowy, wiec moment powinien byc rowny co do modulu
mp = np.array(results["PITCH"][0]); my = np.array(results["YAW"][0])
rel = float(np.max(np.abs(my - mp) / np.maximum(mp, 1e-12)))
print(f"\nSymetria pitch/yaw: max rozjazd momentu = {rel*100:.2f}%")
check("moment pitch i yaw rowny co do modulu (uklad krzyzowy, <5%)", rel < 0.05,
      f"(rozjazd {rel*100:.2f}%)")

print("\n" + "=" * 72)
total = PASS + FAIL
print(f"  Wynik: {PASS}/{total} testow zaliczonych")
print("  STATUS: OK" if FAIL == 0 else f"  STATUS: {FAIL} FAIL")
print("=" * 72)
sys.exit(1 if FAIL else 0)
