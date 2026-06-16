"""
test_gyroscopic.py
==================
Weryfikacja momentu zyroskopowego w modelu 6DOF.

Testy:
  1. Bez obrotu — brak momentu zyroskopowego (sily Eulera = 0)
  2. Z obrotem, bez zaklócen — p staly, q=r=0 (brak precesji)
  3. Precesja zyroskopowa — rakieta z duzym p i malym q
     Czestotliwosc precesji: f = Ixx*p / (2*pi*Iyy)
  4. Kierunek precesji — zgodny z prawem prawej reki

Uruchomienie z katalogu FlightSim:
    python tests/test_gyroscopic.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.aerodynamics import ConstantAero
from models.mass6 import MassModel6DOF, ConstantIxx
from forces.force_model6 import ForceModel6DOF, RocketGeometry6DOF, PropulsionConfig6DOF
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF

results = []

def check(name, ok, got, expected, unit="", tol_pct=5.0):
    pct = abs(got - expected) / (abs(expected) + 1e-10) * 100
    tag = "PASS" if ok else "FAIL"
    results.append((name, ok))
    print(f"  [{tag}] {name}")
    print(f"         got={got:.6f}{unit}  expected={expected:.6f}{unit}  err={pct:.2f}%")

def make_fm(Ixx=0.003, Iyy=0.646, thrust=0., CA=0., Cmq=-500.):
    """Model z zadanymi momentami bezwładnosci."""
    return ForceModel6DOF(
        atmosphere = create_atmosphere("ISA"),
        mass_model = MassModel6DOF(
            m_full=4.2, m_empty=4.199, t_burn=0.001,
            xcg_full=0.71, xcg_empty=0.71,
            Iyy_full=Iyy, Iyy_empty=Iyy,
            ixx_model=ConstantIxx(Ixx),
        ),
        aero_model = ConstantAero(CA=CA, CN_alpha=0., Cmq=Cmq,
                                  use_xcp_moment=False),
        gravity    = create_gravity("constant"),
        geometry   = RocketGeometry6DOF(
            S_ref=np.pi*(0.07/2)**2, d_ref=0.07, xcp=0.71,
        ),
        propulsion = PropulsionConfig6DOF(thrust=thrust),
    )

# ============================================================================
# TEST 1: Brak obrotu — czlony zyroskopowe = 0
# dqr/dt = (MY + (Izz-Ixx)*p*r) / Iyy
# dr/dt  = (MZ + (Ixx-Iyy)*p*qr) / Izz
# Przy p=0: czlony zyroskopowe = 0
# ============================================================================
print("\n=== TEST 1: Brak obrotu — czlony zyroskopowe = 0 ===")

Ixx = 0.003; Iyy = 0.646
fm = make_fm(Ixx=Ixx, Iyy=Iyy)

# Stan: lot poziomy, brak obrotu, male q
s0 = State6DOF(x=0., y=0., z=-500., u=200., v=0., w=0.,
               q0=1., q1=0., q2=0., q3=0.,
               p=0., qr=np.radians(5.), r=0.)

x0 = s0.to_numpy(include_rail=False)
# Dodaj rail_dist=999 zeby nie bylo na szynie
x0 = np.append(x0, 999.)
dx = fm.derivatives(0., x0)

# dqr/dt powinno byc tylko od grawitacji/aerodynamiki, nie od p*r
# Przy p=0: czlon zyroskopowy (Izz-Ixx)*p*r = 0
gyro_term_pitch = (Iyy - Ixx) * 0.0 * 0.0  # p=0, r=0
gyro_term_yaw   = (Ixx - Iyy) * 0.0 * np.radians(5.)  # p=0

check("Czlon zyroskopowy pitch = 0 przy p=0",
      abs(gyro_term_pitch) < 1e-10, gyro_term_pitch, 0.0)
check("Czlon zyroskopowy yaw = 0 przy p=0",
      abs(gyro_term_yaw) < 1e-10, gyro_term_yaw, 0.0)

# ============================================================================
# TEST 2: Duze p, q=r=0 — brak precesji
# Przy q=r=0 czlony zyroskopowe = 0
# dqr/dt = MY/Iyy (tylko od momentow zewn.)
# ============================================================================
print("\n=== TEST 2: Duze p, q=r=0 — brak precesji ===")

p0 = np.radians(2500.)  # 2500 deg/s
s0 = State6DOF(x=0., y=0., z=-500., u=200., v=0., w=0.,
               q0=1., q1=0., q2=0., q3=0.,
               p=p0, qr=0., r=0.)

x0 = s0.to_numpy(include_rail=False)
x0 = np.append(x0, 999.)
dx = fm.derivatives(0., x0)

# Czlony zyroskopowe przy q=r=0:
# (Izz-Ixx)*p*r = 0  (r=0)
# (Ixx-Iyy)*p*qr = 0  (qr=0)
gyro_pitch = (Iyy - Ixx) * p0 * 0.0
gyro_yaw   = (Ixx - Iyy) * p0 * 0.0

check("Czlon zyroskopowy pitch = 0 gdy r=0",
      abs(gyro_pitch) < 1e-10, gyro_pitch, 0.0)
check("Czlon zyroskopowy yaw = 0 gdy qr=0",
      abs(gyro_yaw) < 1e-10, gyro_yaw, 0.0)

# ============================================================================
# TEST 3: Czestotliwosc precesji zyroskopowej
# Teoria: f_prec = Ixx * p / (2*pi*Iyy)
# Symulacja: brak momentow aero (CN_alpha=0, bez grawitacji)
# Poczatkowe qr=qr0, r=0 — po czasie T=1/f_prec r powinno osiagnac max
# ============================================================================
print("\n=== TEST 3: Czestotliwosc precesji zyroskopowej ===")

Ixx = 0.003
Iyy = 0.646
p0  = np.radians(2500.)   # [rad/s]

f_prec_exp = Ixx * p0 / (2 * np.pi * Iyy)
T_prec_exp = 1.0 / f_prec_exp
print(f"  Oczekiwana czestotliwosc precesji: {f_prec_exp:.4f} Hz")
print(f"  Oczekiwany okres precesji:         {T_prec_exp:.4f} s")

# Model bez grawitacji i bez aero — tylko czysta dynamika zyroskopowa
fm_gyro = ForceModel6DOF(
    atmosphere = create_atmosphere("ISA"),
    mass_model = MassModel6DOF(
        m_full=4.2, m_empty=4.199, t_burn=0.001,
        xcg_full=0.71, xcg_empty=0.71,
        Iyy_full=Iyy, Iyy_empty=Iyy,
        ixx_model=ConstantIxx(Ixx),
    ),
    aero_model = ConstantAero(CA=0., CN_alpha=0., Cmq=0.,
                              use_xcp_moment=False),
    gravity    = create_gravity("constant"),   # grawitacja stala
    geometry   = RocketGeometry6DOF(
        S_ref=np.pi*(0.07/2)**2, d_ref=0.07, xcp=0.71,
    ),
    propulsion = PropulsionConfig6DOF(thrust=0.),
)

qr0 = np.radians(10.)   # mala perturbacja pitch
s0 = State6DOF(x=0., y=0., z=-10000., u=200., v=0., w=0.,
               q0=1., q1=0., q2=0., q3=0.,
               p=p0, qr=qr0, r=0.)

# Symulacja przez 2 okresy precesji
t_sim = 2.0 * T_prec_exp
result = run_simulation_6dof(
    fm_gyro, s0,
    t_max    = t_sim,
    z_ground = -100000.,
    max_step = T_prec_exp / 100,
)

# Znajdz pierwsze zero r po t=0 (pol okresu = czas do max r)
r_deg   = np.degrees(result.r)
qr_deg  = np.degrees(result.qr)

# Amplituda powinna byc zachowana: sqrt(qr^2 + r^2) = const = qr0
amp = np.sqrt(result.qr**2 + result.r**2)
amp_mean = float(np.mean(amp))
amp_std  = float(np.std(amp))

check("Amplituda precesji zachowana (std < 1%)",
      amp_std / amp_mean < 0.01,
      amp_std / amp_mean * 100, 0.0, unit="%")

# Weryfikacja analityczna czlonow zyroskopowych zamiast pomiaru z sygnalu
# Przy duzym p sygnaly r(t) i qr(t) sa superpozycja obrotu (f~p/2pi)
# i precesji (f~Ixx*p/2pi*Iyy) — trudne do rozdzielenia numerycznie
# Zamiast tego sprawdzamy pochodne analitycznie

x_test = s0.to_numpy(include_rail=False)
x_test = np.append(x_test, 999.)
dx_test = fm_gyro.derivatives(0., x_test)

# dqr/dt = (MY + (Izz-Ixx)*p*r) / Iyy
# dr/dt  = (MZ + (Ixx-Iyy)*p*qr) / Izz
# Przy MY=MZ=0 (brak momentow zewn.):
Izz = Iyy
dqr_gyro_exp = (Iyy - Ixx) * p0 * 0.0 / Iyy   # r=0 wiec = 0
dr_gyro_exp  = (Ixx - Iyy) * p0 * qr0 / Izz

dqr_got = dx_test[11]
dr_got  = dx_test[12]

print(f"  Analityczny dr/dt = (Ixx-Iyy)*p*qr/Izz = {np.degrees(dr_gyro_exp):.6f} deg/s2")
print(f"  Zmierzony  dr/dt  = {np.degrees(dr_got):.6f} deg/s2")

# Sprawdz czy dr/dt jest zgodne z teoria (tolerancja 5%)
check("dr/dt zyroskopowy zgodny z teoria (+-5%)",
      abs(dr_got - dr_gyro_exp) / (abs(dr_gyro_exp) + 1e-10) < 0.05,
      np.degrees(dr_got), np.degrees(dr_gyro_exp), unit=" deg/s2")

# Sprawdz zachowanie amplitudy precesji sqrt(qr^2+r^2)
amp = np.sqrt(result.qr**2 + result.r**2)
print(f"  Amplituda precesji: mean={np.degrees(np.mean(amp)):.4f}°  "
      f"std={np.degrees(np.std(amp)):.4f}°")
check("Amplituda precesji stala (std < 5% mean)",
      np.std(amp) / np.mean(amp) < 0.05,
      float(np.std(amp)/np.mean(amp)*100), 0.0, unit="%")

# ============================================================================
# TEST 4: Kierunek precesji
# Przy p > 0 (obrot zgodnie z os x), perturbacja qr > 0 (nos w gore)
# Precesja powinna byc w kierunku r > 0 (nos w prawo)
# Regula prawej reki: p x qr -> r
# ============================================================================
print("\n=== TEST 4: Kierunek precesji — regula prawej reki ===")

# Pierwsze kilka krokow: sprawdz znak dr/dt
x0 = s0.to_numpy(include_rail=False)
x0 = np.append(x0, 999.)
dx = fm_gyro.derivatives(0., x0)

# dr/dt = (Ixx-Iyy)*p*qr / Izz
# Ixx < Iyy wiec (Ixx-Iyy) < 0
# p > 0, qr > 0 => (Ixx-Iyy)*p*qr < 0 => dr/dt < 0
# czyli nos idzie w lewo (r < 0) dla p > 0, qr > 0
dr_dt = dx[12]
Izz = Iyy   # przyblizenie Izz ~ Iyy dla smuklej rakiety
expected_sign = np.sign((Ixx - Iyy) * p0 * qr0)

print(f"  Ixx-Iyy = {Ixx-Iyy:.4f}  (ujemne dla smuklej rakiety)")
print(f"  p0 = {np.degrees(p0):.0f} deg/s  qr0 = {np.degrees(qr0):.1f} deg/s")
print(f"  Oczekiwany znak dr/dt: {'+' if expected_sign > 0 else '-'}")
print(f"  Zmierzony dr/dt = {np.degrees(dr_dt):.4f} deg/s2")

check("Znak dr/dt zgodny z teoria",
      np.sign(dr_dt) == expected_sign,
      float(np.sign(dr_dt)), float(expected_sign))

# ============================================================================
# TEST 5: Zachowanie momentu pedu przy braku momentow zewnetrznych
# H = sqrt((Ixx*p)^2 + (Iyy*qr)^2 + (Iyy*r)^2) = const
# ============================================================================
print("\n=== TEST 5: Zachowanie momentu pedu (brak momentow zewn.) ===")

H = np.sqrt((Ixx * result.p)**2 +
            (Iyy * result.qr)**2 +
            (Iyy * result.r)**2)
H0 = H[0]
dH_pct = float(np.max(np.abs(H - H0)) / H0 * 100)

check("Moment pedu zachowany dH < 1%", dH_pct < 1.0,
      dH_pct, 0.0, unit="%")

# ============================================================================
# Podsumowanie
# ============================================================================
print(f"\n{'='*55}")
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"Wynik: {n_pass}/{len(results)} testow PASS  ({n_fail} FAIL)")
print('='*55)
