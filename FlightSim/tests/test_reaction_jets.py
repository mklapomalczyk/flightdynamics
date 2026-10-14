"""
tests/test_reaction_jets.py
===========================
Testy sterowania REACTION JETS:
    komenda (silnik) -> zaplon -> impuls ciagu -> odpowiedz 6DOF

Nie wymaga DATCOM. Testy calkowe uzywaja modelu "prozniowego" (zerowa aero,
zerowe tlumienie), wiec wynik da sie policzyc recznie z zasady pedu:
    dq = M * dt / Iyy,   dp = Mx * dt / Ixx,   dv = F * dt / m

Uruchomienie:  python3 tests/test_reaction_jets.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control import (JetCommander, JetLayout, JetRow, JetThrustProfile,
                     NoJetCommander, ReactionJetEffector, ReactionJetSystem,
                     ScheduleJetCommander, build_reaction_jet_system)
from control.types import FlightState
from core.quaternion import quat_to_dcm
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF
from datcom_io.config_reader import load_config
from forces.force_model6 import (ForceModel6DOF, PropulsionConfig6DOF,
                                 RocketGeometry6DOF)
from models.aerodynamics import TableAero
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from models.mass6 import ConstantIxx, MassModel6DOF

ROOT = Path(__file__).resolve().parent.parent

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


def rel_err(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


# --------------------------------------------------------------------------
# Model "prozniowy": zerowa aerodynamika i tlumienie => czysta zasada pedu.
# --------------------------------------------------------------------------
M_ROCKET, XCG, IYY, IXX = 4.2, 0.71, 0.646, 0.0032
R_BODY = 0.035


def build_model(control=None):
    a = np.deg2rad(np.array([-10.0, 0.0, 10.0]))
    mach = np.array([0.1, 1.0, 3.0])
    z = np.zeros((3, 3))
    mass = MassModel6DOF(m_full=M_ROCKET, m_empty=M_ROCKET - 1e-6, t_burn=0.5,
                         xcg_full=XCG, xcg_empty=XCG,
                         Iyy_full=IYY, Iyy_empty=IYY,
                         ixx_model=ConstantIxx(IXX))
    return ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"), mass_model=mass,
        aero_model=TableAero(alpha_table=a, mach_table=mach, CA_table=z,
                             CN_table=z, Cm_table=z, Cmq=0.0, xcg_ref=XCG),
        gravity=create_gravity("constant"),
        geometry=RocketGeometry6DOF.from_diameter(2 * R_BODY, xcp=0.95),
        propulsion=PropulsionConfig6DOF(thrust=0.0),
        launcher=LauncherConfig(L_rail=0.0), control=control)


def fly(control, t_max=3.0, max_step=0.05, dt_output=0.01):
    st = State6DOF.initial(elevation_deg=80.0, azimuth_deg=0.0, speed_0=200.0)
    return run_simulation_6dof(build_model(control), st, t_max=t_max,
                               dt_output=dt_output, z_ground=-1e9,
                               max_step=max_step)


def jets(rows=None, thrust=100.0, burn=0.01, delay=0.0, commander=None,
         rate=None, shape="rectangular", tilt_a=0.0, tilt_t=0.0):
    rows = rows if rows is not None else [JetRow(0.30, 8, 0.0), JetRow(0.34, 8, 22.5)]
    lay = JetLayout(rows, R_BODY, tilt_axial_deg=tilt_a, tilt_tangential_deg=tilt_t)
    eff = ReactionJetEffector(lay, JetThrustProfile(thrust, burn, shape))
    return ReactionJetSystem(eff, commander, ignition_delay=delay,
                             control_rate_hz=rate)


def at(res, t):
    return int(np.argmin(np.abs(res.t - t)))


def v_launch(res, i):
    q = np.array([res.q0[i], res.q1[i], res.q2[i], res.q3[i]])
    return quat_to_dcm(q).T @ np.array([res.u[i], res.v[i], res.w[i]])


FS = FlightState(t=1.0, xcg=XCG)   # stan zamrozony do testow efektora


# ==========================================================================
print("=" * 66)
print("TEST REACTION JETS (komenda -> zaplon -> impuls -> 6DOF)")
print("=" * 66)

# --- 1. Profil ciagu ------------------------------------------------------
print("\n1. Profil ciagu")
pr = JetThrustProfile(100.0, 0.01)
check("prostokatny: impuls = F*t", abs(pr.impulse - 1.0) < 1e-12)
check("prostokatny: staly ciag", np.allclose(pr(np.array([0.0, 0.005, 0.01])), 100.0))

shape = [[0.0, 0.0], [0.2, 1.5], [0.7, 1.0], [1.0, 0.0]]
pc = JetThrustProfile(100.0, 0.01, shape)
tau = np.linspace(0.0, 0.01, 20001)
f = pc(tau)
impulse_num = float(np.sum(np.diff(tau) * (f[1:] + f[:-1]) / 2.0))
check("profil wlasny: impuls zachowany (= F*t)", rel_err(impulse_num, 1.0) < 1e-6,
      f"{impulse_num:.6f}")
check("profil wlasny: zero na poczatku", abs(pc(np.array([0.0]))[0]) < 1e-12)
check("profil wlasny: wezly w czasie bezwzgl.", np.allclose(pc.knots, [0, 0.002, 0.007, 0.01]))

for bad, why in [([[0.1, 1.0], [1.0, 1.0]], "start != 0"),
                 ([[0.0, 1.0], [0.5, 1.0]], "koniec != 1"),
                 ([[0.0, 1.0], [0.6, 1.0], [0.4, 1.0], [1.0, 1.0]], "t nie rosnie"),
                 ([[0.0, -1.0], [1.0, 1.0]], "F ujemne"),
                 ([[0.0, 0.0], [1.0, 0.0]], "zerowy impuls"),
                 ("triangle", "nieznana nazwa")]:
    try:
        JetThrustProfile(100.0, 0.01, bad)
        check(f"profil: odrzuca bledny ({why})", False)
    except ValueError:
        check(f"profil: odrzuca bledny ({why})", True)

# --- 2. Geometria ---------------------------------------------------------
print("\n2. Geometria ukladu silnikow")
lay = JetLayout([JetRow(0.30, 8, 0.0), JetRow(0.34, 8, 22.5)], R_BODY)
check("liczba silnikow 2x8", lay.n_engines == 16)
check("nazwy R1E1..R2E8", lay.names[0] == "R1E1" and lay.names[-1] == "R2E8")
check("rowny podzial 45 deg", np.allclose(np.degrees(lay.phi[:8]), np.arange(8) * 45.0))
check("przesuniecie fazowe rzedu 2 (22.5 deg)",
      np.allclose(np.degrees(lay.phi[8:]), 22.5 + np.arange(8) * 45.0))
check("R1E1 na +Z", np.allclose(lay.e_r[0], [0, 0, 1]))
check("R1E3 na +Y", np.allclose(lay.e_r[2], [0, 1, 0]))
check("wersory kierunku jednostkowe", np.allclose(np.linalg.norm(lay.direction, axis=1), 1.0))
check("indeks z nazwy / liczby", lay.index("r2e5") == 12 and lay.index(12) == 12)
for bad in ["R3E1", 99]:
    try:
        lay.index(bad)
        check(f"indeks: odrzuca {bad!r}", False)
    except ValueError:
        check(f"indeks: odrzuca {bad!r}", True)
lay_t = JetLayout([JetRow(0.3, 4)], R_BODY, tilt_axial_deg=20.0, tilt_tangential_deg=10.0)
check("pochylenia: wersory dalej jednostkowe",
      np.allclose(np.linalg.norm(lay_t.direction, axis=1), 1.0))

# --- 3. Efektor (sily i momenty liczone recznie) ---------------------------
print("\n3. Efektor — sily i momenty")
T, DX = 100.0, XCG - 0.30
eff = ReactionJetEffector(JetLayout([JetRow(0.30, 4, 0.0)], R_BODY),
                          JetThrustProfile(T, 0.01))
inf = np.inf


def wr(t_ign, fs=FS, e=eff):
    return e.wrench(np.asarray(t_ign, float), fs)


w = wr([1.0, inf, inf, inf])
check("silnik na +Z: F = (0,0,-T)", np.allclose(w.F, [0, 0, -T]))
check("silnik na +Z przed SC: My = +T*dx (nos w gore)", np.allclose(w.M, [0, T * DX, 0]))
w = wr([inf, 1.0, inf, inf])
check("silnik na +Y: F = (0,-T,0)", np.allclose(w.F, [0, -T, 0]))
check("silnik na +Y przed SC: Mz = -T*dx (nos w lewo)", np.allclose(w.M, [0, 0, -T * DX]))
w = wr([1.0, inf, 1.0, inf])
check("przeciwlegle silniki znosza sie", np.allclose(w.F, 0) and np.allclose(w.M, 0))
w = wr([inf] * 4)
check("brak odpalonych -> zerowy wrench", np.allclose(w.F, 0) and np.allclose(w.M, 0))
w = wr([0.5, inf, inf, inf])
check("po wygaszeniu -> zero", np.allclose(w.F, 0))
w = wr([1.0005, inf, inf, inf])
check("przed zaplonem -> zero", np.allclose(w.F, 0))
w = wr([1.0, inf, inf, inf], FlightState(t=1.0, xcg=0.30))
check("rzad w SC: sila bez momentu", np.allclose(w.M, 0) and np.allclose(w.F, [0, 0, -T]))
w = wr([1.0, inf, inf, inf], FlightState(t=1.0, xcg=0.20))
check("rzad ZA SC: znak momentu odwrocony", w.M[1] < 0)

a_t = 10.0
eff_t = ReactionJetEffector(JetLayout([JetRow(0.30, 4)], R_BODY, tilt_tangential_deg=a_t),
                            JetThrustProfile(T, 0.01))
w = wr([1.0, 1.0, 1.0, 1.0], e=eff_t)
check("pochylenie obwodowe: 4 silniki -> czysty roll +4RTsin",
      np.allclose(w.M, [4 * R_BODY * T * np.sin(np.radians(a_t)), 0, 0]) and np.allclose(w.F, 0))
eff_a = ReactionJetEffector(JetLayout([JetRow(0.30, 4)], R_BODY, tilt_axial_deg=a_t),
                            JetThrustProfile(T, 0.01))
w = wr([1.0, inf, inf, inf], e=eff_a)
check("pochylenie osiowe: Fx = T sin, Fz = -T cos",
      np.allclose(w.F, [T * np.sin(np.radians(a_t)), 0, -T * np.cos(np.radians(a_t))]))
w = wr([1.0, inf, inf, inf])
check("diagnostyka efektora", w.diag["jet_n_active"] == 1.0 and abs(w.diag["jet_My"] - T * DX) < 1e-9)

# --- 4. System: komendy, one-shot, opoznienie -------------------------------
print("\n4. System — komendy i pamiec silnikow")
s = jets(delay=0.05)
ev = s.fire(["R1E1"], 2.0)
check("odpalenie: t_ign = t_cmd + delay", ev[0].status == "fired" and abs(ev[0].t_ign - 2.05) < 1e-12)
check("odpalenie: t_end = t_ign + burn", abs(ev[0].t_end - 2.06) < 1e-12)
ev = s.fire(["R1E1"], 3.0)
check("ponowne odpalenie zuzytego -> odrzucone", ev[0].status == "rejected_spent")
check("odrzucenie nie zmienia czasu zaplonu", abs(s.t_ign[0] - 2.05) < 1e-12)
check("log zawiera oba zdarzenia", len(s.events) == 2)
check("granice: zaplon i wygaszenie", np.allclose(s.switch_times(), [2.05, 2.06]))
check("next_switch_after", abs(s.next_switch_after(2.05) - 2.06) < 1e-12)
s.reset()
check("reset: wszystkie silniki dostepne", np.all(np.isinf(s.t_ign)) and not s.events)

sc = ScheduleJetCommander([(2.0, "R1E1"), (2.0, ["R1E2", 9]), (4.0, "R2E1")])
check("harmonogram: planned_times", sorted(set(sc.planned_times())) == [2.0, 4.0])
from control import JetTickContext
mk = lambda t, tp: JetTickContext(t=t, t_prev=tp, x=np.zeros(14),
                                  t_ign=(inf,) * 16, names=lay.names)
check("harmonogram: wydaje w (t_prev, t]", list(sc.decide(mk(2.0, 1.99))) == ["R1E1", "R1E2", 9])
check("harmonogram: nie powtarza", list(sc.decide(mk(2.01, 2.0))) == [])
check("harmonogram: zaleglosci gdy tick pozniej", list(sc.decide(mk(4.5, 3.0))) == ["R2E1"])

# --- 5. Calkowanie: impuls a zasada pedu -----------------------------------
print("\n5. Calkowanie — zasada pedu (model prozniowy)")
r0 = fly(None)
t_f = 1.5

res = fly(jets(commander=ScheduleJetCommander([(t_f, "R1E1")])))
i0, i1 = at(res, t_f - 0.1), at(res, t_f + 0.5)
check("bez wplywu przed impulsem", np.allclose(res.qr[:i0], r0.qr[:i0], atol=1e-12)
      and np.allclose(res.z[:i0], r0.z[:i0], atol=1e-9))
dq_exp = 1.0 * DX / IYY
dq = res.qr[i1] - r0.qr[i1]
check("dq = I*dx/Iyy (10 ms, 100 N)", rel_err(dq, dq_exp) < 1e-3, f"{dq:.6f} vs {dq_exp:.6f}")

# Impuls 1 ms << max_step 50 ms: nie moze zostac przeskoczony.
res = fly(jets(thrust=1000.0, burn=0.001, commander=ScheduleJetCommander([(t_f, "R1E1")])),
          max_step=0.05)
dq = res.qr[i1] - r0.qr[i1]
check("impuls 1 ms przy max_step 50 ms nie jest gubiony", rel_err(dq, dq_exp) < 1e-3,
      f"{dq:.6f} vs {dq_exp:.6f}")

# Profil wlasny o tym samym impulsie -> ta sama zmiana pedu.
res = fly(jets(shape=shape, commander=ScheduleJetCommander([(t_f, "R1E1")])))
dq = res.qr[i1] - r0.qr[i1]
check("profil wlasny: ten sam impuls -> to samo dq", rel_err(dq, dq_exp) < 1e-3,
      f"{dq:.6f} vs {dq_exp:.6f}")

# Silnik w SC: czysta zmiana predkosci, bez obrotu.
res = fly(jets(rows=[JetRow(XCG, 4, 0.0)], commander=ScheduleJetCommander([(t_f, "R1E1")])))
dv = np.linalg.norm(v_launch(res, i1) - v_launch(r0, i1))
check("silnik w SC: |dv| = I/m", rel_err(dv, 1.0 / M_ROCKET) < 1e-3,
      f"{dv:.6f} vs {1.0 / M_ROCKET:.6f}")
check("silnik w SC: brak obrotu", abs(res.qr[i1] - r0.qr[i1]) < 1e-9)

# Pochylenie obwodowe: dp = 4*R*I*sin/Ixx (4 silniki naraz).
res = fly(jets(rows=[JetRow(0.30, 4, 0.0)], tilt_t=10.0,
               commander=ScheduleJetCommander([(t_f, ["R1E1", "R1E2", "R1E3", "R1E4"])])))
dp_exp = 4 * R_BODY * 1.0 * np.sin(np.radians(10.0)) / IXX
dp = res.p[i1] - r0.p[i1]
check("pochylenie obwodowe: dp = 4RI sin/Ixx", rel_err(dp, dp_exp) < 1e-3,
      f"{dp:.5f} vs {dp_exp:.5f}")

# Dwa silniki w roznych chwilach — pedy sie sumuja.
res = fly(jets(commander=ScheduleJetCommander([(1.0, "R1E1"), (1.5, "R1E2")])), t_max=3.0)
d1 = res.qr[i1] - r0.qr[i1]
dr = res.r[i1] - r0.r[i1]
ang = np.radians(45.0)
check("dwa impulsy: skladowa pitch sumuje sie",
      rel_err(d1, dq_exp * (1 + np.cos(ang))) < 2e-3, f"{d1:.5f}")
check("dwa impulsy: skladowa yaw z silnika pod 45 deg",
      rel_err(dr, -dq_exp * np.sin(ang)) < 2e-3, f"{dr:.5f}")

# --- 6. Opoznienie zaplonu ---------------------------------------------------
print("\n6. Opoznienie zaplonu")
delay = 0.2
ra = fly(jets(commander=ScheduleJetCommander([(1.0, "R1E1")])))
rb = fly(jets(delay=delay, commander=ScheduleJetCommander([(1.0, "R1E1")])))
check("opoznienie: zaplon przesuniety w logu",
      abs(rb.jet_events[0].t_ign - 1.2) < 1e-12 and abs(rb.jet_events[0].t_cmd - 1.0) < 1e-12)
k = at(rb, 1.15)
check("opoznienie: brak reakcji przed zaplonem", abs(rb.qr[k] - r0.qr[k]) < 1e-12)
k = at(rb, 2.0)
check("opoznienie: po zaplonie to samo dq", rel_err(rb.qr[k] - r0.qr[k], dq_exp) < 1e-3)

# --- 7. Komendy nieznane z gory (sterownik w tickach) -------------------------
print("\n7. Sterownik w petli — czasy zaplonu nieznane przed symulacja")


class FireWhenRolledCommander(JetCommander):
    """Przykladowy sterownik: odpala R1E1, gdy pitch rate spadnie ponizej
    progu (warunek zalezny od STANU, nie od czasu)."""
    rate_hz = 100.0

    def __init__(self, thr):
        self.thr = thr
        self.calls = []

    def reset(self):
        self.calls = []

    def decide(self, ctx):
        self.calls.append(ctx.t)
        if not ctx.is_fired("R1E1") and ctx.state.qr < self.thr:
            return ["R1E1"]
        return []


# W modelu prozniowym qr = 0 -> najpierw zrob impuls z harmonogramu "w dol"
# (R1E5, na -Z), a sterownik w petli odpowie impulsem przeciwnym.
class Combined(JetCommander):
    rate_hz = 100.0

    def __init__(self):
        self.a = ScheduleJetCommander([(1.003, "R1E5")])
        self.b = FireWhenRolledCommander(-0.5 * dq_exp)

    def planned_times(self):
        return self.a.planned_times()

    def reset(self):
        self.b.reset()

    def decide(self, ctx):
        return list(self.a.decide(ctx)) + list(self.b.decide(ctx))


cmb = Combined()
res = fly(jets(delay=0.002, commander=cmb))
fired = {e.engine: e for e in res.jet_events}
check("sterownik odpalil w odpowiedzi na stan", "R1E1" in fired)
e1 = fired.get("R1E1")
if e1 is not None:
    check("komenda w ticku 100 Hz (kwantyzacja do siatki)",
          abs(e1.t_cmd * 100 - round(e1.t_cmd * 100)) < 1e-6, f"t_cmd={e1.t_cmd}")
    # R1E5 pali sie 1.003..1.013 s; w ticku 1.01 s oddal 70% impulsu, wiec
    # qr juz przekroczylo prog -50% — sterownik reaguje jeszcze W TRAKCIE impulsu.
    check("pierwszy tick z qr ponizej progu (1.01 s, w trakcie impulsu R1E5)",
          abs(e1.t_cmd - 1.01) < 1e-9, f"t_cmd={e1.t_cmd}")
    check("zaplon = komenda + opoznienie", abs(e1.t_ign - e1.t_cmd - 0.002) < 1e-12)
k = at(res, 2.0)
check("impulsy przeciwne -> pitch rate wraca do zera", abs(res.qr[k] - r0.qr[k]) < 1e-3 * dq_exp,
      f"{res.qr[k] - r0.qr[k]:.2e}")
check("planowany tick 1.003 s mimo siatki 100 Hz", 1.003 in [round(c, 9) for c in cmb.b.calls])
check("ticki tylko monotonicznie", np.all(np.diff(cmb.b.calls) > 0))

# --- 8. Zgodnosc wsteczna ---------------------------------------------------
print("\n8. Zgodnosc wsteczna")
res = fly(jets(commander=NoJetCommander(), rate=100.0))
check("system bez komend == brak sterowania",
      np.allclose(res.qr, r0.qr, atol=1e-10) and np.allclose(res.z, r0.z, atol=1e-6))
check("ta sama siatka czasu wyjscia", res.t.shape == r0.t.shape and np.allclose(res.t, r0.t))
check("brak jetow -> pusty log", r0.jet_events == [])

# --- 9. YAML ---------------------------------------------------------------
print("\n9. Konfiguracja YAML")
cfg = load_config(ROOT / "configurations" / "rocket_70mm_reaction_jets.yaml")
check("control_type: reaction_jets", cfg.control_type == "reaction_jets")
sysj = build_reaction_jet_system(cfg)
check("z YAML: 2 rzedy x 8", sysj.n_engines == 16)
check("z YAML: promien domyslny = promien korpusu", abs(sysj.layout.radius - 0.035) < 1e-12)
check("z YAML: impuls silnika 1 N*s", abs(sysj.profile.impulse - 1.0) < 1e-12)
check("z YAML: control_rate_hz", sysj.tick_rate_hz == 100.0)
cfg_c = load_config(ROOT / "configurations" / "rocket_70mm_canards.yaml")
check("canardy: control_type: canards", cfg_c.control_type == "canards")
cfg_b = load_config(ROOT / "configurations" / "rocket_70mm_baseline.yaml")
check("bez sterowania: control_type wywnioskowany jako none", cfg_b.control_type == "none")

import tempfile
import yaml as _yaml
base = _yaml.safe_load(open(ROOT / "configurations" / "rocket_70mm_reaction_jets.yaml",
                            encoding="utf-8"))


def load_variant(mod):
    d = _yaml.safe_load(_yaml.safe_dump(base))
    mod(d)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        _yaml.safe_dump(d, fh)
        p = fh.name
    try:
        return load_config(p)
    finally:
        Path(p).unlink()


c = load_variant(lambda d: d.pop("control_type"))
check("bez klucza: wywnioskowane reaction_jets", c.control_type == "reaction_jets")
c = load_variant(lambda d: d["reaction_jets"]["engine"].update(
    profile=[[0, 0], [0.5, 1], [1, 0]], ignition_delay=0.003, tilt_axial_deg=5.0))
check("profil tabelaryczny i opcje silnika z YAML",
      c.reaction_jets.engine.profile == [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]
      and c.reaction_jets.engine.ignition_delay == 0.003)
for mod, why in [(lambda d: d.update(control_type="rcs"), "nieznany control_type"),
                 (lambda d: d.pop("reaction_jets"), "brak sekcji reaction_jets"),
                 (lambda d: d["reaction_jets"]["engine"].pop("thrust"), "brak thrust"),
                 (lambda d: d["reaction_jets"].update(rows=[]), "puste rows")]:
    try:
        load_variant(mod)
        check(f"YAML: odrzuca ({why})", False)
    except ValueError:
        check(f"YAML: odrzuca ({why})", True)

# --- 10. Sterownik wg kata przechylenia (RollAwareJetCommander) -------------
print("\n10. RollAwareJetCommander — kierunek w ukladzie startowym przy wirowaniu")
import dataclasses
from control import JetPulseRequest, RollAwareJetCommander


def fly_spin(control, p0_deg_s, t_max=2.0):
    st = dataclasses.replace(State6DOF.initial(elevation_deg=80.0, speed_0=200.0),
                             p=np.radians(p0_deg_s))
    return run_simulation_6dof(build_model(control), st, t_max=t_max,
                               dt_output=0.01, z_ground=-1e9)


def dv_angle_from(res, r0, target):
    """Kat [deg] miedzy zmiana predkosci (LF) a kierunkiem zadanym."""
    i = len(res.t) - 1
    dv = v_launch(res, i) - v_launch(r0, i)
    c = float(np.dot(dv, target) / np.linalg.norm(dv))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))), float(np.linalg.norm(dv))


rows_cg = [JetRow(XCG, 8, 0.0), JetRow(XCG, 8, 22.5)]     # w SC: czysty dv, bez obrotu
for p0 in (0.0, 2000.0, -2000.0):
    r0s = fly_spin(None, p0)
    cmd = RollAwareJetCommander([JetPulseRequest(1.0, "right")])
    rs = fly_spin(jets(rows=rows_cg, delay=0.005, commander=cmd), p0)
    ang, dv = dv_angle_from(rs, r0s, np.array([0.0, 1.0, 0.0]))
    sel = cmd.decisions[0][2]
    check(f"p={p0:+.0f} deg/s: impuls w +Y_LF (blad {ang:.1f} deg <= blad wyboru + 1 deg)",
          ang <= sel + 1.0, f"(wybor {sel:.1f} deg)")
    check(f"p={p0:+.0f} deg/s: blad wyboru <= tolerancja", sel <= 15.0)
    if p0 != 0.0:
        cmd_nl = RollAwareJetCommander([JetPulseRequest(1.0, "right")], lead_compensation=False)
        rn = fly_spin(jets(rows=rows_cg, delay=0.005, commander=cmd_nl), p0)
        ang_nl, _ = dv_angle_from(rn, r0s, np.array([0.0, 1.0, 0.0]))
        check(f"p={p0:+.0f} deg/s: kompensacja obrotu poprawia kierunek ({ang:.1f} < {ang_nl:.1f} deg)",
              ang < ang_nl - 5.0)

# Kilka impulsow w odstepach, kazdy innym silnikiem, wszystkie w tym samym kierunku.
r0s = fly_spin(None, 2000.0, t_max=3.0)
cmd = RollAwareJetCommander([JetPulseRequest(1.0, "left", n_pulses=4, min_interval_s=0.2)])
rs = fly_spin(jets(rows=rows_cg, delay=0.005, commander=cmd), 2000.0, t_max=3.0)
fired = [e for e in rs.jet_events if e.status == "fired"]
ang, dv = dv_angle_from(rs, r0s, np.array([0.0, -1.0, 0.0]))
check("4 impulsy: 4 rozne silniki", len({e.engine for e in fired}) == 4)
check("4 impulsy: odstep >= min_interval",
      np.all(np.diff([e.t_cmd for e in fired]) >= 0.2 - 1e-9))
check("4 impulsy: dv ~ 4 * I/m (kierunki zgodne)", rel_err(dv, 4.0 / M_ROCKET) < 0.02,
      f"{dv:.4f} vs {4.0 / M_ROCKET:.4f}")
check("4 impulsy: wypadkowa w -Y_LF (< 6 deg)", ang < 6.0, f"({ang:.1f} deg)")
check("roll-aware: ticki tylko gdy zadanie aktywne",
      cmd.next_tick_after(2.9, 500.0) == np.inf)

# Wyczerpanie silnikow: 18 impulsow zadanych, 16 silnikow. Roll-aware nigdy
# nie wybiera zuzytego silnika, a niewykonalne zadanie jest zamykane i
# logowane — zamiast tykac 500 Hz do konca lotu.
cmd = RollAwareJetCommander([JetPulseRequest(1.0, "left", n_pulses=18, min_interval_s=0.02)])
rs = fly_spin(jets(rows=rows_cg, commander=cmd), 2000.0, t_max=3.0)
st_ = [e.status for e in rs.jet_events]
check("wyczerpanie: odpalono wszystkie 16 silnikow, kazdy raz",
      st_.count("fired") == 16 and len({e.engine for e in rs.jet_events}) == 16)
check("wyczerpanie: roll-aware nie komenderuje zuzytych silnikow", "rejected_spent" not in st_)
check("wyczerpanie: zadanie zamkniete z 2 niewykonanymi impulsami",
      len(cmd.unfulfilled) == 1 and cmd.unfulfilled[0][2] == 2, f"{cmd.unfulfilled}")
check("wyczerpanie: po zamknieciu brak dalszych tickow",
      cmd.next_tick_after(cmd.unfulfilled[0][0], 500.0) == np.inf)

# Brak wirowania: zaden silnik nie trafia w tolerancje -> odpal najlepszy po max_wait.
cmd = RollAwareJetCommander([JetPulseRequest(1.0, (0.0, 1.0, 0.1))], tolerance_deg=0.1,
                            max_wait_s=0.05)
fly_spin(jets(rows=rows_cg, commander=cmd), 0.0)
check("bez wirowania: odpala najlepszy silnik po max_wait",
      len(cmd.decisions) == 1 and abs(cmd.decisions[0][0] - 1.05) < 1e-6,
      f"{cmd.decisions}")
for bad in ["sideways", (0.0, 0.0, 0.0)]:
    try:
        RollAwareJetCommander([JetPulseRequest(1.0, bad)])
        check(f"roll-aware: odrzuca kierunek {bad!r}", False)
    except ValueError:
        check(f"roll-aware: odrzuca kierunek {bad!r}", True)

# --------------------------------------------------------------------------
print("\n" + "=" * 66)
print(f"  Wynik: {PASS}/{PASS + FAIL} testow zaliczonych")
print(f"  STATUS: {'OK' if FAIL == 0 else 'BLAD'}")
print("=" * 66)
sys.exit(0 if FAIL == 0 else 1)
