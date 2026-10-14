"""
run_reaction_jets_matrix.py
===========================
Macierz testow sterowania REACTION JETS na realnej rakiecie 70mm
(configurations/rocket_70mm_reaction_jets.yaml). Cel: sprawdzic MODEL
STEROWANIA (komenda -> zaplon -> impuls -> odpowiedz), nie konfiguracje —
polozenia rzedow i silniki w YAML to wartosci przykladowe.

TEST 1 — lot syntetyczny: poziomo, stala predkosc i wysokosc, bez grawitacji
         i bez oporu, bez wirowania (moment od zaklinowania pletw wylaczony).
         Masa stala (stan "empty", jak w fazie lotu biernego).
  1a  "proznia"      : cala aerodynamika wylaczona -> czysta zasada pedu,
                       porownanie z wartosciami analitycznymi.
  1b  "aero bez oporu": CA = 0, sila normalna, Cm, tlumienie z DATCOM ->
                       realny mechanizm zakrzywiania toru (jet obraca nos,
                       kat natarcia daje sile normalna, ta skreca wektor V).
      Przypadki: kazdy z 16 silnikow osobno ("rozeta"), dla 1b dodatkowo
      liniowosc (1 / sqrt2 / 1+sqrt2 impulsu w tym samym kierunku) i
      warunki lotu (V = 100/200/300 m/s, h = 0/3000 m).

TEST 2 — lot rzeczywisty (missions/mission_01.yaml, ciag, szyna 2 m, pelna
         aero z wirowaniem od zaklinowania pletw), porownanie z lotem bez
         sterowania:
  S0  bez odpalen (odniesienie)
  S1  krzywa skutecznosci: 1 impuls "w lewo" (roll-aware) w roznych chwilach
  S2  ten sam czas, kolejno R1E1..R1E8 BEZ uwzglednienia przechylenia
  S3  roll-aware "w lewo": 1 / 4 / 8 impulsow co 0.2 s
  S4  dublet: "w lewo", po 0.3 s "w prawo"
  S5  dwa przeciwlegle silniki naraz (R1E1 + R1E5) -> brak efektu
  S6  wszystkie 16 silnikow roll-aware "w lewo", co 0.5 s

Wyniki: results/reaction_jets_matrix/  (CSV + wykresy PNG)

Uzycie:
    python run_reaction_jets_matrix.py                 # wszystko
    python run_reaction_jets_matrix.py --test 1a 1b    # tylko test 1
    python run_reaction_jets_matrix.py --test 2 --quick
    python run_reaction_jets_matrix.py --jobs 4        # liczba procesow
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CFG_PATH = ROOT / "configurations" / "rocket_70mm_reaction_jets.yaml"
BASELINE_CASE = "rocket_70mm_baseline"     # ten sam platowiec -> te same tablice DATCOM
MISSION = ROOT / "missions" / "mission_01.yaml"
OUT_DIR = ROOT / "results" / "reaction_jets_matrix"

# Test 1
T1_V, T1_H = 200.0, 1000.0        # domyslne warunki lotu [m/s], [m]
T1_T_FIRE, T1_T_END = 1.0, 4.0    # chwila impulsu i koniec przebiegu [s]
# Test 2
T2_T_REF = 3.0                    # chwila odniesienia dla S2..S5 [s]
ROLL_RATE_HZ = 500.0              # ticki sterownika roll-aware [Hz]


# ============================================================================
# Budowanie modelu
# ============================================================================

_CACHE: dict = {}


def load_cfg():
    if "cfg" not in _CACHE:
        from datcom_io.config_reader import load_config
        _CACHE["cfg"] = load_config(CFG_PATH)
    return _CACHE["cfg"]


def load_aero():
    """
    Aero DATCOM dla platowca z YAML. Konfiguracja jets ma ten sam korpus,
    pletwy i warunki lotu co baseline (canardow brak), wiec uzywamy tablic
    baseline — po sprawdzeniu, ze geometria jest identyczna.
    """
    if "aero" in _CACHE:
        return _CACHE["aero"]
    from aero import get_aero_model
    from datcom_io.config_reader import load_config
    cfg = load_cfg()
    base = load_config(ROOT / "configurations" / f"{BASELINE_CASE}.yaml")
    same = all(getattr(cfg, k) == getattr(base, k)
               for k in ("flight_conditions", "body", "fins", "mass"))
    case = BASELINE_CASE if same else cfg.name
    _CACHE["aero"] = get_aero_model(case, method="missile_datcom")
    _CACHE["aero_case"] = case
    return _CACHE["aero"]


def make_commander(spec):
    """Sterownik z opisu (tuple/dict — musi dac sie przeslac do procesu)."""
    from control import (JetPulseRequest, NoJetCommander, RollAwareJetCommander,
                         ScheduleJetCommander)
    if spec is None:
        return None
    kind = spec["kind"]
    if kind == "none":
        return NoJetCommander()
    if kind == "schedule":
        return ScheduleJetCommander(spec["schedule"])
    if kind == "roll_aware":
        reqs = [JetPulseRequest(t, d, n, dt) for (t, d, n, dt) in spec["requests"]]
        return RollAwareJetCommander(reqs, rate_hz=ROLL_RATE_HZ)
    raise ValueError(kind)


def make_jets(spec):
    from control import build_reaction_jet_system
    cmd = make_commander(spec)
    if cmd is None:
        return None
    return build_reaction_jet_system(load_cfg(), commander=cmd)


def synthetic_model(variant, control):
    """Test 1: poziomo, bez grawitacji i oporu, bez wirowania, masa stala."""
    from forces.force_model6 import ForceModel6DOF, PropulsionConfig6DOF
    from datcom_io.rocket_builder import build_geometry
    from models.aerodynamics import TableAero
    from models.atmosphere import create_atmosphere
    from models.gravity import create_gravity
    from models.launcher import LauncherConfig
    from models.mass6 import ConstantIxx, MassModel6DOF

    cfg = load_cfg()
    real = load_aero()
    if variant == "1a":
        z = np.zeros_like(real.CN_table)
        aero = TableAero(alpha_table=real.alpha_table, mach_table=real.mach_table,
                         CA_table=z, CN_table=z, Cm_table=z, Cmq=0.0,
                         xcg_ref=real.xcg_ref, lref_ref=real.lref_ref)
    else:
        aero = copy.deepcopy(real)
        aero.CA_table = np.zeros_like(aero.CA_table)    # bez oporu
        aero.CA_base_table = None
        aero.CLL_table = None                           # bez wirowania od zaklinowania
        aero.Cl0_manufacturing = 0.0
    e = cfg.mass_model.empty
    # MassModel6DOF wymaga m_empty < m_full — roznica 1 mg jest pomijalna.
    mass = MassModel6DOF(m_full=e.mass + 1e-6, m_empty=e.mass, t_burn=cfg.propulsion.t_burn,
                         xcg_full=e.xcg, xcg_empty=e.xcg,
                         Iyy_full=e.Iyy, Iyy_empty=e.Iyy, ixx_model=ConstantIxx(e.Ixx))
    geom = dataclasses.replace(build_geometry(cfg), cant_angle_rad=0.0)
    return ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"), mass_model=mass, aero_model=aero,
        gravity=create_gravity("none"), geometry=geom,
        propulsion=PropulsionConfig6DOF(thrust=0.0),
        launcher=LauncherConfig(L_rail=0.0), control=control)


def real_model(control):
    """Test 2: pelny model z YAML jak w MAIN_new.py."""
    from datcom_io.rocket_builder import (build_geometry, build_mass_model,
                                          build_propulsion)
    from forces.force_model6 import ForceModel6DOF
    from models.atmosphere import create_atmosphere
    from models.gravity import create_gravity
    from models.launcher import LauncherConfig
    cfg = load_cfg()
    return ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"), mass_model=build_mass_model(cfg),
        aero_model=load_aero(), gravity=create_gravity("constant"),
        geometry=build_geometry(cfg), propulsion=build_propulsion(cfg),
        launcher=LauncherConfig(L_rail=2.0), control=control)


# ============================================================================
# Pojedynczy przebieg (wykonywany w procesie roboczym)
# ============================================================================

def run_case(case):
    from core.solver6 import run_simulation_6dof
    from core.state6 import State6DOF
    t0 = time.time()
    jets = make_jets(case.get("commander"))
    if case["test"] in ("1a", "1b"):
        st = dataclasses.replace(
            State6DOF.initial(elevation_deg=0.0, speed_0=case.get("V", T1_V)),
            z=-case.get("h", T1_H))
        res = run_simulation_6dof(synthetic_model(case["test"], jets), st,
                                  t_max=T1_T_END, dt_output=0.005, z_ground=-1e9)
    else:
        from datcom_io.rocket_builder import build_initial_state
        from geo.geographic import MissionConfig
        st = build_initial_state(MissionConfig.from_yaml(str(MISSION)))
        res = run_simulation_6dof(real_model(jets), st, t_max=120.0,
                                  z_ground=-1.0, max_step=0.05)
    out = {k: getattr(res, k) for k in ("t", "x", "y", "z", "u", "v", "w",
                                         "p", "qr", "r", "q0", "q1", "q2", "q3",
                                         "speed", "alpha", "beta")}
    out["status"] = res.status
    out["events"] = [dataclasses.asdict(e) for e in res.jet_events]
    out["runtime"] = time.time() - t0
    out["name"] = case["name"]
    return out


def run_all(cases, jobs):
    if jobs <= 1:
        return [run_case(c) for c in cases]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(run_case, cases))


# ============================================================================
# Wielkosci wynikowe
# ============================================================================

def v_launch(r, i):
    from core.quaternion import quat_to_dcm
    q = np.array([r["q0"][i], r["q1"][i], r["q2"][i], r["q3"][i]])
    return quat_to_dcm(q).T @ np.array([r["u"][i], r["v"][i], r["w"][i]])


def fired(r):
    return [e for e in r["events"] if e["status"] == "fired"]


def alpha_total_deg(r):
    return np.degrees(np.arccos(np.clip(r["u"] / np.maximum(r["speed"], 1e-6), -1, 1)))


def after_first_fire(r, arr):
    ev = fired(r)
    if not ev:
        return arr
    return arr[r["t"] >= ev[0]["t_ign"]]


def metrics_t1(r, r0, case, mass, Iyy, xcg):
    i = len(r["t"]) - 1
    dv = v_launch(r, i) - v_launch(r0, i)
    V = case.get("V", T1_V)
    vl, vl0 = v_launch(r, i), v_launch(r0, i)
    head = np.degrees(np.arctan2(vl[1], vl[0]) - np.arctan2(vl0[1], vl0[0]))
    climb = np.degrees(np.arctan2(-vl[2], vl[0]) - np.arctan2(-vl0[2], vl0[0]))
    ev = fired(r)
    m = {
        "case": case["name"], "test": case["test"], "V": V, "h": case.get("h", T1_H),
        "engines": "+".join(e["engine"] for e in ev),
        "dy_m": float(r["y"][i] - r0["y"][i]),
        "dh_m": float(-(r["z"][i] - r0["z"][i])),
        "dheading_deg": float(head), "dclimb_deg": float(climb),
        "dv_ms": float(np.linalg.norm(dv)),
        "max_alpha_deg": float(np.max(np.abs(np.degrees(r["alpha"])))),
        "max_beta_deg": float(np.max(np.abs(np.degrees(r["beta"])))),
        "max_q_dps": float(np.max(np.abs(np.degrees(r["qr"])))),
        "max_r_dps": float(np.max(np.abs(np.degrees(r["r"])))),
        "status": r["status"], "runtime_s": round(r["runtime"], 1),
    }
    if case["test"] == "1a" and len(ev) == 1:
        cfg = load_cfg()
        I = cfg.reaction_jets.engine.thrust * cfg.reaction_jets.engine.burn_time
        row = int(ev[0]["engine"][1]) - 1
        arm = xcg - cfg.reaction_jets.rows[row].x
        m["dv_expected_ms"] = I / mass
        m["drate_expected_dps"] = float(np.degrees(I * arm / Iyy))
        m["drate_dps"] = float(np.degrees(np.hypot(r["qr"][i] - r0["qr"][i],
                                                   r["r"][i] - r0["r"][i])))
    return m


def metrics_t2(r, r0, case):
    i, i0 = len(r["t"]) - 1, len(r0["t"]) - 1
    ev = fired(r)
    a_tot = after_first_fire(r, alpha_total_deg(r))
    return {
        "case": case["name"], "group": case["group"],
        "n_fired": len(ev),
        "engines": "+".join(e["engine"] for e in ev),
        "t_first_ign": round(ev[0]["t_ign"], 4) if ev else "",
        "impact_x_m": float(r["x"][i]), "impact_y_m": float(r["y"][i]),
        "d_downrange_m": float(r["x"][i] - r0["x"][i0]),
        "d_crossrange_m": float(r["y"][i] - r0["y"][i0]),
        "apogee_m": float(-np.min(r["z"])),
        "d_apogee_m": float(-np.min(r["z"]) + np.min(r0["z"])),
        "flight_time_s": float(r["t"][i]),
        "max_alpha_total_after_fire_deg": float(np.max(a_tot)) if a_tot.size else 0.0,
        "alpha_gt_10deg": bool(a_tot.size and np.max(a_tot) > 10.0),
        "max_q_dps": float(np.max(np.abs(np.degrees(after_first_fire(r, r["qr"]))))),
        "max_r_dps": float(np.max(np.abs(np.degrees(after_first_fire(r, r["r"]))))),
        "status": r["status"], "runtime_s": round(r["runtime"], 1),
    }


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"  -> {path.relative_to(ROOT)}")


def savefig(fig, path):
    if path.exists():
        path.unlink()              # Windows moze trzymac lock na starym pliku
    fig.savefig(path, dpi=120)
    print(f"  -> {path.relative_to(ROOT)}")


# ============================================================================
# TEST 1
# ============================================================================

def test1(variant, jobs, quick):
    import matplotlib.pyplot as plt
    cfg = load_cfg()
    names = [f"R{i + 1}E{k + 1}" for i, row in enumerate(cfg.reaction_jets.rows)
             for k in range(row.n_engines)]
    sched = lambda engines: {"kind": "schedule", "schedule": [(T1_T_FIRE, engines)]}

    cases = [{"name": "baseline", "test": variant, "commander": {"kind": "none"}}]
    cases += [{"name": n, "test": variant, "commander": sched(n)} for n in names]
    if variant == "1b":
        # Liniowosc: wypadkowa w kierunku R1E1 o module 1, sqrt2, 1+sqrt2 impulsu.
        cases += [{"name": "lin_R1E2+R1E8", "test": variant, "commander": sched(["R1E2", "R1E8"])},
                  {"name": "lin_R1E1+R1E2+R1E8", "test": variant,
                   "commander": sched(["R1E1", "R1E2", "R1E8"])}]
        conds = [(100.0, 0.0), (300.0, 0.0), (100.0, 3000.0), (200.0, 3000.0), (300.0, 3000.0),
                 (200.0, 0.0)]
        if quick:
            conds = conds[:2]
        for V, h in conds:
            for nm, com in (("baseline", {"kind": "none"}), ("R1E1", sched("R1E1"))):
                cases.append({"name": f"cond_V{V:.0f}_h{h:.0f}_{nm}", "test": variant,
                              "commander": com, "V": V, "h": h})

    print(f"\nTEST {variant}: {len(cases)} przebiegow ...")
    res = {r["name"]: r for r in run_all(cases, jobs)}
    e = cfg.mass_model.empty
    rows = []
    for c in cases:
        if c["name"].endswith("baseline") or c["name"] == "baseline":
            continue
        base = ("baseline" if not c["name"].startswith("cond_")
                else c["name"].rsplit("_", 1)[0] + "_baseline")
        rows.append(metrics_t1(res[c["name"]], res[base], c, e.mass, e.Iyy, e.xcg))
    write_csv(rows, OUT_DIR / f"test{variant}.csv")

    # Kontrola 1a wzgledem analityki
    if variant == "1a":
        bad = [r for r in rows if "dv_expected_ms" in r and
               (abs(r["dv_ms"] / r["dv_expected_ms"] - 1) > 0.01 or
                abs(r["drate_dps"] / r["drate_expected_dps"] - 1) > 0.01)]
        print(f"  1a vs analityka (dv = I/m, dq = I*arm/Iyy, tolerancja 1%): "
              f"{'OK' if not bad else 'ROZBIEZNOSC: ' + ', '.join(r['case'] for r in bad)}")
    if variant == "1b":
        lin = {r["case"]: r for r in rows}
        h1 = lin["R1E1"]["dh_m"]
        for nm, k in (("lin_R1E2+R1E8", np.sqrt(2)), ("lin_R1E1+R1E2+R1E8", 1 + np.sqrt(2))):
            print(f"  liniowosc {nm}: dh = {lin[nm]['dh_m'] / h1:.3f} x R1E1 "
                  f"(oczekiwane {k:.3f})")

    # Wykresy: rozeta (dy, dh) + przebiegi czasowe
    single = [r for r in rows if r["case"] in names]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Test {variant}: {'proznia' if variant == '1a' else 'aero bez oporu'} — "
                 f"V={T1_V:.0f} m/s, h={T1_H:.0f} m, impuls w t={T1_T_FIRE:.1f} s",
                 fontweight="bold")
    for r in single:
        col = "tab:blue" if r["case"].startswith("R1") else "tab:orange"
        ax[0].plot(r["dy_m"], r["dh_m"], "o", color=col)
        ax[0].annotate(r["case"], (r["dy_m"], r["dh_m"]), fontsize=8,
                       xytext=(4, 4), textcoords="offset points")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].set_xlabel("dy (w prawo) [m]"); ax[0].set_ylabel("dh (w gore) [m]")
    ax[0].set_title(f"Odchylenie toru po {T1_T_END - T1_T_FIRE:.0f} s "
                    f"(niebieskie R1, pomaranczowe R2)")
    ax[0].grid(alpha=0.3)
    b = res["baseline"]
    for nm in ("R1E1", "R1E3", "R2E1"):
        r = res[nm]
        ax[1].plot(r["t"], -(r["z"] - b["z"]), label=f"{nm} dh")
        ax[1].plot(r["t"], r["y"] - b["y"], ls="--", label=f"{nm} dy")
    ax[1].axvline(T1_T_FIRE, color="grey", lw=0.6, ls=":")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("odchylenie [m]")
    ax[1].set_title("Przebieg odchylenia w czasie"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, OUT_DIR / f"test{variant}_rosette.png")
    plt.close(fig)

    if variant == "1b":
        cond = [r for r in rows if r["case"].startswith("cond_")]
        if cond:
            fig, ax = plt.subplots(figsize=(8, 5))
            for h in sorted({r["h"] for r in cond}):
                rr = sorted([r for r in cond if r["h"] == h], key=lambda r: r["V"])
                ax.plot([r["V"] for r in rr], [r["dclimb_deg"] for r in rr], "o-",
                        label=f"h = {h:.0f} m")
            ax.set_xlabel("V [m/s]"); ax.set_ylabel("zmiana kata toru [deg]")
            ax.set_title("Test 1b: skutecznosc 1 impulsu (R1E1) vs warunki lotu")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            savefig(fig, OUT_DIR / "test1b_conditions.png")
            plt.close(fig)
    return rows


# ============================================================================
# TEST 2
# ============================================================================

def test2(jobs, quick):
    import matplotlib.pyplot as plt
    print("\nTEST 2: lot odniesienia (S0) ...")
    r0 = run_case({"name": "S0", "test": "2", "commander": None})
    t, dist = r0["t"], np.hypot(r0["x"], r0["z"])
    t_rail = float(t[np.argmax(dist >= 2.0)])
    t_apo = float(t[np.argmin(r0["z"])])
    print(f"  zejscie z szyny t={t_rail:.2f} s, apogeum t={t_apo:.1f} s "
          f"({-r0['z'].min():.0f} m), upadek t={t[-1]:.1f} s, "
          f"max p={np.degrees(np.abs(r0['p']).max()):.0f} deg/s")

    ra = lambda reqs: {"kind": "roll_aware", "requests": reqs}
    sc = lambda sched: {"kind": "schedule", "schedule": sched}
    cases = []
    s1_times = [t_rail + 0.1, 0.8, 1.8, 3.0, 6.0, 10.0, t_apo - 2.0]
    if quick:
        s1_times = [1.8, 6.0]
    for tf in s1_times:
        cases.append({"name": f"S1_left_t{tf:.2f}", "group": "S1", "t_fire": tf,
                      "commander": ra([(tf, "left", 1, 0.0)])})
    for k in (range(1, 9) if not quick else (1, 3, 5)):
        cases.append({"name": f"S2_R1E{k}", "group": "S2",
                      "commander": sc([(T2_T_REF, f"R1E{k}")])})
    for n in ((1, 4, 8) if not quick else (4,)):
        cases.append({"name": f"S3_left_x{n}", "group": "S3",
                      "commander": ra([(T2_T_REF, "left", n, 0.2)])})
    cases.append({"name": "S4_doublet", "group": "S4",
                  "commander": ra([(T2_T_REF, "left", 1, 0.0), (T2_T_REF + 0.3, "right", 1, 0.0)])})
    cases.append({"name": "S5_R1E1+R1E5", "group": "S5",
                  "commander": sc([(T2_T_REF, ["R1E1", "R1E5"])])})
    cases.append({"name": "S6_all16_left", "group": "S6",
                  "commander": ra([(2.5, "left", 16, 0.5)])})
    for c in cases:
        c["test"] = "2"

    print(f"  {len(cases)} przebiegow ...")
    res = {r["name"]: r for r in run_all(cases, jobs)}
    rows = [metrics_t2(res[c["name"]], r0, c) for c in cases]
    for c, row in zip(cases, rows):
        if "t_fire" in c:
            row["t_fire_cmd"] = c["t_fire"]
    write_csv(rows, OUT_DIR / "test2.csv")

    print(f"\n  {'przypadek':22s} {'odpal.':>6s} {'d_cross':>9s} {'d_down':>9s} "
          f"{'d_apog':>8s} {'max a_tot':>9s}  status")
    for row in rows:
        flag = " (alpha>10!)" if row["alpha_gt_10deg"] else ""
        print(f"  {row['case']:22s} {row['n_fired']:6d} {row['d_crossrange_m']:9.2f} "
              f"{row['d_downrange_m']:9.2f} {row['d_apogee_m']:8.2f} "
              f"{row['max_alpha_total_after_fire_deg']:9.2f}  {row['status']}{flag}")

    # Wykresy
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Test 2: lot rzeczywisty 70mm (mission_01) — reaction jets vs lot bez sterowania",
                 fontweight="bold")
    groups = {"S3": "tab:blue", "S4": "tab:orange", "S5": "tab:grey", "S6": "tab:red"}
    a = ax[0, 0]
    for c in cases:
        if c["group"] in groups:
            r = res[c["name"]]
            a.plot(r["t"], r["y"] - np.interp(r["t"], r0["t"], r0["y"]),
                   color=groups[c["group"]], alpha=0.8, label=c["name"])
    a.set_xlabel("t [s]"); a.set_ylabel("odchylenie boczne vs S0 [m]")
    a.set_title("S3-S6: odchylenie boczne w czasie"); a.legend(fontsize=7); a.grid(alpha=0.3)

    a = ax[0, 1]
    s1 = [row for row in rows if row["group"] == "S1"]
    a.plot([row["t_fire_cmd"] for row in s1], [row["d_crossrange_m"] for row in s1], "o-")
    a.axvline(t_rail, color="grey", ls=":", lw=0.8)
    a.axvline(load_cfg().propulsion.t_burn, color="grey", ls="--", lw=0.8)
    a.set_xlabel("chwila impulsu [s]  (kropki: szyna, kreski: koniec pracy silnika)")
    a.set_ylabel("przesuniecie punktu upadku w bok [m]")
    a.set_title("S1: skutecznosc 1 impulsu 'w lewo' vs chwila odpalenia")
    a.grid(alpha=0.3)

    a = ax[1, 0]
    for row in rows:
        if row["group"] == "S2":
            a.plot(row["d_crossrange_m"], row["d_downrange_m"], "o", color="tab:purple")
            a.annotate(row["case"][3:], (row["d_crossrange_m"], row["d_downrange_m"]),
                       fontsize=8, xytext=(3, 3), textcoords="offset points")
    s3 = [row for row in rows if row["group"] == "S3"]
    a.plot([row["d_crossrange_m"] for row in s3], [row["d_downrange_m"] for row in s3],
           "s", color="tab:blue", label="S3 roll-aware")
    a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
    a.set_xlabel("d cross-range [m]"); a.set_ylabel("d down-range [m]")
    a.set_title(f"S2: staly silnik w t={T2_T_REF:.0f} s (bez uwzgl. przechylenia)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(r0["y"], r0["x"] / 1000.0, "k", lw=1.5, label="S0")
    for c in cases:
        if c["group"] in ("S3", "S6"):
            r = res[c["name"]]
            a.plot(r["y"], r["x"] / 1000.0, color=groups[c["group"]], alpha=0.8, label=c["name"])
    a.set_xlabel("y (w prawo) [m]"); a.set_ylabel("x (wzdluz azymutu) [km]")
    a.set_title("Slad na ziemi"); a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, OUT_DIR / "test2_summary.png")
    plt.close(fig)
    return rows


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--test", nargs="+", default=["1a", "1b", "2"], choices=["1a", "1b", "2"])
    ap.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--quick", action="store_true", help="mniej przypadkow (sprawdzenie)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    load_aero()
    eng = cfg.reaction_jets.engine
    print("=" * 74)
    print(f"MACIERZ TESTOW REACTION JETS — {cfg.name} (aero: {_CACHE['aero_case']})")
    print(f"  {sum(r.n_engines for r in cfg.reaction_jets.rows)} silnikow, "
          f"{eng.thrust:.0f} N x {eng.burn_time * 1e3:.1f} ms, "
          f"rzedy x = {[r.x for r in cfg.reaction_jets.rows]} m (wartosci przykladowe)")
    print(f"  procesy: {args.jobs}{'  (--quick)' if args.quick else ''}")
    print("=" * 74)
    t0 = time.time()
    for v in ("1a", "1b"):
        if v in args.test:
            test1(v, args.jobs, args.quick)
    if "2" in args.test:
        test2(args.jobs, args.quick)
    print(f"\nGotowe w {time.time() - t0:.0f} s. Wyniki: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
