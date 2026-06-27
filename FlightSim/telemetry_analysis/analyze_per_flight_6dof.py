"""
analyze_per_flight_6dof.py
===========================
Per-flight deterministyczne porownanie 6DOF vs dane polowe, dla KAZDEGO
lotu z 'to analyze? == yes' w configs.txt (nie tylko 3 skorygowanych) —
zamiast usrednien per typ nosa (ktore myla loty o roznej rzeczywistej
masie w jedna grupe), dajemy jedna czysta tabele: jeden wiersz na lot.

Dla kazdego lotu izolowane sa NARAZ trzy zmienne wczesniej trzymane na
wartosciach nominalnych ze wzgledu na uproszczenie (cant_angle z YAML,
elewacja/azymut z mission_01.yaml, atmosfera ISA standard):

  1. Masa: mass_model.full/empty z YAML skalowane proporcjonalnie tak,
     by m_full_skalowane == configs.txt 'm_rocket' tego lotu (xcg/Iyy
     trzymane jako te same FRAKCJE wzgledem masy co w YAML — nie mamy
     osobnych pomiarow xcg/Iyy per lot).
  2. Elewacja/azymut: z configs.txt tego lotu (nie z mission_01.yaml).
  3. Atmosfera: ISALaunchSiteAtmosphere zakotwiczona w T/p/Rh tego lotu
     (models/atmosphere.py), zamiast ISA standard.

Cant_angle: z configs.txt tego lotu. Jesli != cant_angle juz w cache
DATCOM (z YAML), wymagany jest przebieg DATCOM dla TEGO cant_angle —
robione przez tymczasowy YAML (wzorzec z analyze_cant_sweep.py /
run_6dof_cant_montecarlo.py), usuwany po uzyciu. MAIN.py i
configurations/rocket_70mm_baseline.yaml NIE sa modyfikowane.

WAZNE: loty o cant_angle != YAML wymagaja PRAWDZIWEGO DATCOM — ten
skrypt musi byc uruchomiony LOKALNIE (Windows + MissileDATCOM.exe) dla
pelnych wynikow. W kontenerze mozna jedynie sprawdzic logike skryptu z
--no-rerun-datcom (uzywa wylacznie cache bazowego YAML — wyniki
fizycznie poprawne tylko dla lotow, ktorych cant_angle juz odpowiada
YAML).

Uzycie (lokalnie, z prawdziwym DATCOM):
    python analyze_per_flight_6dof.py
    python analyze_per_flight_6dof.py --case-ostra rocket_70mm_baseline --case-tepa rocket_70mm_baseline_tepa

Sanity-check w kontenerze (bez DATCOM, tylko logika kodu):
    python analyze_per_flight_6dof.py --no-rerun-datcom
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir
from run_6dof_cant_montecarlo import get_aero_for_cant
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_propulsion, build_geometry
from models.mass6 import MassModel6DOF, LinearIxx
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF


# --------------------------------------------------------------------------
def read_flights(base):
    """Parsuje configs.txt -> lista dict per lot (tylko 'to analyze? == yes')."""
    cfg_path = Path(base) / "configs.txt"
    lines = open(cfg_path, encoding="utf-8", errors="replace").readlines()
    header = [h.strip() for h in lines[0].strip().split("\t")]
    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.strip().split("\t")]
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        try:
            fno = int(row["flight no"])
        except (ValueError, KeyError):
            continue
        if row.get("to analyze?", "no").strip().lower() != "yes":
            continue

        def f(k, default=0.0):
            try:
                return float(row.get(k, default))
            except ValueError:
                return default

        head = row.get("head configuration", "").lower().strip()
        nose = "tepa" if ("t" in head and ("pa" in head or "epa" in head)) else "ostra"

        rows.append(dict(
            fno=fno, nose=nose,
            cant=f("cant angle"), azimuth=f("azimuth"), elevation=f("elevation"),
            m_rocket=f("m_rocket"),
            T_C=f("T [C]"), p_hpa=f("p [hpa]"), RH_pct=f("Rh [%]"),
        ))
    rows.sort(key=lambda r: r["fno"])
    return rows


def actual_apogee_vmax(base, fno):
    csv_path = Path(base) / "results" / "trajectory_closure_summary.csv"
    if not csv_path.exists():
        return None, None
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["flight_no"]) == fno:
                v = float(row["V_max_actual_mps"]) if row.get("V_max_actual_mps") else None
                return float(row["h_apo_actual_m"]), v
    return None, None


# --------------------------------------------------------------------------
def build_scaled_mass(cfg, t_ignition, m_rocket_flight):
    """Skaluje mass_model.full/empty proporcjonalnie do m_rocket_flight,
    zachowujac frakcje xcg/Iyy/Ixx wzgledem masy nominalnej z YAML (per-lot
    pomiary xcg/Iyy nie istnieja, wiec skalujemy jednorodnie cala krzywa
    masy, nie tylko punkt full)."""
    mm = cfg.mass_model
    t_b = cfg.propulsion.t_burn
    m_full_nom = mm.full.mass
    scale = m_rocket_flight / m_full_nom if m_full_nom > 0 else 1.0

    m_full_new = mm.full.mass * scale
    m_empty_new = mm.empty.mass * scale

    return MassModel6DOF(
        m_full=m_full_new, m_empty=m_empty_new, t_burn=t_b,
        xcg_full=mm.full.xcg, xcg_empty=mm.empty.xcg,
        Iyy_full=mm.full.Iyy * scale, Iyy_empty=mm.empty.Iyy * scale,
        ixx_model=LinearIxx(Ixx_full=mm.full.Ixx * scale, Ixx_empty=mm.empty.Ixx * scale, t_burn=t_b),
        t_ignition=t_ignition,
    )


def run_one(aero, geom, atm, gravity, launcher, mass, prop, initial_state):
    force_model = ForceModel6DOF(
        atmosphere=atm, mass_model=mass, aero_model=aero,
        gravity=gravity, geometry=geom, propulsion=prop, launcher=launcher,
    )
    result = run_simulation_6dof(force_model, initial_state, t_max=100, dt_output=0.02,
                                  rtol=1e-6, atol=1e-8, max_step=0.05)
    h = -result.z
    i_apo = int(np.argmax(h))
    return float(h[i_apo]), float(np.max(result.speed)), result.status


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Per-lot 6DOF (masa+elewacja/azymut+atmosfera skorygowane) vs dane polowe")
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline",
                         help="case YAML dla lotow z nosem ostrym")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa",
                         help="case YAML dla lotow z nosem tepym")
    parser.add_argument("--no-rerun-datcom", action="store_true",
                         help="uzyj wylacznie istniejacego cache bazowego YAML (bez DATCOM); "
                              "fizycznie poprawne tylko dla lotow, ktorych cant_angle == YAML")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent

    case_by_nose = {"ostra": args.case_ostra, "tepa": args.case_tepa}
    cfg_base_by_nose = {
        nose: load_config(str(root / "configurations" / f"{case}.yaml"))
        for nose, case in case_by_nose.items()
    }
    cant_yaml_by_nose = {
        nose: sorted(set(round(fin.cant_angle, 5) for fin in cfg.fins))
        for nose, cfg in cfg_base_by_nose.items()
    }
    t_ignition_by_nose = {nose: cfg.propulsion.t_ignition for nose, cfg in cfg_base_by_nose.items()}

    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)

    flights = read_flights(base)
    print(f"Loty do analizy: {[r['fno'] for r in flights]}")
    print(f"case per nos: {case_by_nose}")
    print(f"cant_angle w YAML: {cant_yaml_by_nose}  (loty o innym cant_angle wymagaja DATCOM)\n")

    print(f"{'lot':>4} {'nos':>6} {'cant':>5} {'m_kg':>6} {'el':>5} {'az':>5} "
          f"{'T[C]':>6} {'h_pred':>8} {'h_act':>8} {'dh[%]':>7} "
          f"{'V_pred':>7} {'V_act':>7} {'dV[%]':>7}")

    out_rows = []
    for r in flights:
        case = case_by_nose[r["nose"]]
        cfg_base = cfg_base_by_nose[r["nose"]]
        t_ignition = t_ignition_by_nose[r["nose"]]

        aero, _ = get_aero_for_cant(case, r["cant"], force_rerun=not args.no_rerun_datcom)
        geom = build_geometry(load_config(str(root / "configurations" / f"{case}.yaml")))
        import math
        geom.cant_angle_rad = math.radians(r["cant"])

        prop = build_propulsion(cfg_base)
        mass = build_scaled_mass(cfg_base, t_ignition, r["m_rocket"])

        atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])

        initial_state = State6DOF.initial(elevation_deg=r["elevation"], azimuth_deg=r["azimuth"])

        try:
            h_pred, v_pred, status = run_one(aero, geom, atm, gravity, launcher, mass, prop, initial_state)
        except Exception as e:
            h_pred, v_pred, status = float("nan"), float("nan"), f"error:{e}"

        h_act, v_act = actual_apogee_vmax(base, r["fno"])
        dh_pct = 100.0 * (h_pred - h_act) / h_act if h_act else float("nan")
        dv_pct = 100.0 * (v_pred - v_act) / v_act if (v_act is not None) else float("nan")

        print(f"{r['fno']:4d} {r['nose']:>6} {r['cant']:5.2f} {r['m_rocket']:6.2f} "
              f"{r['elevation']:5.1f} {r['azimuth']:5.1f} {r['T_C']:6.1f} "
              f"{h_pred:8.1f} {h_act if h_act else float('nan'):8.1f} {dh_pct:+7.2f} "
              f"{v_pred:7.1f} {v_act if v_act else float('nan'):7.1f} {dv_pct:+7.2f}")

        out_rows.append(dict(
            fno=r["fno"], nose=r["nose"], cant=r["cant"], m_rocket=r["m_rocket"],
            elevation=r["elevation"], azimuth=r["azimuth"],
            T_C=r["T_C"], p_hpa=r["p_hpa"], RH_pct=r["RH_pct"],
            h_apo_pred=h_pred, v_max_pred=v_pred, status=status,
            h_apo_actual=h_act, v_max_actual=v_act,
            dh_pct=dh_pct, dv_pct=dv_pct,
        ))

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_flight_6dof_per_nose.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nZapisano: {csv_path}")

    valid = [r for r in out_rows if r["h_apo_actual"] and not np.isnan(r["dh_pct"])]
    if valid:
        dh_all = np.array([r["dh_pct"] for r in valid])
        print(f"\nBias apogeum (vs GPS/baro, per-lot masa+elewacja+atmosfera) na {len(valid)} lotach:")
        print(f"  dh_apo: mean={np.mean(dh_all):+.2f}%  std={np.std(dh_all):.2f}%  "
              f"[{np.min(dh_all):+.2f}%, {np.max(dh_all):+.2f}%]")

    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    fnos = [r["fno"] for r in out_rows]
    dh = [r["dh_pct"] for r in out_rows]
    colors = ["tab:blue" if r["nose"] == "ostra" else "tab:orange" for r in out_rows]
    ax.bar([str(f) for f in fnos], dh, color=colors, alpha=0.8)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Lot")
    ax.set_ylabel("Δ apogeum (pred vs actual) [%]")
    ax.set_title("Per-lot bias apogeum (masa+elewacja/azymut+atmosfera skorygowane, "
                 "baseline per ksztalt nosa)")
    ax.grid(alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="tab:blue", label=f"ostra ({args.case_ostra})"),
                        Patch(color="tab:orange", label=f"tepa ({args.case_tepa})")], fontsize=9)
    plt.tight_layout()
    out_png = out_dir / "per_flight_6dof_per_nose.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
