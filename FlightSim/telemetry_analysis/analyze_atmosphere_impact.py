"""
analyze_atmosphere_impact.py
==============================
Sprawdza wplyw zmierzonych warunkow atmosferycznych w dniu startu
(T [C], p [hPa], Rh [%] — kolumny dodane do field_test_data/configs.txt)
na apogeum i V_max z PRAWDZIWEGO modelu 6DOF (ten sam pipeline co
MAIN.py), w porownaniu do standardowej atmosfery ISA (MSL).

Metoda — izolowana od innych zmiennych (silnik, masa per-lot, elewacja
per-lot NIE sa tu zmieniane — uzywane sa wartosci nominalne z YAML/misji,
tak jak w run_6dof_cant_montecarlo.py), aby pokazac WYLACZNIE efekt
atmosfery:
  Dla kazdego lotu z configs.txt (T/p/Rh zmierzone): jeden deterministyczny
  przebieg 6DOF (bez Monte Carlo, profil ciagu nominalny z YAML) z
  atmosfera ISA (standard) i drugi z ISA_LAUNCH (zakotwiczona w
  zmierzonych T0/p0/Rh tego lotu, models/atmosphere.py:
  ISALaunchSiteAtmosphere) — ten sam cant_angle (domyslnie 1.2 deg,
  zgodny z YAML, cache DATCOM NIE jest przeliczany).

  Różnica (apogee_launch - apogee_ISA) / apogee_ISA pokazuje, jak duzy
  jest efekt samej atmosfery — niezalezny od zmiennosci silnika juz
  sprawdzonej w monte_carlo_thrust.py / run_6dof_cant_montecarlo.py.

Nie modyfikuje MAIN.py ani configurations/*.yaml.

Uzycie:
    python analyze_atmosphere_impact.py
    python analyze_atmosphere_impact.py --case rocket_70mm_baseline --cant 1.2
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
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_mass_model, build_propulsion, build_geometry, build_initial_state
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from geo.geographic import MissionConfig
from aero import get_aero_model


# --------------------------------------------------------------------------
def read_flight_atmosphere(base):
    """Parsuje configs.txt — kolumny T [C], p [hpa], Rh [%] per lot.
    Zwraca liste dict: fno, nose, T_C, p_hpa, RH_pct, analyze."""
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

        def f(k, default=0.0):
            try:
                return float(row.get(k, default))
            except ValueError:
                return default

        head = row.get("head configuration", "").lower().strip()
        nose = "tepa" if ("t" in head and ("pa" in head or "epa" in head)) else "ostra"

        if "T [C]" not in row or "p [hpa]" not in row:
            continue
        rows.append(dict(
            fno=fno, nose=nose,
            T_C=f("T [C]"), p_hpa=f("p [hpa]"), RH_pct=f("Rh [%]"),
            analyze=row.get("to analyze?", "no").strip().lower() == "yes",
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


def run_one(atm, mass, prop, geom, gravity, launcher, aero, initial_state):
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
        description="Wplyw zmierzonej atmosfery (dzien startu) na 6DOF apogeum/V_max vs ISA standard")
    parser.add_argument("--case", default="rocket_70mm_baseline")
    parser.add_argument("--cant", type=float, default=1.2)
    parser.add_argument("--only-analyzed", action="store_true",
                         help="tylko loty z 'to analyze? == yes' w configs.txt")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent

    cfg = load_config(str(root / "configurations" / f"{args.case}.yaml"))
    cant_in_yaml = sorted(set(round(fin.cant_angle, 5) for fin in cfg.fins))
    if cant_in_yaml != [round(args.cant, 5)]:
        print(f"UWAGA: cant_angle w YAML={cant_in_yaml}, docelowy={args.cant} — "
              f"uzyte zostanie cache DATCOM JAK JEST (bez przeliczania); "
              f"wynik bedzie scisly tylko jesli te wartosci sie zgadzaja.")

    aero = get_aero_model(args.case, method="missile_datcom", force_rerun=False)
    mass = build_mass_model(cfg)
    prop = build_propulsion(cfg)
    geom = build_geometry(cfg)
    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)

    mission = MissionConfig.from_yaml(str(root / "missions" / "mission_01.yaml"))
    initial_state = build_initial_state(mission)

    isa_std = create_atmosphere("ISA")
    h_isa, v_isa, status_isa = run_one(isa_std, mass, prop, geom, gravity, launcher, aero, initial_state)
    print(f"Referencja: ISA standard (288.15 K, 101325 Pa) -> h_apo={h_isa:.1f} m  "
          f"V_max={v_isa:.1f} m/s  status={status_isa}")

    flights = read_flight_atmosphere(base)
    if args.only_analyzed:
        flights = [r for r in flights if r["analyze"]]

    print(f"\n{'lot':>4} {'nos':>6} {'T[C]':>6} {'p[hPa]':>7} {'Rh[%]':>6} "
          f"{'h_apo[m]':>9} {'dh[%]':>7} {'V_max':>7} {'dV[%]':>7} "
          f"{'GPS_h[m]':>9} {'GPS_V':>7}")

    out_rows = []
    for r in flights:
        atm_launch = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])
        h_l, v_l, status_l = run_one(atm_launch, mass, prop, geom, gravity, launcher, aero, initial_state)
        dh_pct = 100.0 * (h_l - h_isa) / h_isa
        dv_pct = 100.0 * (v_l - v_isa) / v_isa
        h_act, v_act = actual_apogee_vmax(base, r["fno"])

        print(f"{r['fno']:4d} {r['nose']:>6} {r['T_C']:6.1f} {r['p_hpa']:7.1f} {r['RH_pct']:6.1f} "
              f"{h_l:9.1f} {dh_pct:+7.2f} {v_l:7.1f} {dv_pct:+7.2f} "
              f"{h_act if h_act else float('nan'):9.1f} {v_act if v_act else float('nan'):7.1f}")

        out_rows.append(dict(fno=r["fno"], nose=r["nose"], T_C=r["T_C"], p_hpa=r["p_hpa"],
                              RH_pct=r["RH_pct"], h_apo_ISA=h_isa, v_max_ISA=v_isa,
                              h_apo_launch=h_l, v_max_launch=v_l,
                              dh_pct=dh_pct, dv_pct=dv_pct,
                              h_apo_actual=h_act, v_max_actual=v_act))

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"atmosphere_impact_{args.case}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nZapisano: {csv_path}")

    dh_all = np.array([r["dh_pct"] for r in out_rows])
    dv_all = np.array([r["dv_pct"] for r in out_rows])
    print(f"\nZakres efektu atmosfery (vs ISA standard) na {len(out_rows)} lotach:")
    print(f"  dh_apo: mean={np.mean(dh_all):+.2f}%  std={np.std(dh_all):.2f}%  "
          f"[{np.min(dh_all):+.2f}%, {np.max(dh_all):+.2f}%]")
    print(f"  dV_max: mean={np.mean(dv_all):+.2f}%  std={np.std(dv_all):.2f}%  "
          f"[{np.min(dv_all):+.2f}%, {np.max(dv_all):+.2f}%]")

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    fnos = [r["fno"] for r in out_rows]
    ax = axes[0]
    ax.bar([str(f) for f in fnos], dh_all, color="tab:blue", alpha=0.75)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Lot")
    ax.set_ylabel("Δ apogeum vs ISA standard [%]")
    ax.set_title("Efekt zmierzonej atmosfery na apogeum")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar([str(f) for f in fnos], dv_all, color="tab:green", alpha=0.75)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Lot")
    ax.set_ylabel("Δ V_max vs ISA standard [%]")
    ax.set_title("Efekt zmierzonej atmosfery na V_max")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Wplyw zmierzonej atmosfery (dzien startu) — {args.case}, cant={args.cant} deg")
    plt.tight_layout()
    out_png = out_dir / f"atmosphere_impact_{args.case}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
