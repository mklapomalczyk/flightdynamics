"""
run_6dof_cant_montecarlo.py
=============================
Skrypt do uruchamiania LOKALNEGO (DATCOM dziala tylko na Windows, nie w
kontenerze) — dla kazdego cant_angle z listy:

  1. Generuje TYMCZASOWY plik configurations/<case>_cant<x>.yaml (kopia
     bazowego configu z podmienionym cant_angle na wszystkich zestawach
     pletw) — taki sam wzorzec jak juz uzywany w analyze_cant_sweep.py
     dla rakiety 36mm "malarakieta". Plik tymczasowy jest usuwany po
     przebiegu DATCOM; configurations/rocket_70mm_baseline.yaml (i
     MAIN.py) NIE sa modyfikowane.
  2. Uruchamia PRAWDZIWY MissileDATCOM.exe dla tego configu
     (aero.get_aero_model(..., method="missile_datcom", force_rerun=True))
     — wymaga Windows + datcom/MissileDATCOM.exe (patrz config.py).
     Wynik jest cache'owany w datcom_runs/<case>_cant<x>/aero_table_missile.pkl.
  3. Monte Carlo (domyslnie 250 przebiegow) PRAWDZIWEGO modelu 6DOF
     (core/solver6.run_simulation_6dof, ten sam pipeline co MAIN.py) z
     losowana zmiennoscia silnika — taki sam model jak w
     monte_carlo_thrust.py (jeden los. wspolczynnik skali amplitudy
     ciagu na przebieg + niezaleznie losowany czas spalania t_burn z
     burn_time_stats.csv), ale tutaj napedza PELNY model 6DOF (mase,
     bezwladnosci, geometrie, aero z DATCOM), NIE uproszczony model 2D.
  4. Apogeum (max wysokosci) i V_max kazdego przebiegu porownywane z
     polowymi danymi (field_test_data/results/trajectory_closure_summary.csv).
  5. Wynik: tabela w terminalu (latwa do analizy bez otwierania plikow)
     + CSV per-run + wykresy PNG (rozklad apogeum/V_max per cant_angle,
     na tle danych rzeczywistych).

WAZNE: do prawdziwego przebiegu DATCOM ten skrypt MUSI byc uruchomiony
LOKALNIE (Windows, gdzie dziala MissileDATCOM.exe). W kontenerze
(Linux, brak Wine) mozna jedynie sprawdzic logike skryptu z
--no-rerun-datcom (uzywa istniejacego cache, bez przeliczania DATCOM —
wynik bedzie POPRAWNY tylko dla cant_angle juz wbudowanego w cache, tj.
domyslnie tego z YAML).

Uzycie (lokalnie, z prawdziwym DATCOM):
    python run_6dof_cant_montecarlo.py
    python run_6dof_cant_montecarlo.py --cants 0.0 0.6 1.2 1.6 --n 250
    python run_6dof_cant_montecarlo.py --case rocket_70mm_baseline --nose ostra

Sanity-check w kontenerze (bez DATCOM, tylko logika kodu):
    python run_6dof_cant_montecarlo.py --no-rerun-datcom --cants 1.2 --n 5
"""

import sys
import csv
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir
from run_all_flights import read_configs, check_data_exists

from aero import get_aero_model
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import (
    build_mass_model, build_propulsion, build_geometry, build_initial_state,
    ThrustProfile, PropulsionConfig6DOFDynamic,
)
from models.mass6 import MassModel6DOF, LinearIxx
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from geo.geographic import MissionConfig


# --------------------------------------------------------------------------
# 1. Tymczasowy YAML per cant_angle (wzorzec z analyze_cant_sweep.py)
# --------------------------------------------------------------------------
def make_cant_case_name(case_name, cant_deg):
    return f"{case_name}_cant{cant_deg:.3f}".replace(".", "p")


def write_temp_cant_yaml(case_name, cant_deg):
    """Zapisuje configurations/<case>_cant<x>.yaml z podmienionym
    cant_angle na wszystkich zestawach pletw. Zwraca (path, case_name_cant).
    Caller jest odpowiedzialny za usuniecie pliku po uzyciu."""
    root = Path(__file__).resolve().parent.parent
    base_yaml_path = root / "configurations" / f"{case_name}.yaml"
    with open(base_yaml_path, encoding="utf-8", errors="replace") as f:
        raw = yaml.safe_load(f.read())
    if "fins" in raw:
        for fin in raw["fins"]:
            fin["cant_angle"] = round(float(cant_deg), 5)
    case_name_cant = make_cant_case_name(case_name, cant_deg)
    out_path = root / "configurations" / f"{case_name_cant}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, allow_unicode=True)
    return out_path, case_name_cant


def get_aero_for_cant(case_name, cant_deg, force_rerun):
    """Generuje tymczasowy YAML, odpala prawdziwy DATCOM (force_rerun=True,
    wymaga lokalnego MissileDATCOM.exe), usuwa tymczasowy YAML, zwraca
    (aero_model, case_name_cant).

    force_rerun=False: uzywa WYLACZNIE istniejacego cache bazowego case_name
    (bez generowania nowego case'u per cant) — tylko do sanity-checku
    skryptu bez DATCOM; fizycznie poprawne tylko gdy cant_deg odpowiada
    cant_angle juz wbudowanemu w ten cache (tj. wartosci z YAML)."""
    if not force_rerun:
        aero = get_aero_model(case_name, method="missile_datcom", force_rerun=False)
        return aero, case_name
    temp_path, case_name_cant = write_temp_cant_yaml(case_name, cant_deg)
    try:
        aero = get_aero_model(case_name_cant, method="missile_datcom", force_rerun=True)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
    return aero, case_name_cant


# --------------------------------------------------------------------------
# 2. Model zmiennosci silnika (jak w monte_carlo_thrust.py), ale tworzy
#    obiekty 6DOF (MassModel6DOF + PropulsionConfig6DOFDynamic)
# --------------------------------------------------------------------------
def load_thrust_mean_std(csv_path):
    frac, T_mean, T_std = [], [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frac.append(float(row["t_frac_burn"]))
            T_mean.append(float(row["T_mean_N"]))
            T_std.append(float(row["T_std_N"]))
    return np.array(frac), np.array(T_mean), np.array(T_std)


def load_burn_time_stats(csv_path):
    stats = {}
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0] in ("mean_s", "std_s", "min_s", "max_s", "n_flights"):
                stats[parts[0]] = parts[1]
    return dict(mean_s=float(stats["mean_s"]), std_s=float(stats["std_s"]),
                min_s=float(stats["min_s"]), max_s=float(stats["max_s"]))


def build_mc_mass_and_propulsion(cfg, t_ignition, t_burn_draw, scale):
    """Buduje MassModel6DOF i PropulsionConfig6DOFDynamic dla jednego
    przebiegu MC: ciag = scale * profil_nominalny, os czasu rozciagnieta
    tak, by spalanie konczylo sie po t_burn_draw (od t_ignition);
    masa/xcg/Iyy/Ixx interpoluja full->empty w tym samym, rozciagnietym
    oknie t_burn_draw (te same wartosci krancowe co w YAML)."""
    mm = cfg.mass_model
    nominal_profile = np.array(cfg.propulsion.thrust_profile, dtype=float)
    t_burn_nominal = cfg.propulsion.t_burn
    stretch = t_burn_draw / t_burn_nominal if t_burn_nominal > 0 else 1.0

    new_profile = nominal_profile.copy()
    new_profile[:, 0] = new_profile[:, 0] * stretch
    new_profile[:, 1] = new_profile[:, 1] * scale

    prop = PropulsionConfig6DOFDynamic(
        thrust_profile=ThrustProfile(new_profile.tolist()),
        t_ignition=t_ignition,
        nozzle_diameter=cfg.propulsion.nozzle_diameter,
    )

    mass = MassModel6DOF(
        m_full=mm.full.mass, m_empty=mm.empty.mass, t_burn=t_burn_draw,
        xcg_full=mm.full.xcg, xcg_empty=mm.empty.xcg,
        Iyy_full=mm.full.Iyy, Iyy_empty=mm.empty.Iyy,
        ixx_model=LinearIxx(Ixx_full=mm.full.Ixx, Ixx_empty=mm.empty.Ixx, t_burn=t_burn_draw),
        t_ignition=t_ignition,
    )
    return mass, prop


# --------------------------------------------------------------------------
# 3. Dane rzeczywiste (porownanie)
# --------------------------------------------------------------------------
def actual_stats(base, nose):
    rows = read_configs(base)
    flights = [r["fno"] for r in rows if r["nose"] == nose and check_data_exists(base, r["fno"])]
    csv_path = Path(base) / "results" / "trajectory_closure_summary.csv"
    h_vals, v_vals = [], []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["flight_no"]) in flights:
                    h_vals.append(float(row["h_apo_actual_m"]))
                    if row.get("V_max_actual_mps"):
                        v_vals.append(float(row["V_max_actual_mps"]))
    return dict(
        h_mean=float(np.mean(h_vals)) if h_vals else None,
        h_std=float(np.std(h_vals)) if h_vals else None,
        v_mean=float(np.mean(v_vals)) if v_vals else None,
        v_std=float(np.std(v_vals)) if v_vals else None,
        flights=flights,
    )


# --------------------------------------------------------------------------
# 4. Jeden przebieg 6DOF
# --------------------------------------------------------------------------
def run_one_6dof(aero, geom, atm, gravity, launcher, mass, prop, initial_state):
    force_model = ForceModel6DOF(
        atmosphere=atm, mass_model=mass, aero_model=aero,
        gravity=gravity, geometry=geom, propulsion=prop, launcher=launcher,
    )
    result = run_simulation_6dof(force_model, initial_state, t_max=100, dt_output=0.02,
                                  rtol=1e-5, atol=1e-7, max_step=0.1)
    h = -result.z
    i_apo = int(np.argmax(h))
    return float(h[i_apo]), float(np.max(result.speed)), result.status


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="6DOF Monte Carlo per cant_angle (prawdziwy DATCOM lokalnie) vs dane polowe")
    parser.add_argument("--case", default="rocket_70mm_baseline")
    parser.add_argument("--nose", default="ostra")
    parser.add_argument("--cants", nargs="*", type=float, default=[0.0, 0.6, 1.2, 1.6])
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rerun-datcom", action="store_true",
                         help="uzyj istniejacego cache DATCOM, NIE przeliczaj (tylko do "
                              "szybkiego sanity-checku skryptu w kontenerze bez DATCOM; "
                              "fizycznie poprawne tylko dla cant_angle juz w cache)")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent

    cfg_base = load_config(str(root / "configurations" / f"{args.case}.yaml"))
    mission = MissionConfig.from_yaml(str(root / "missions" / "mission_01.yaml"))
    t_ignition = cfg_base.propulsion.t_ignition

    burn = load_burn_time_stats(Path(base) / "results" / "burn_time_stats.csv")
    thrust_frac, thrust_mean, thrust_std = load_thrust_mean_std(Path(base) / "results" / "thrust_mean_std.csv")
    mid = (thrust_frac > 0.05) & (thrust_frac < 0.95)
    rel_std = float(np.mean(thrust_std[mid] / thrust_mean[mid]))

    act = actual_stats(base, args.nose)
    print(f"Konfiguracja: {args.case}  cants={args.cants}  n_MC={args.n}  "
          f"DATCOM rerun={'NIE (--no-rerun-datcom)' if args.no_rerun_datcom else 'TAK (prawdziwy)'}")
    print(f"Rozrzut wzgledny amplitudy ciagu: {rel_std:.3f}   "
          f"t_burn: mean={burn['mean_s']:.3f}s std={burn['std_s']:.3f}s")
    if act["h_mean"] is not None:
        print(f"Rzeczywiste apogeum GPS (nos '{args.nose}', loty {act['flights']}): "
              f"mean={act['h_mean']:.1f} m std={act['h_std']:.1f} m")
    if act["v_mean"] is not None:
        print(f"Rzeczywiste V_max (nos '{args.nose}'): mean={act['v_mean']:.1f} m/s std={act['v_std']:.1f} m/s")

    atm = create_atmosphere("ISA")
    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)
    initial_state = build_initial_state(mission)

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cant_csv = out_dir / f"cant_montecarlo_{args.case}_{args.nose}_runs.csv"
    summary_rows = []

    with open(per_cant_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cant_deg", "run", "h_apo_m", "v_max_mps", "status"])

        print(f"\n{'cant[deg]':>10} {'n_ok':>6} {'h_mean':>9} {'h_std':>8} "
              f"{'v_mean':>9} {'v_std':>8} {'bias_h[%]':>10} {'bias_v[%]':>10}")

        for cant in args.cants:
            print(f"  -> DATCOM dla cant={cant:.3f} deg ...", flush=True)
            aero, case_name_cant = get_aero_for_cant(args.case, cant, force_rerun=not args.no_rerun_datcom)
            geom = build_geometry(load_config(str(root / "configurations" / f"{args.case}.yaml")))
            # geom.cant_angle_rad uzywany tylko jako fallback roll-moment; sila osiowa
            # pochodzi z aero (CA_table) DATCOM przeliczonego dla TEGO cant_deg.
            import math
            geom.cant_angle_rad = math.radians(cant)

            rng = np.random.default_rng(args.seed + int(round(cant * 1000)))
            h_arr = np.full(args.n, np.nan)
            v_arr = np.full(args.n, np.nan)
            status_arr = []

            for i in range(args.n):
                scale = max(rng.normal(1.0, rel_std), 0.3)
                t_burn_draw = float(np.clip(rng.normal(burn["mean_s"], burn["std_s"]),
                                             burn["min_s"], burn["max_s"]))
                mass, prop = build_mc_mass_and_propulsion(cfg_base, t_ignition, t_burn_draw, scale)
                try:
                    h_apo, v_max, status = run_one_6dof(aero, geom, atm, gravity, launcher,
                                                         mass, prop, initial_state)
                except Exception as e:
                    h_apo, v_max, status = float("nan"), float("nan"), f"error:{e}"
                h_arr[i] = h_apo
                v_arr[i] = v_max
                status_arr.append(status)
                writer.writerow([cant, i, round(h_apo, 2) if not np.isnan(h_apo) else "",
                                  round(v_max, 2) if not np.isnan(v_max) else "", status])

            ok = ~np.isnan(h_arr)
            n_ok = int(np.sum(ok))
            h_mean = float(np.mean(h_arr[ok])) if n_ok else float("nan")
            h_std = float(np.std(h_arr[ok])) if n_ok else float("nan")
            v_mean = float(np.mean(v_arr[ok])) if n_ok else float("nan")
            v_std = float(np.std(v_arr[ok])) if n_ok else float("nan")
            bias_h = 100.0 * (h_mean - act["h_mean"]) / act["h_mean"] if act["h_mean"] else float("nan")
            bias_v = 100.0 * (v_mean - act["v_mean"]) / act["v_mean"] if act["v_mean"] else float("nan")

            print(f"{cant:10.2f} {n_ok:6d} {h_mean:9.1f} {h_std:8.1f} "
                  f"{v_mean:9.1f} {v_std:8.1f} {bias_h:+10.1f} {bias_v:+10.1f}")

            summary_rows.append(dict(cant_deg=cant, n_ok=n_ok, h_mean=h_mean, h_std=h_std,
                                      v_mean=v_mean, v_std=v_std, bias_h_pct=bias_h, bias_v_pct=bias_v,
                                      h_runs=h_arr[ok].copy(), v_runs=v_arr[ok].copy()))

    summary_csv = out_dir / f"cant_montecarlo_{args.case}_{args.nose}_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cant_deg", "n_ok", "h_mean", "h_std",
                                                 "v_mean", "v_std", "bias_h_pct", "bias_v_pct"])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"\nZapisano: {per_cant_csv}\nZapisano: {summary_csv}")

    # ------------------------------------------------------------------
    # Wykresy: rozklad (box) apogeum i V_max per cant, na tle danych rzeczywistych
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    box_h = [r["h_runs"] for r in summary_rows]
    box_v = [r["v_runs"] for r in summary_rows]
    labels = [f"{r['cant_deg']:.2f}" for r in summary_rows]

    try:
        ax1.boxplot(box_h, tick_labels=labels, showmeans=True)
    except TypeError:
        ax1.boxplot(box_h, labels=labels, showmeans=True)
    if act["h_mean"] is not None:
        ax1.axhline(act["h_mean"], color="tab:red", ls="--", lw=1.5,
                     label=f"GPS mean ({args.nose})")
        ax1.axhspan(act["h_mean"] - act["h_std"], act["h_mean"] + act["h_std"],
                     color="tab:red", alpha=0.1)
    ax1.set_xlabel("Cant angle [deg]")
    ax1.set_ylabel("Apogeum [m]")
    ax1.set_title(f"Apogeum MC (n={args.n}/cant) vs cant_angle — {args.case}")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    try:
        ax2.boxplot(box_v, tick_labels=labels, showmeans=True)
    except TypeError:
        ax2.boxplot(box_v, labels=labels, showmeans=True)
    if act["v_mean"] is not None:
        ax2.axhline(act["v_mean"], color="tab:red", ls="--", lw=1.5,
                     label=f"V_max actual mean ({args.nose})")
        ax2.axhspan(act["v_mean"] - act["v_std"], act["v_mean"] + act["v_std"],
                     color="tab:red", alpha=0.1)
    ax2.set_xlabel("Cant angle [deg]")
    ax2.set_ylabel("V_max [m/s]")
    ax2.set_title(f"V_max MC (n={args.n}/cant) vs cant_angle — {args.case}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_png = out_dir / f"cant_montecarlo_{args.case}_{args.nose}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
