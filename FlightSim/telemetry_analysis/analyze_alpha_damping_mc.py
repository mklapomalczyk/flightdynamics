"""
analyze_alpha_damping_mc.py
=============================
Sprawdza hipoteze "lift/CN_alpha jest przeszacowany -> nadmierny opor
indukowany od kata natarcia", uzywajac PRAWDZIWEGO modelu 6DOF (ten sam
pipeline co MAIN.py / run_6dof_cant_montecarlo.py) z Monte Carlo
zmiennosci silnika (jak w monte_carlo_thrust.py), ale TYLKO dla jednego
cant_angle (domyslnie 1.2 deg — sprawdzane wzgledem configurations/
rocket_70mm_baseline.yaml; jesli YAML ma inna wartosc, generowany jest
tymczasowy YAML z 1.2 deg, taki sam wzorzec co run_6dof_cant_montecarlo.py,
plik tymczasowy jest usuwany po przebiegu DATCOM).

W przeciwienstwie do run_6dof_cant_montecarlo.py, ten skrypt zapisuje
PELNA historie alpha(t)/beta(t) z kazdego przebiegu MC i liczy:
  - alpha_max, beta_max, total_max = max sqrt(alpha^2+beta^2)  [deg]
  - t_damp_2deg: czas po wypaleniu, po ktorym total < 2 deg na zawsze
    (NaN jesli nigdy nie zejdzie pod ten prog)
  - I_alpha2_coast: calka total^2 dt w fazie bezsilowej (do apogeum)
    [deg^2 * s] — proxy energii rozproszonej przez opor indukowany
  - h_apo, v_max (jak wczesniej)

Pozwala sprawdzic, czy rozrzut/wielkosc apogeum koreluje z historia
kata natarcia (czyli czy "zbyt duzy CN_alpha / zbyt slabe tlumienie"
moglyby tlumaczyc rozjazd z danymi polowymi), czy nie (a wiec szukac
przyczyny gdzie indziej).

WAZNE: do prawdziwego przebiegu DATCOM (force_rerun=True, domyslnie)
ten skrypt MUSI byc uruchomiony LOKALNIE (Windows, MissileDATCOM.exe).
W kontenerze tylko --no-rerun-datcom (cache) do sanity-checku logiki.

Uzycie:
    python analyze_alpha_damping_mc.py
    python analyze_alpha_damping_mc.py --cant 1.2 --n 250
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
from datcom_io.rocket_builder import build_geometry, build_initial_state
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from geo.geographic import MissionConfig

from run_6dof_cant_montecarlo import (
    get_aero_for_cant, load_thrust_mean_std, load_burn_time_stats,
    build_mc_mass_and_propulsion, actual_stats,
)


# --------------------------------------------------------------------------
def check_baseline_cant(case_name, root):
    """Wypisuje cant_angle z YAML i sprawdza, czy odpowiada wartosci docelowej."""
    cfg = load_config(str(root / "configurations" / f"{case_name}.yaml"))
    vals = sorted(set(round(fin.cant_angle, 5) for fin in cfg.fins))
    return cfg, vals


def run_one_with_history(aero, geom, atm, gravity, launcher, mass, prop, initial_state):
    force_model = ForceModel6DOF(
        atmosphere=atm, mass_model=mass, aero_model=aero,
        gravity=gravity, geometry=geom, propulsion=prop, launcher=launcher,
    )
    result = run_simulation_6dof(force_model, initial_state, t_max=100, dt_output=0.02,
                                  rtol=1e-5, atol=1e-7, max_step=0.1)
    return result


def analyze_history(result, t_burnout):
    t = result.t
    alpha_deg = np.degrees(result.alpha)
    beta_deg = np.degrees(result.beta)
    total_deg = np.sqrt(alpha_deg**2 + beta_deg**2)

    h = -result.z
    i_apo = int(np.argmax(h))
    h_apo = float(h[i_apo])
    v_max = float(np.max(result.speed))

    alpha_max = float(np.max(np.abs(alpha_deg)))
    beta_max = float(np.max(np.abs(beta_deg)))
    total_max = float(np.max(total_deg))

    # Calka total^2 dt w fazie bezsilowej (od burnout do apogeum)
    coast_mask = (t >= t_burnout) & (t <= t[i_apo])
    if np.sum(coast_mask) > 1:
        I_alpha2_coast = float(np.trapezoid(total_deg[coast_mask] ** 2, t[coast_mask]))
    else:
        I_alpha2_coast = 0.0

    # Czas po wypaleniu, po ktorym total < 2 deg NA ZAWSZE (do apogeum)
    after_burnout = t >= t_burnout
    below = total_deg < 2.0
    t_damp = float("nan")
    idxs = np.where(after_burnout)[0]
    for k in idxs:
        if np.all(below[k:]):
            t_damp = float(t[k] - t_burnout)
            break

    return dict(h_apo=h_apo, v_max=v_max, alpha_max_deg=alpha_max, beta_max_deg=beta_max,
                total_max_deg=total_max, I_alpha2_coast=I_alpha2_coast,
                t_damp_2deg_s=t_damp, status=result.status)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="6DOF MC (1 cant_angle): koreluje historie alpha/beta z apogeum/Vmax")
    parser.add_argument("--case", default="rocket_70mm_baseline")
    parser.add_argument("--nose", default="ostra")
    parser.add_argument("--cant", type=float, default=1.2)
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rerun-datcom", action="store_true",
                         help="uzyj cache DATCOM bez przeliczania (sanity-check w kontenerze)")
    parser.add_argument("--n-traces", type=int, default=8,
                         help="liczba przykladowych przebiegow alpha(t) na wykresie")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent

    cfg_check, cant_vals_in_yaml = check_baseline_cant(args.case, root)
    print(f"cant_angle w configurations/{args.case}.yaml: {cant_vals_in_yaml} deg")
    if cant_vals_in_yaml != [round(args.cant, 5)]:
        print(f"  -> rozni sie od docelowego {args.cant} deg; uzyje tymczasowego YAML "
              f"z cant_angle={args.cant} (oryginalny YAML NIE jest modyfikowany).")
    else:
        print(f"  -> zgodne z docelowym {args.cant} deg.")

    t_ignition = cfg_check.propulsion.t_ignition
    t_burn_nominal = cfg_check.propulsion.t_burn

    burn = load_burn_time_stats(Path(base) / "results" / "burn_time_stats.csv")
    thrust_frac, thrust_mean, thrust_std = load_thrust_mean_std(Path(base) / "results" / "thrust_mean_std.csv")
    mid = (thrust_frac > 0.05) & (thrust_frac < 0.95)
    rel_std = float(np.mean(thrust_std[mid] / thrust_mean[mid]))

    act = actual_stats(base, args.nose)
    print(f"\nKonfiguracja: {args.case}  cant={args.cant}  n_MC={args.n}  "
          f"DATCOM rerun={'NIE (--no-rerun-datcom)' if args.no_rerun_datcom else 'TAK (prawdziwy)'}")
    if act["h_mean"] is not None:
        print(f"Rzeczywiste apogeum GPS (nos '{args.nose}'): mean={act['h_mean']:.1f} m std={act['h_std']:.1f} m")
    if act["v_mean"] is not None:
        print(f"Rzeczywiste V_max (nos '{args.nose}'): mean={act['v_mean']:.1f} m/s std={act['v_std']:.1f} m/s")

    print(f"\n  -> DATCOM dla cant={args.cant:.3f} deg ...", flush=True)
    aero, case_name_cant = get_aero_for_cant(args.case, args.cant, force_rerun=not args.no_rerun_datcom)
    geom = build_geometry(cfg_check)
    import math
    geom.cant_angle_rad = math.radians(args.cant)

    mission = MissionConfig.from_yaml(str(root / "missions" / "mission_01.yaml"))
    atm = create_atmosphere("ISA")
    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)
    initial_state = build_initial_state(mission)

    rng = np.random.default_rng(args.seed)
    rows = []
    traces = []  # (t, total_deg, t_burnout) dla kilku przebiegow

    for i in range(args.n):
        scale = max(rng.normal(1.0, rel_std), 0.3)
        t_burn_draw = float(np.clip(rng.normal(burn["mean_s"], burn["std_s"]),
                                     burn["min_s"], burn["max_s"]))
        mass, prop = build_mc_mass_and_propulsion(cfg_check, t_ignition, t_burn_draw, scale)
        try:
            result = run_one_with_history(aero, geom, atm, gravity, launcher, mass, prop, initial_state)
            metrics = analyze_history(result, t_ignition + t_burn_draw)
        except Exception as e:
            metrics = dict(h_apo=float("nan"), v_max=float("nan"), alpha_max_deg=float("nan"),
                            beta_max_deg=float("nan"), total_max_deg=float("nan"),
                            I_alpha2_coast=float("nan"), t_damp_2deg_s=float("nan"), status=f"error:{e}")
            result = None
        metrics["run"] = i
        metrics["scale"] = scale
        metrics["t_burn_draw"] = t_burn_draw
        rows.append(metrics)

        if result is not None and len(traces) < args.n_traces:
            total_deg = np.sqrt(np.degrees(result.alpha) ** 2 + np.degrees(result.beta) ** 2)
            traces.append((result.t, total_deg, t_ignition + t_burn_draw))

        if (i + 1) % max(1, args.n // 10) == 0:
            print(f"    {i + 1}/{args.n} przebiegow...", flush=True)

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"alpha_damping_mc_{args.case}_{args.nose}.csv"
    fieldnames = ["run", "scale", "t_burn_draw", "h_apo", "v_max", "alpha_max_deg",
                  "beta_max_deg", "total_max_deg", "I_alpha2_coast", "t_damp_2deg_s", "status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nZapisano: {csv_path}")

    ok = [r for r in rows if not np.isnan(r["h_apo"])]
    n_ok = len(ok)
    h_apo = np.array([r["h_apo"] for r in ok])
    v_max = np.array([r["v_max"] for r in ok])
    alpha_max = np.array([r["alpha_max_deg"] for r in ok])
    total_max = np.array([r["total_max_deg"] for r in ok])
    I_alpha2 = np.array([r["I_alpha2_coast"] for r in ok])
    t_damp = np.array([r["t_damp_2deg_s"] for r in ok])

    print(f"\nWyniki ({n_ok}/{args.n} przebiegow OK):")
    print(f"  h_apo:        mean={np.mean(h_apo):.1f} m   std={np.std(h_apo):.1f} m   "
          f"[{np.min(h_apo):.1f}, {np.max(h_apo):.1f}]")
    print(f"  v_max:        mean={np.mean(v_max):.1f} m/s std={np.std(v_max):.1f} m/s")
    print(f"  alpha_max:    mean={np.mean(alpha_max):.2f} deg std={np.std(alpha_max):.2f} deg "
          f"[{np.min(alpha_max):.2f}, {np.max(alpha_max):.2f}]")
    print(f"  total_max:    mean={np.mean(total_max):.2f} deg std={np.std(total_max):.2f} deg")
    n_damp_ok = int(np.sum(~np.isnan(t_damp)))
    print(f"  t_damp_2deg:  {n_damp_ok}/{n_ok} przebiegow zejscia < 2 deg; "
          f"mean={np.nanmean(t_damp):.2f} s std={np.nanstd(t_damp):.2f} s (z tych, co zejdzie)")
    print(f"  I_alpha2_coast: mean={np.mean(I_alpha2):.2f} deg^2*s  std={np.std(I_alpha2):.2f} deg^2*s")

    if act["h_mean"] is not None:
        bias_h = 100.0 * (np.mean(h_apo) - act["h_mean"]) / act["h_mean"]
        print(f"\n  bias apogeum vs GPS: {bias_h:+.1f}%")

    # Korelacja: czy wieksze I_alpha2_coast (proxy oporu indukowanego) tlumaczy
    # NIZSZE apogeum (tj. czy redukuje rozjazd z danymi polowymi)?
    if n_ok > 3 and np.std(I_alpha2) > 1e-9:
        corr_h_I = float(np.corrcoef(h_apo, I_alpha2)[0, 1])
        print(f"\n  Korelacja h_apo vs I_alpha2_coast (Pearson r): {corr_h_I:+.3f}")
        # apogeum przy najmniejszej 10% calce alpha^2 vs przy najwiekszej 10%
        k = max(1, n_ok // 10)
        idx_sorted = np.argsort(I_alpha2)
        low_I = h_apo[idx_sorted[:k]]
        high_I = h_apo[idx_sorted[-k:]]
        print(f"  h_apo @ najmniejsze 10% I_alpha2 (najmniej oporu indukowanego): "
              f"mean={np.mean(low_I):.1f} m")
        print(f"  h_apo @ najwieksze 10% I_alpha2 (najwiecej oporu indukowanego): "
              f"mean={np.mean(high_I):.1f} m")
        print(f"  roznica: {np.mean(low_I) - np.mean(high_I):+.1f} m  "
              f"(jesli mala wzgledem rozjazdu z GPS — alpha/induced drag NIE tlumaczy rozjazdu)")
    else:
        corr_h_I = float("nan")

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    for t, total_deg, t_bo in traces:
        ax.plot(t, total_deg, lw=1.0, alpha=0.7)
        ax.axvline(t_bo, color="gray", lw=0.5, ls=":")
    ax.axhline(2.0, color="red", lw=1.0, ls="--", label="2 deg")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("sqrt(alpha^2+beta^2) [deg]")
    ax.set_title(f"Przykladowe przebiegi kata natarcia ({len(traces)} z {args.n})")
    ax.set_ylim(0, max(10.0, np.nanmax(total_max) * 1.1) if n_ok else 10.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(alpha_max, bins=30, color="tab:blue", alpha=0.7)
    ax.set_xlabel("alpha_max [deg]")
    ax.set_ylabel("Liczba przebiegow")
    ax.set_title("Rozklad max |alpha| w przebiegu")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(I_alpha2, h_apo, s=12, alpha=0.6, color="tab:purple")
    if act["h_mean"] is not None:
        ax.axhline(act["h_mean"], color="tab:red", ls="--", lw=1.5, label="GPS mean")
    ax.set_xlabel("I_alpha2_coast [deg^2 * s]")
    ax.set_ylabel("Apogeum [m]")
    ax.set_title(f"Apogeum vs energia w kacie natarcia (r={corr_h_I:+.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(I_alpha2, v_max, s=12, alpha=0.6, color="tab:green")
    if act["v_mean"] is not None:
        ax.axhline(act["v_mean"], color="tab:red", ls="--", lw=1.5, label="V_max actual mean")
    ax.set_xlabel("I_alpha2_coast [deg^2 * s]")
    ax.set_ylabel("V_max [m/s]")
    ax.set_title("V_max vs energia w kacie natarcia")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Diagnostyka kata natarcia — {args.case}, cant={args.cant} deg, n={args.n}")
    plt.tight_layout()
    out_png = out_dir / f"alpha_damping_mc_{args.case}_{args.nose}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nZapisano: {out_png}")


if __name__ == "__main__":
    main()
