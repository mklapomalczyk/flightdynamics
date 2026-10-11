"""
compare_burn_acceleration.py
============================
Porownanie przyspieszenia osiowego w fazie spalania: model 6DOF vs telemetria.

Akcelerometr mierzy SPECIFIC FORCE (bez grawitacji), wiec porownujemy z
modelem te sama wielkosc:
    telemetria: a_x = acc_x * acc_scale * G0
    model:      a_x = (FX - Fg_x) / m = (F_thrust + FA_x) / m

Dwa warianty modelu (te same co w analyze_per_flight_6dof.py):
  baseline  — usredniony profil ciagu z YAML (NIEZALEZNY od akcelerometru
              tego lotu -> uczciwy test)
  adjusted  — profil ciagu TEGO lotu z kalibracji Pc->T (estimate_thrust.py).
              UWAGA: ten ciag byl wyznaczony Z TEGO SAMEGO akcelerometru
              (T = m*a + D w oknie Ma 0.2-0.5), wiec zgodnosc jest czesciowo
              wymuszona. Wartosc testu: zgodnosc POZA oknem kalibracyjnym.

Drugi panel: calka a_x dt (przyrost predkosci od sil wlasnych) — pokazuje,
czy calkowity impuls netto (ciag - opor) w spalaniu sie zgadza.

Uzycie:
    python compare_burn_acceleration.py                 # wszystkie loty
    python compare_burn_acceleration.py 15 19           # wybrane loty
    python compare_burn_acceleration.py --no-rerun-datcom
"""

import sys
import math
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir, parse_telemetry, resolve_data_file
from imu_reconstruction import detect_ignition
from diag_drag import detect_events, calib_acc_scale, G0
from analyze_per_flight_6dof import read_flights, build_scaled_mass, build_flight_thrust
from run_6dof_cant_montecarlo import get_aero_for_cant
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_propulsion, build_geometry
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from models.force_logger import ForceLogger
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF

PRE_S = 0.2     # okno przed zaplonem [s]
POST_S = 0.6    # okno po burnoucie [s]


def telemetry_accel(fno):
    tel = parse_telemetry(resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt"), verbose=False)
    t_ign = detect_ignition(tel)
    i_ign, i_bo, _, _ = detect_events(tel, t_ign)
    acc_scale, _ = calib_acc_scale(tel, t_ign)
    t = tel.time - tel.time[i_ign]
    t_bo = t[i_bo]
    m = (t >= -PRE_S) & (t <= t_bo + POST_S)
    return t[m], tel.acc_x[m] * acc_scale * G0, t_bo


def model_accel(aero, geom, atm, mass, prop, initial_state, t_end, log_dir, tag):
    logger = ForceLogger(case_name=tag, log_dir=log_dir, enabled=True)
    fm = ForceModel6DOF(atmosphere=atm, mass_model=mass, aero_model=aero,
                        gravity=create_gravity("constant"), geometry=geom,
                        propulsion=prop, launcher=LauncherConfig(L_rail=3.0),
                        logger=logger)
    run_simulation_6dof(fm, initial_state, t_max=t_end, dt_output=0.005,
                        rtol=1e-6, atol=1e-8, max_step=0.005)
    logger.close()
    df = pd.read_csv(logger.filepath)
    Path(logger.filepath).unlink()
    # logger zapisuje kazdy etap RK (czasy niemonotoniczne) -> sortuj i usun duplikaty
    df = df.sort_values("t").drop_duplicates("t", keep="last")
    a = (df["FX"].values - df["Fg_x"].values) / df["mass"].values
    return df["t"].values, a


def cumtrapz(y, x):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def stats(t_tel, a_tel, t_mod, a_mod, t_bo):
    burn = (t_tel >= 0.0) & (t_tel <= t_bo)
    a_m = np.interp(t_tel[burn], t_mod, a_mod)
    err = a_m - a_tel[burn]
    dv_tel = np.trapezoid(a_tel[burn], t_tel[burn])
    dv_mod = np.trapezoid(a_m, t_tel[burn])
    return dict(rms_g=np.sqrt(np.mean(err**2)) / G0,
                bias_g=np.mean(err) / G0,
                peak_tel_g=a_tel[burn].max() / G0,
                peak_mod_g=a_m.max() / G0,
                dv_tel=dv_tel, dv_mod=dv_mod,
                dv_err_pct=100.0 * (dv_mod - dv_tel) / dv_tel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flights", nargs="*", type=int)
    ap.add_argument("--case-ostra", default="rocket_70mm_baseline")
    ap.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    ap.add_argument("--no-rerun-datcom", action="store_true")
    args = ap.parse_args()

    base = get_data_dir()
    root = Path(base).parent
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    # poza OneDrive — synchronizacja blokuje katalog i rmdir rzuca PermissionError
    tmp = tempfile.TemporaryDirectory(prefix="burn_accel_", ignore_cleanup_errors=True)
    log_dir = Path(tmp.name)

    case_by_nose = {"ostra": args.case_ostra, "tepa": args.case_tepa}
    flights = read_flights(base)
    if args.flights:
        flights = [r for r in flights if r["fno"] in args.flights]

    rows = []
    for r in flights:
        fno = r["fno"]
        case = case_by_nose[r["nose"]]
        cfg = load_config(str(root / "configurations" / f"{case}.yaml"))
        t_ign = cfg.propulsion.t_ignition

        t_tel, a_tel, t_bo = telemetry_accel(fno)

        aero, _ = get_aero_for_cant(case, r["cant"], force_rerun=not args.no_rerun_datcom)
        geom = build_geometry(cfg)
        geom.cant_angle_rad = math.radians(r["cant"])
        mass = build_scaled_mass(cfg, t_ign, r["m_rocket"])
        atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])
        s0 = State6DOF.initial(elevation_deg=r["elevation"], azimuth_deg=r["azimuth"])
        t_end = t_ign + t_bo + POST_S

        variants = {"baseline": build_propulsion(cfg),
                    "adjusted": build_flight_thrust(base, fno, t_ign)}

        fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        ax[0].plot(t_tel, a_tel / G0, "k", lw=1.0, label="telemetria")
        burn = t_tel >= 0
        ax[1].plot(t_tel[burn], cumtrapz(a_tel[burn], t_tel[burn]), "k", lw=1.2,
                   label="telemetria")

        line = f"lot {fno:3d} ({r['nose']}):"
        for (name, prop), col in zip(variants.items(), ["tab:blue", "tab:red"]):
            if prop is None:
                line += f"  {name}: brak kalibracji ciagu"
                continue
            t_mod, a_mod = model_accel(aero, geom, atm, mass, prop, s0, t_end,
                                       log_dir, f"acc_{fno}_{name}")
            t_mod = t_mod - t_ign
            s = stats(t_tel, a_tel, t_mod, a_mod, t_bo)
            rows.append(dict(flight_no=fno, nose=r["nose"], variant=name,
                             t_burn_tel_s=t_bo, **s))
            ax[0].plot(t_mod, a_mod / G0, color=col, lw=1.3,
                       label=f"model {name} (RMS {s['rms_g']:.2f} g)")
            mb = t_mod >= 0
            ax[1].plot(t_mod[mb], cumtrapz(a_mod[mb], t_mod[mb]), color=col, lw=1.3,
                       label=f"model {name} (dV {s['dv_err_pct']:+.1f}%)")
            line += f"  {name}: RMS={s['rms_g']:.2f}g dV={s['dv_err_pct']:+.1f}%"
        print(line)

        for a in ax:
            a.axvline(0.0, color="gray", ls=":", lw=0.8)
            a.axvline(t_bo, color="gray", ls="--", lw=0.8)
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        ax[0].set_ylabel("a_x specific force [g]")
        ax[0].set_title(f"Lot {fno} ({r['nose']}, cant={r['cant']:.1f} deg) — "
                        f"przyspieszenie w spalaniu (burnout tel. = {t_bo:.2f} s)")
        ax[1].set_ylabel("calka a_x dt [m/s]")
        ax[1].set_xlabel("czas od zaplonu [s]")
        fig.tight_layout()
        png = out_dir / f"burn_accel_flight_{fno}.png"
        if png.exists():
            png.unlink()
        fig.savefig(png, dpi=130, bbox_inches="tight")
        plt.close(fig)

    tmp.cleanup()
    if not rows:
        print("Brak wynikow.")
        return
    out = pd.DataFrame(rows)
    csv_path = out_dir / "burn_accel_summary.csv"
    out.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nZapisano: {csv_path}")
    print(out.groupby("variant")[["rms_g", "bias_g", "dv_err_pct"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
