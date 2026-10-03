"""
compare_coast_drag.py
=====================
Cd(Mach) w fazie COAST WZNOSZENIA (burnout -> apogeum): telemetria vs model 6DOF.
Pierwszy pomiar Cd w zakresie transsonicznym (Ma ~1.3 -> ~0.4).

BEZ zyroskopu (nasycony) i BEZ GPS tuz po burnoucie (GPS zamarza na kilka
sekund po spalaniu — dokladnie w zakresie transsonicznym).

Rekonstrukcja trajektorii tylko z akcelerometru osiowego + elewacji wyrzutni
(model gravity turn, rakieta stabilna -> os ciala ~ wektor predkosci):
    dV/dt     = a_x - g*sin(gamma)
    dgamma/dt = -g*cos(gamma) / V            (gamma(0) = elewacja)
    dh/dt     = V*sin(gamma)
gdzie a_x = specific force z akcelerometru (skala skalibrowana na padzie).

Cd w coascie (T=0, akcelerometr mierzy wprost opor):
    Cd_tel = -m_coast * a_x / (q * S),   q = 0.5*rho(h)*V^2

Kontrola rekonstrukcji (niezalezna): wysokosc i czas apogeum vs GPS
(GPS jest juz wtedy wiarygodny). Duzy blad zamkniecia = nie ufac Cd.

Model: ten sam setup per lot co analyze_per_flight_6dof.py (profil ciagu
TEGO lotu jesli istnieje), Cd_mod = -(FX - Fg_x)/(q_dyn*S) w coascie.

Zalozenia/ograniczenia: alpha ~ 0, brak wiatru (predkosc wzgledem
powietrza = wzgledem ziemi), staly bias akcelerometru nieskorygowany
(wplyw ~ m*b*g/(q*S), dlatego odrzucamy q < Q_MIN).

Uzycie:
    python compare_coast_drag.py
    python compare_coast_drag.py 15 19
    python compare_coast_drag.py --no-rerun-datcom
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
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir, parse_telemetry, resolve_data_file
from imu_reconstruction import detect_ignition
from diag_drag import detect_events, calib_acc_scale, G0, S
from run_all_flights import read_configs
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

COAST_START_S = 0.15   # pomin ogon spalania po burnoucie [s]
Q_MIN = 3000.0         # [Pa] ponizej bias akcelerometru dominuje Cd
SMOOTH_S = 0.10        # wygladzanie a_x do Cd [s]
L_RAIL = 3.0           # dlugosc szyny [m] — jak LauncherConfig w modelu
FIT_WIN_S = 5.0        # okno dopasowania do GPS przed apogeum [s]
SIGMA_H, SIGMA_V = 5.0, 2.0   # wagi residuow GPS [m], [m/s]
MACH_BINS = np.arange(0.3, 1.55, 0.1)


def moving_avg(y, t, win_s):
    n = max(1, int(round(win_s / np.median(np.diff(t)))))
    return np.convolve(y, np.ones(n) / n, mode="same")


def gps_valid_mask(t, y, freeze_s=0.6):
    """GPS 'zywy' = wartosc zmienila sie w ostatnich freeze_s sekundach."""
    valid = np.zeros(len(t), bool)
    last_change = -np.inf
    for i in range(1, len(t)):
        if y[i] != y[i - 1]:
            last_change = t[i]
        valid[i] = (t[i] - last_change) < freeze_s
    return valid


def integrate(t, a_x, elev_rad):
    V = np.zeros(len(t)); gam = np.zeros(len(t)); h = np.zeros(len(t))
    gam[0] = elev_rad
    s_path = 0.0
    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        V[k + 1] = max(V[k] + (a_x[k] - G0 * math.sin(gam[k])) * dt, 0.0)
        # na szynie kierunek wymuszony = elewacja
        dg = 0.0 if s_path < L_RAIL else -G0 * math.cos(gam[k]) / max(V[k], 1.0)
        gam[k + 1] = gam[k] + dg * dt
        h[k + 1] = h[k] + V[k] * math.sin(gam[k]) * dt
        s_path += V[k] * dt
    return V, gam, h


def fit_bias_elevation(t, a_x, elev_deg, h_gps, v_gps, mask):
    """Offset akcelerometru b [m/s^2] i poprawka kata startu dgam [deg]
    dopasowane do GPS (h i Vh) w koncowce coastu, gdzie GPS jest wiarygodny.
    Vh_rec = V*cos(gamma) ~ predkosc pozioma GPS."""
    def cost(p):
        V, gam, h = integrate(t, a_x - p[0], math.radians(elev_deg + p[1]))
        return (np.mean(((h[mask] - h_gps[mask]) / SIGMA_H) ** 2)
                + np.mean(((V[mask] * np.cos(gam[mask]) - v_gps[mask]) / SIGMA_V) ** 2))
    res = minimize(cost, x0=[0.0, 0.0], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-4, maxiter=300))
    return float(res.x[0]), float(res.x[1]), float(np.sqrt(res.fun / 2.0))


def reconstruct(fno, elev_deg, atm):
    tel = parse_telemetry(resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt"), verbose=False)
    t_ign_abs = detect_ignition(tel)
    i_ign, i_bo, i_apo, _ = detect_events(tel, t_ign_abs)
    acc_scale, _ = calib_acc_scale(tel, t_ign_abs)

    seg = slice(i_ign, i_apo + int(3.0 / 0.004))
    t = tel.time[seg] - tel.time[i_ign]
    a_raw = tel.acc_x[seg] * acc_scale * G0
    h_gps = tel.alt_onboard[seg] - tel.alt_onboard[i_ign]
    v_gps = tel.vel_onboard[seg]
    valid = gps_valid_mask(t, h_gps)
    t_apo_gps = tel.time[i_apo] - tel.time[i_ign]

    V0, _, h0 = integrate(t, a_raw, math.radians(elev_deg))
    fit_mask = valid & (t > t_apo_gps - FIT_WIN_S) & (t < t_apo_gps + 3.0)
    if np.sum(fit_mask) >= 20:
        b, dgam, fit_rms = fit_bias_elevation(t, a_raw, elev_deg, h_gps, v_gps, fit_mask)
    else:
        b, dgam, fit_rms = 0.0, 0.0, float("nan")

    a_x = a_raw - b
    V, gam, h = integrate(t, a_x, math.radians(elev_deg + dgam))
    i_apo_rec = int(np.argmax(h))
    rho = np.array([atm.at(hh).density for hh in h])
    a_snd = np.array([atm.at(hh).speed_of_sound for hh in h])
    return dict(
        t=t, a_x=a_x, V=V, h=h, gam=gam, mach=V / a_snd, q=0.5 * rho * V**2,
        V_nom=V0, h_nom=h0, bias=b, dgam=dgam, fit_rms=fit_rms, fit_mask=fit_mask,
        t_bo=tel.time[i_bo] - tel.time[i_ign],
        h_gps=h_gps, v_gps=v_gps, gps_valid=valid,
        h_apo_nom=float(h0.max()), h_apo_rec=h[i_apo_rec], t_apo_rec=t[i_apo_rec],
        h_apo_gps=tel.alt_onboard[i_apo] - tel.alt_onboard[i_ign],
        t_apo_gps=t_apo_gps,
    )


def cd_telemetry(rec, m_coast):
    a_s = moving_avg(rec["a_x"], rec["t"], SMOOTH_S)
    coast = ((rec["t"] > rec["t_bo"] + COAST_START_S) & (rec["t"] < rec["t_apo_rec"])
             & (rec["q"] > Q_MIN))
    return rec["mach"][coast], -m_coast * a_s[coast] / (rec["q"][coast] * S), coast


def run_model(aero, geom, atm, mass, prop, s0, t_end, log_dir, tag):
    logger = ForceLogger(case_name=tag, log_dir=log_dir, enabled=True)
    fm = ForceModel6DOF(atmosphere=atm, mass_model=mass, aero_model=aero,
                        gravity=create_gravity("constant"), geometry=geom,
                        propulsion=prop, launcher=LauncherConfig(L_rail=L_RAIL),
                        logger=logger)
    res = run_simulation_6dof(fm, s0, t_max=t_end, dt_output=0.01,
                              rtol=1e-6, atol=1e-8, max_step=0.01)
    logger.close()
    df = pd.read_csv(logger.filepath).sort_values("t").drop_duplicates("t", keep="last")
    t_apo = res.t[int(np.argmax(-res.z))]
    df["a_x"] = (df["FX"] - df["Fg_x"]) / df["mass"]
    coast = (df["F_thrust"] <= 0.0) & (df["t"] > 0.5) & (df["t"] < t_apo) & (df["q_dyn"] > Q_MIN)
    df["coast"] = coast
    df["cd"] = -(df["FX"] - df["Fg_x"]) / (df["q_dyn"] * geom.S_ref)
    return df, t_apo, float(np.max(-res.z))


def load_descent_cd(base, nose):
    p = Path(base) / "results" / "fit_cd_mach_curves.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d["nose"] == nose].sort_values("Mach")
    return (d["Mach"].values, d["Cd"].values) if len(d) else None


def binned(mach, cd):
    idx = np.digitize(mach, MACH_BINS)
    out = {}
    for b in range(1, len(MACH_BINS)):
        sel = idx == b
        if np.sum(sel) >= 10:
            out[round(0.5 * (MACH_BINS[b - 1] + MACH_BINS[b]), 2)] = (float(np.median(cd[sel])), int(np.sum(sel)))
    return out


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
    tmp = tempfile.TemporaryDirectory(prefix="coast_cd_", ignore_cleanup_errors=True)
    log_dir = Path(tmp.name)

    case_by_nose = {"ostra": args.case_ostra, "tepa": args.case_tepa}
    m_coast_by_fno = {r["fno"]: r["m_coast"] for r in read_configs(base)}
    flights = read_flights(base)
    if args.flights:
        flights = [r for r in flights if r["fno"] in args.flights]

    bin_rows, closure_rows, agg = [], [], {}
    for r in flights:
        fno, nose = r["fno"], r["nose"]
        case = case_by_nose[nose]
        cfg = load_config(str(root / "configurations" / f"{case}.yaml"))
        t_ign = cfg.propulsion.t_ignition
        atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])

        rec = reconstruct(fno, r["elevation"], atm)
        m_coast = m_coast_by_fno[fno]
        M_tel, cd_tel, coast_tel = cd_telemetry(rec, m_coast)

        aero, _ = get_aero_for_cant(case, r["cant"], force_rerun=not args.no_rerun_datcom)
        geom = build_geometry(cfg)
        geom.cant_angle_rad = math.radians(r["cant"])
        mass = build_scaled_mass(cfg, t_ign, r["m_rocket"])
        prop = build_flight_thrust(base, fno, t_ign)
        variant = "adjusted"
        if prop is None:
            prop, variant = build_propulsion(cfg), "baseline"
        s0 = State6DOF.initial(elevation_deg=r["elevation"], azimuth_deg=r["azimuth"])
        df, t_apo_mod, h_apo_mod = run_model(aero, geom, atm, mass, prop, s0,
                                             t_ign + rec["t_apo_gps"] + 5.0,
                                             log_dir, f"coast_{fno}")
        df["t"] -= t_ign
        dm = df[df["coast"]]

        # model z katem startu skorygowanym o dgam z dopasowania do GPS —
        # rozdziela wplyw ksztaltu toru (kat) od wplywu oporu na apogeum
        s0_fit = State6DOF.initial(elevation_deg=r["elevation"] + rec["dgam"],
                                   azimuth_deg=r["azimuth"])
        df_fit, _, h_apo_mod_fit = run_model(aero, geom, atm, mass, prop, s0_fit,
                                             t_ign + rec["t_apo_gps"] + 5.0,
                                             log_dir, f"coast_{fno}_fit")
        df_fit["t"] -= t_ign

        h_gps_apo = rec["h_apo_gps"]
        pct = lambda h: 100.0 * (h - h_gps_apo) / h_gps_apo
        dh_pct = pct(rec["h_apo_rec"])
        closure_rows.append(dict(
            flight_no=fno, nose=nose, thrust=variant, m_coast=m_coast,
            acc_bias_g=rec["bias"] / G0, dgamma_deg=rec["dgam"], fit_rms=rec["fit_rms"],
            h_apo_gps=h_gps_apo,
            h_apo_rec_nominal=rec["h_apo_nom"], dh_rec_nominal_pct=pct(rec["h_apo_nom"]),
            h_apo_rec_fit=rec["h_apo_rec"], dh_rec_fit_pct=dh_pct,
            h_apo_model=h_apo_mod, dh_model_pct=pct(h_apo_mod),
            h_apo_model_elevfit=h_apo_mod_fit, dh_model_elevfit_pct=pct(h_apo_mod_fit),
            t_apo_gps=rec["t_apo_gps"], t_apo_rec=rec["t_apo_rec"],
            t_apo_model=t_apo_mod - t_ign,
            mach_bo_rec=float(rec["mach"][rec["t"] <= rec["t_bo"]][-1]),
        ))
        print(f"lot {fno:3d} ({nose}): bias acc={rec['bias']/G0:+.3f} g  dgam={rec['dgam']:+.2f} deg  "
              f"(fit rms {rec['fit_rms']:.2f}) | apogeum vs GPS {h_gps_apo:.0f} m: "
              f"rek. nominal {pct(rec['h_apo_nom']):+.1f}%  rek. fit {dh_pct:+.1f}%  "
              f"model {pct(h_apo_mod):+.1f}%  model(kat fit) {pct(h_apo_mod_fit):+.1f}%")

        b_tel, b_mod = binned(M_tel, cd_tel), binned(dm["mach"].values, dm["cd"].values)
        for mach, (cd_t, n) in b_tel.items():
            cd_m = b_mod.get(mach, (np.nan, 0))[0]
            bin_rows.append(dict(flight_no=fno, nose=nose, mach=mach, cd_tel=cd_t,
                                 cd_model=cd_m, ratio_tel_model=cd_t / cd_m, n=n,
                                 dh_rec_pct=dh_pct))
            agg.setdefault(nose, []).append((mach, cd_t / cd_m))

        # --- wykres per lot ---
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        t = rec["t"]
        ax[0, 0].plot(t, moving_avg(rec["a_x"], t, SMOOTH_S) / G0, "k", lw=1, label="telemetria")
        ax[0, 0].plot(df["t"], df["a_x"] / G0, "tab:red", lw=1.2, label=f"model ({variant})")
        ax[0, 0].set_xlim(rec["t_bo"] - 0.3, rec["t_apo_gps"])
        ax[0, 0].set_ylim(-15, 2)
        ax[0, 0].set_ylabel("a_x [g]"); ax[0, 0].set_title("Przyspieszenie osiowe w coascie")

        v = rec["gps_valid"]
        ax[0, 1].plot(t, rec["V_nom"], ":", color="gray", lw=1, label="rekonstrukcja nominalna")
        ax[0, 1].plot(t, rec["V"], "k", lw=1.2, label="rekonstrukcja (bias+kat z GPS)")
        ax[0, 1].plot(df["t"], df["speed"], "tab:red", lw=1.2, label="model")
        ax[0, 1].plot(df_fit["t"], df_fit["speed"], "--", color="tab:purple", lw=1.2,
                      label=f"model, kat {rec['dgam']:+.1f} deg")
        ax[0, 1].plot(t[v], rec["v_gps"][v], ".", color="tab:green", ms=2,
                      label="GPS Vh (tylko gdy nie zamrozony)")
        ax[0, 1].set_ylabel("V [m/s]"); ax[0, 1].set_title("Predkosc")

        ax[1, 0].plot(t, rec["h_nom"], ":", color="gray", lw=1, label="rekonstrukcja nominalna")
        ax[1, 0].plot(t, rec["h"], "k", lw=1.2, label="rekonstrukcja (bias+kat z GPS)")
        ax[1, 0].plot(df["t"], -df["z"], "tab:red", lw=1.2, label="model")
        ax[1, 0].plot(df_fit["t"], -df_fit["z"], "--", color="tab:purple", lw=1.2,
                      label=f"model, kat {rec['dgam']:+.1f} deg")
        fm = rec["fit_mask"]
        ax[1, 0].plot(t[fm], rec["h_gps"][fm], "o", color="tab:green", ms=3, mfc="none",
                      label="GPS uzyty w dopasowaniu")
        ax[1, 0].plot(t[v], rec["h_gps"][v], ".", color="tab:green", ms=2, label="GPS")
        ax[1, 0].set_ylabel("h AGL [m]"); ax[1, 0].set_xlabel("czas od zaplonu [s]")
        ax[1, 0].set_title(f"Wysokosc — dopasowanie: bias {rec['bias']/G0:+.3f} g, "
                           f"kat {rec['dgam']:+.2f} deg, apogeum {dh_pct:+.1f}%")

        ax[1, 1].plot(M_tel, cd_tel, ".", color="gray", ms=1.5, alpha=0.4, label="telemetria (probki)")
        if b_tel:
            ax[1, 1].plot(list(b_tel), [v[0] for v in b_tel.values()], "ko-", lw=1.5,
                          label="telemetria (mediana w binie)")
        ax[1, 1].plot(dm["mach"], dm["cd"], "tab:red", lw=1.5, label="model 6DOF (DATCOM)")
        desc = load_descent_cd(base, nose)
        if desc is not None:
            ax[1, 1].plot(desc[0], desc[1], "--", color="tab:green", lw=1.5,
                          label="fit ze znizania (fit_cd_mach)")
        ax[1, 1].set_xlabel("Mach"); ax[1, 1].set_ylabel("Cd (osiowe)")
        ax[1, 1].set_ylim(0, 1.2); ax[1, 1].set_title("Cd(Mach) w coascie wznoszenia")
        for a in ax.ravel():
            a.grid(alpha=0.3); a.legend(fontsize=8)
        fig.suptitle(f"Lot {fno} ({nose}, cant={r['cant']:.1f} deg, m_coast={m_coast:.2f} kg)",
                     fontweight="bold")
        fig.tight_layout()
        png = out_dir / f"coast_cd_flight_{fno}.png"
        if png.exists():
            png.unlink()
        fig.savefig(png, dpi=130, bbox_inches="tight")
        plt.close(fig)

    tmp.cleanup()
    if not bin_rows:
        print("Brak wynikow.")
        return

    pd.DataFrame(closure_rows).to_csv(out_dir / "coast_cd_closure.csv", index=False, float_format="%.4f")
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(out_dir / "coast_cd_bins.csv", index=False, float_format="%.4f")

    # --- zbiorczo: Cd_tel / Cd_model vs Mach ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for nose, col in [("ostra", "tab:blue"), ("tepa", "tab:orange")]:
        d = bins[bins["nose"] == nose]
        if d.empty:
            continue
        ax.plot(d["mach"], d["ratio_tel_model"], "o", color=col, alpha=0.35, ms=4)
        g = d.groupby("mach")["ratio_tel_model"].median()
        ax.plot(g.index, g.values, "-o", color=col, lw=2, label=f"{nose} (mediana)")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("Mach"); ax.set_ylabel("Cd_telemetria / Cd_model")
    ax.set_title("Opor w coascie wznoszenia: telemetria vs model (>1 = model za maly opor)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    png = out_dir / "coast_cd_ratio_summary.png"
    if png.exists():
        png.unlink()
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nZapisano: coast_cd_bins.csv, coast_cd_closure.csv, coast_cd_ratio_summary.png, "
          f"coast_cd_flight_*.png  ({out_dir})")
    print("\nCd_tel / Cd_model — mediana per Mach (wszystkie loty):")
    print(bins.groupby(["nose", "mach"])["ratio_tel_model"].agg(["median", "count"]).round(2).to_string())


if __name__ == "__main__":
    main()
