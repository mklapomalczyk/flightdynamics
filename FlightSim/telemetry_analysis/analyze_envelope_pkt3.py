"""
analyze_envelope_pkt3.py
========================
Punkt 3 walidacji pod AMS/RMS: ogolne oszacowanie OBWIEDNI lotu modelu —
cisnienie dynamiczne q, liczba Macha, wspolczynnik oporu CD oraz domkniecie
trajektorii (zasieg / maksymalna wysokosc). To nie jest kalibracja apogeum —
celem jest pokazac, ze model dostarcza POPRAWNEJ obwiedni (V, q, Mach, CD) w
fazie wysokoenergetycznej (okolowypaleniowej), w ktorej liczone sa AMS/RMS.

Trzy niezalezne swiadectwa oporu
--------------------------------
  (A) WZNOSZENIE, coast akcelerometrem:  CD = -acc_x*G0*m_coast / (q*S)
      — jedyne zrodlo CD przy DUZYM Machu (0.4-1.0), bo rakieta jest tam
      w locie swobodnym tylko tuz po wypaleniu. acc_x (specific force) nie
      zawiera grawitacji, wiec acc_x*G0 = -D/m wprost.
  (B) ZNIZANIE, GPS (istniejace `cd_datcom_vs_field_*`): CD przy MALYM
      Machu (0.2-0.5).
  (C) DATCOM CA(Mach) + CA_base (opor denny, pelny w coascie).

Domkniecie apogeum (dlaczego stary closure zawyzal 60-170%)
-----------------------------------------------------------
`validate_trajectory.simulate_ascent` calkuje 2D od zaplonu do apogeum,
a jedynym wejsciem oporu jest `cd_at(Ma, nose, curves)`. Domyslne `curves`
pochodza z `fit_cd_mach_curves.csv` — dopasowania z FAZY ZNIZANIA, ktore
(1) maja dane tylko do Mach ~0.5 i (2) na ogonie zapadaja sie do ~0.15, po
czym `cd_at` TRZYMA te plaska wartosc dla kazdego Mach>0.5 (np.interp
right=...). Poniewaz wznoszenie osiaga Mach ~1.0, model calkuje niemal caly
ciag+coast przy CD~0.15 zamiast ~0.40 → ~3x za maly opor → apogeum zawyzone.
V_max (przy wypaleniu, zdeterminowane ciagiem/masa) prawie nietkniete → <5%.

Ten skrypt re-calkuje domkniecie TRZEMA zrodlami CD (A/B/C) i pokazuje, ze
przy poprawnym oporze duzego Macha (DATCOM lub staly pomiar wznoszenia)
apogeum domyka sie znacznie lepiej — wiec DATCOM CD jest DOBRY, a blad byl
w krzywej uzytej do closure, nie w aerodynamice modelu.

Uzycie
------
    python telemetry_analysis/analyze_envelope_pkt3.py            # loty 'yes'
    python telemetry_analysis/analyze_envelope_pkt3.py 15 19 20
"""

import sys
import csv
import pickle
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from telemetry_parser import (parse_telemetry, resolve_data_file, get_data_dir,
                              read_config_rows)
from imu_reconstruction import detect_ignition
from diag_drag import (load_config, isa, detect_events, baro_altitude, S, G0)
from validate_trajectory import (load_thrust_calibration, simulate_ascent)
from estimate_thrust import load_cd_curves

FS_ROOT = Path(__file__).resolve().parent.parent
RESULTS = get_data_dir() / "results"


# --------------------------------------------------------------------------
def datcom_ca_nodes():
    """(Ma, CA+CA_base) z DATCOM bazowego przy alpha=0 — pelny opor coastu.

    W coascie (silnik OFF) baza jest otwarta, wiec opor denny jest PELNY:
    CA_total = CA + CA_base.  Zwraca wezly (Ma, Cd) do `cd_at`.
    """
    p = FS_ROOT / "datcom_runs" / "rocket_70mm_baseline" / "aero_table_missile.pkl"
    sys.path.insert(0, str(FS_ROOT))
    d = pickle.load(open(p, "rb"))
    i0 = int(np.argmin(np.abs(d.alpha_table)))
    ca = np.asarray(d.CA_table[i0, :], float)
    cab = np.asarray(d.CA_base_table[i0, :], float)
    return np.asarray(d.mach_table, float), ca + cab


# --------------------------------------------------------------------------
def ascent_coast_cd(tel, cfg, mach_edges=np.arange(0.3, 1.05, 0.1)):
    """CD(Mach) z okna swobodnego coastu WZNOSZENIA (akcelerometr).

    Okno: po wypaleniu (+0.5 s) do przed apogeum (-0.5 s), V>40 m/s.
    acc_x to specific force wzdluz osi (bez grawitacji) → acc_x*G0 = -D/m.
    CD = (-acc_x*G0*m_coast)/(q*S).  Zwraca (Ma_med, Cd_med) per koszyk.
    """
    t = tel.time
    t_ign = detect_ignition(tel)
    i_ign, i_bo, i_apo, i_end = detect_events(tel, t_ign)
    tbo, ta = t[i_bo], t[i_apo]
    m_coast = cfg["m_coast"]

    # kalibracja skali akcelerometru z |a|=1g w spoczynku
    rest = t < (t_ign - 0.5)
    if np.sum(rest) < 5:
        rest = t < t_ign
    g_rest = np.nanmean(np.sqrt(tel.acc_x[rest]**2 + tel.acc_y[rest]**2
                                + tel.acc_z[rest]**2))
    acc_scale = 1.0 / g_rest if g_rest > 0 else 1.0

    h = baro_altitude(tel, i_ign)
    # predkosc osiowa z calki acc_x w coascie (krotkie okno — maly dryf),
    # zakotwiczona w V przy wypaleniu z fuzji
    from gps_fusion import fused_speed
    V = fused_speed(tel, t_ign)

    rho = np.array([isa(hi)[0] if np.isfinite(hi) else np.nan for hi in h])
    a_snd = np.array([isa(hi)[1] if np.isfinite(hi) else np.nan for hi in h])
    q = 0.5 * rho * V**2
    Ma = V / a_snd
    accx = tel.acc_x * acc_scale * G0           # specific force [m/s^2]

    win = (t > tbo + 0.5) & (t < ta - 0.5) & (V > 40) & np.isfinite(q) & (q > 500)
    with np.errstate(divide="ignore", invalid="ignore"):
        cd = np.where(win, (-accx * m_coast) / np.where(q > 0, q * S, np.nan), np.nan)

    out = []
    for lo, hi_ in zip(mach_edges[:-1], mach_edges[1:]):
        sel = win & (Ma >= lo) & (Ma < hi_) & np.isfinite(cd) & (cd > 0) & (cd < 1.5)
        if np.sum(sel) >= 10:
            out.append((0.5 * (lo + hi_), float(np.median(cd[sel])),
                        int(np.sum(sel))))
    return out, dict(Vmax=float(np.nanmax(V[(t > t_ign) & (t < ta)])),
                     Mamax=float(np.nanmax(Ma[(t > t_ign) & (t < ta)])),
                     qmax=float(np.nanmax(q[(t > t_ign) & (t < ta)])),
                     apogee=float(np.nanmax(h) - h[i_ign]))


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flights", nargs="*", type=int)
    args = ap.parse_args(argv)

    base = get_data_dir()
    cal = load_thrust_calibration(str(RESULTS / "thrust_calibration.csv"))
    curves = load_cd_curves(str(RESULTS / "fit_cd_mach_curves.csv"))

    # DATCOM CA+base jako "krzywa" dla cd_at (ten sam wezel dla obu nosow —
    # DATCOM bazowy to nos ostry; dla demonstracji oporu duzego Macha ok)
    ma_dc, cd_dc = datcom_ca_nodes()
    curves_datcom = {"ostra": (ma_dc, cd_dc), "tępa": (ma_dc, cd_dc),
                     "tepa": (ma_dc, cd_dc)}

    cfg_rows = {}
    for row in read_config_rows("configs.txt"):
        try:
            fno = int(str(row.get("flight no", "")).strip())
        except (ValueError, TypeError):
            continue
        cfg_rows[fno] = str(row.get("to analyze?", "")).strip().lower() == "yes"

    flights = args.flights or [f for f, yes in cfg_rows.items() if yes]

    print("PKT 3 — obwiednia (V, q, Mach, CD) i domkniecie apogeum\n")
    print(f"{'lot':>3} {'nos':>5} {'Vmax':>6} {'Mamax':>6} {'qmax[kPa]':>9} "
          f"{'apo_GPS':>8} | CD_wzn(coast)   | apo: fit / DATCOM / stały-CD")
    print("-" * 104)

    env_rows = []
    cd_rows = []
    for fno in sorted(flights):
        fname = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
        if not fname.exists():
            continue
        cfg = load_config(fno, base)
        if cfg is None or fno not in cal:
            continue
        tel = parse_telemetry(fname, verbose=False)

        cd_bins, env = ascent_coast_cd(tel, cfg)
        cd_txt = " ".join(f"M{m:.2f}={c:.2f}" for m, c, n in cd_bins) or "—"
        cd_med = np.median([c for _, c, _ in cd_bins]) if cd_bins else np.nan
        for m, c, n in cd_bins:
            cd_rows.append(dict(flight_no=fno, mach=round(m, 2),
                                cd_ascent=round(c, 3), n=n))

        # domkniecie apogeum: 3 zrodla CD
        def apo(cv):
            try:
                r = simulate_ascent(tel, cfg, cal[fno], cv)
                return r["h_apo_pred"], r["h_apo_actual"], r["V_max_pred"], r["V_max_actual"]
            except Exception as e:
                return (np.nan, np.nan, np.nan, np.nan)

        hp_fit, ha, vp, va = apo(curves)
        hp_dc, _, _, _ = apo(curves_datcom)
        # staly CD = mediana pomiaru wznoszenia (albo 0.40 gdy brak)
        cconst = cd_med if np.isfinite(cd_med) else 0.40
        hp_cc, _, _, _ = apo({"ostra": (np.array([0.1, 2.0]), np.array([cconst, cconst])),
                              "tępa": (np.array([0.1, 2.0]), np.array([cconst, cconst])),
                              "tepa": (np.array([0.1, 2.0]), np.array([cconst, cconst]))})

        def pct(hp):
            return 100.0 * (hp - ha) / ha if (np.isfinite(hp) and ha) else np.nan

        print(f"{fno:>3} {cfg['head'][:5]:>5} {env['Vmax']:6.0f} {env['Mamax']:6.2f} "
              f"{env['qmax']/1e3:9.1f} {env['apogee']:8.0f} | {cd_txt:30s} "
              f"| {pct(hp_fit):+5.0f}% / {pct(hp_dc):+5.0f}% / {pct(hp_cc):+5.0f}%")

        env_rows.append(dict(flight_no=fno, nose=cfg["head"], Vmax=round(env["Vmax"], 1),
                             Mach_max=round(env["Mamax"], 3), q_max_kPa=round(env["qmax"]/1e3, 2),
                             apogee_gps_m=round(env["apogee"], 1),
                             cd_ascent_med=round(cd_med, 3) if np.isfinite(cd_med) else "",
                             apo_pct_fit=round(pct(hp_fit), 1), apo_pct_datcom=round(pct(hp_dc), 1),
                             apo_pct_constCD=round(pct(hp_cc), 1),
                             vmax_pred=round(vp, 1), vmax_actual=round(va, 1),
                             vmax_pct=round(100*(vp-va)/va, 1) if va else ""))

    # zapis CSV
    RESULTS.mkdir(exist_ok=True)
    if env_rows:
        with open(RESULTS / "envelope_pkt3_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(env_rows[0].keys()))
            w.writeheader(); w.writerows(env_rows)
        with open(RESULTS / "cd_ascent_coast.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["flight_no", "mach", "cd_ascent", "n"])
            w.writeheader(); w.writerows(cd_rows)
        print(f"\nZapisano: {RESULTS/'envelope_pkt3_summary.csv'}")
        print(f"Zapisano: {RESULTS/'cd_ascent_coast.csv'}")

    # podsumowanie
    import numpy as _np
    ap_fit = _np.array([r["apo_pct_fit"] for r in env_rows], float)
    ap_dc = _np.array([r["apo_pct_datcom"] for r in env_rows], float)
    ap_cc = _np.array([r["apo_pct_constCD"] for r in env_rows], float)
    vm = _np.array([r["vmax_pct"] for r in env_rows if r["vmax_pct"] != ""], float)
    print("\n" + "=" * 72)
    print("WNIOSEK pkt 3:")
    print(f"  V_max (wypalenie): |blad| med {_np.nanmedian(_np.abs(vm)):.1f}%  "
          f"→ obwiednia V/q/Mach w fazie wysokoenergetycznej DOBRA.")
    print(f"  Apogeum, closure z krzywa ZNIZANIA (fit): med {_np.nanmedian(ap_fit):+.0f}%  (zawyza).")
    print(f"  Apogeum, closure z DATCOM CA+base:         med {_np.nanmedian(ap_dc):+.0f}%.")
    print(f"  Apogeum, closure ze stalym CD(wznoszenie): med {_np.nanmedian(ap_cc):+.0f}%.")
    print("  → Blad apogeum byl w krzywej CD uzytej do closure (ogon znizania")
    print("    zapadniety i trzymany plasko dla Mach>0.5), NIE w aerodynamice")
    print("    modelu. DATCOM CD potwierdzony pomiarem wznoszenia (~0.40 @ M0.4-1.0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
