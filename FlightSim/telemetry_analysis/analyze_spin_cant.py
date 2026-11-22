"""
analyze_spin_cant.py
====================
Punkt 1 nowej walidacji: prędkość obrotowa (spin) w funkcji kąta zaklinowania
stateczników — na potrzeby metody AMS/RMS (envelope: spin rate wchodzi do
AMS(ω) i limitu wychyleń sterów).

Z lotu wyciągamy TRZY rzeczy, każdą z właściwego fragmentu przebiegu:

  1. ω_spin (równowaga napędowa)  — pik spinu przy wypaleniu silnika.
     Tam napęd od zaklinowania równoważy tłumienie: p·d/2V ≈ −(Clδ/Clp)·δ.
  2. Tłumienie (Clp, obserwabla)  — ZANIK spinu w wybiegu (brak napędu rolla),
     mierzone jako stała czasowa zaniku τ. Konwersja τ→Clp wymaga Ixx, q i jest
     zostawiona do kroku porównania z modelem.
  3. p·d/2V (bezwymiarowo)        — vs cant i Mach; nachylenie dopasowania po
     cant daje stosunek Clδ/Clp, wyraz wolny „a" — niesymetrię wykonania.

Źródło spinu
------------
Kanał AX / RXO ("Prędkość obrotowa AX [st./s]") jest PRZELICZONY do °/s i SIĘ
NIE NASYCA (żyroskop roll jest przycięty na ±2000 °/s). Bierzemy go wprost jako
°/s, odejmując baseline przedstartowy (spin = 0 przed startem). Magnetometr
(częstotliwość) służy jako NIEZALEŻNA kontrola (i jedyne źródło dla lotu bez
MGLO/AX, np. 23). Żyroskop NIE jest używany do skalowania.

Uwaga: kanał AX to |roll rate| (moduł), więc analiza jest na module prędkości;
znak/niesymetria kierunkowa poza zakresem tej wersji.

Użycie
------
    python telemetry_analysis/analyze_spin_cant.py                 # wszystkie 'yes'
    python telemetry_analysis/analyze_spin_cant.py 18 19 22 24     # wybrane
    python telemetry_analysis/analyze_spin_cant.py --nose ostra    # tylko ostra
    python telemetry_analysis/analyze_spin_cant.py --no-plots
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from telemetry_parser import (parse_telemetry, resolve_data_file, get_data_dir,
                              read_config_rows, load_flight_config)
from imu_reconstruction import (detect_ignition, calibrate,
                                initial_orientation, reconstruct)

D_REF = 0.070          # średnica odniesienia [m]
R_AIR = 287.05
RESULTS = get_data_dir() / "results"


# --------------------------------------------------------------------------
def isa_sound_speed(h_m):
    """Prędkość dźwięku wg ISA (troposfera) [m/s]. h w [m ASL]."""
    T = 288.15 - 0.0065 * np.clip(h_m, 0.0, 20000.0)
    return np.sqrt(1.4 * R_AIR * T)


def ax_channel(tel):
    """Zwraca kanał AX (stary format) lub acc_RXO (nowy) — spin w [°/s], lub None."""
    ax = tel.raw_columns.get("gyro_ax")
    if ax is None:
        ax = tel.raw_columns.get("acc_RXO_degps")
    return ax


def total_speed(tel):
    """Całkowita prędkość [m/s]: z 3 składowych GPS, inaczej prędkość pokładowa."""
    vN = tel.raw_columns.get("vel_north_ms")
    if vN is not None:
        vE = tel.raw_columns.get("vel_east_ms")
        vD = tel.raw_columns.get("vel_down_ms")
        return np.sqrt(np.nan_to_num(vN) ** 2 + np.nan_to_num(vE) ** 2
                       + np.nan_to_num(vD) ** 2)
    v = tel.raw_columns.get("vel")
    return v if v is not None else tel.vel_onboard


def _smooth(x, n=25):
    """Mediana ruchoma (odporna na szpilki)."""
    n = max(3, int(n) | 1)
    pad = n // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = np.nanmedian(xp[i:i + n])
    return out


def extract_spin(tel, t_ign):
    """p(t) = AX − baseline przedstartowy, w [°/s] (moduł). Zwraca (p, ok)."""
    ax = ax_channel(tel)
    if ax is None:
        return None, False
    rest = tel.time < (t_ign - 0.5)
    base = np.nanmean(ax[rest]) if np.any(rest) else 0.0
    p = np.abs(ax - base)
    return p, True


def detect_burnout(tel, t_ign):
    """
    Czas wypalenia z ciśnienia w komorze: ostatnia chwila, gdy ciśnienie > 10%
    piku rozruchowego. Fallback: pik spinu po zapłonie.
    """
    t = tel.time
    pc = tel.press_cham
    if pc is not None and np.any(np.isfinite(pc)):
        boost = (t > t_ign) & (t < t_ign + 6.0) & np.isfinite(pc)
        if np.any(boost):
            pmax = np.nanmax(pc[boost])
            if pmax > 1.0:                       # mamy sensowny sygnał ciśnienia
                thr = 0.1 * pmax
                idx = np.where(boost & (pc > thr))[0]
                if len(idx):
                    return float(t[idx[-1]])
    return None


def coast_decay_tau(t, p, t_bo, t_end):
    """
    Stała czasowa zaniku spinu w wybiegu: dopasowanie ln(p) ~ liniowe w czasie
    na oknie [t_bo+0.3, min(t_bo+6, t_end)]. Zwraca (tau_s, p0, r2) lub NaN.
    """
    t0 = t_bo + 0.3
    t1 = min(t_bo + 6.0, t_end)
    win = (t > t0) & (t < t1) & (p > 50) & np.isfinite(p)
    if np.sum(win) < 20:
        return np.nan, np.nan, np.nan
    tt = t[win] - t0
    y = np.log(p[win])
    A = np.vstack([tt, np.ones_like(tt)]).T
    (slope, inter), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, inter])
    ss = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
    tau = -1.0 / slope if slope < 0 else np.nan
    return tau, float(np.exp(inter)), float(ss)


def mag_spin_check(tel, t0, t1):
    """
    Niezależna kontrola spinu z magnetometru: dominująca częstotliwość (pik FFT
    o największej prominencji) w oknie [t0,t1]. Zwraca (deg/s, kanał, prom) lub
    (nan, None, 0). Dla mag 1-osiowego ostrożnie (łapie koning przy małym cant).
    """
    fs = 1.0 / tel.dt_mean if tel.dt_mean else 0.0
    if fs <= 0:
        return np.nan, None, 0.0
    m = (tel.time > t0) & (tel.time < t1)
    chs = [c for c in tel.raw_columns if c.startswith("mmag_")]
    one_axis = False
    if not chs and tel.raw_columns.get("mag_y") is not None:
        chs = ["mag_y"]
        one_axis = True
    best = (np.nan, None, 0.0)
    for ch in chs:
        x = tel.raw_columns[ch]
        xm = x[m]
        xm = xm[np.isfinite(xm)]
        if len(xm) < 64:
            continue
        xm = xm - np.mean(xm)
        w = np.hanning(len(xm))
        S = np.abs(np.fft.rfft(xm * w))
        f = np.fft.rfftfreq(len(xm), 1.0 / fs)
        band = (f > 1.0) & (f < 40.0)
        if not np.any(band):
            continue
        k = np.argmax(S[band])
        prom = S[band][k] / (np.median(S[band]) + 1e-9)
        if prom > best[2]:
            best = (360.0 * f[band][k], ch + (" (1-oś!)" if one_axis else ""), prom)
    return best


# --------------------------------------------------------------------------
def reconstruct_speed(tel, t_ign, t_end, fno):
    """
    Prędkość i wysokość z rekonstrukcji IMU (nie z GPS — ten zaniża V w fazie
    napędowej). Zwraca (time, speed, alt) na siatce rekonstrukcji lub None.
    """
    cfg = load_flight_config(fno)
    az = cfg["azimuth"] if cfg else 90.0
    el = cfg["elevation"] if cfg else 45.0
    try:
        cal = calibrate(tel, t_rest_end=-1.0)
        q0 = initial_orientation(cal, az, el)
        traj = reconstruct(tel, cal, q0, t_start=t_ign, t_end=t_end)
        return traj.time, traj.speed, traj.alt
    except Exception:
        return None


def analyze_flight(fno, cant, nose, verbose=True):
    """Pełna analiza spinu jednego lotu. Zwraca dict metryk (NaN gdy brak)."""
    fname = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    rec = dict(fno=fno, cant=cant, nose=nose, source=None, V_src=None,
               t_ign=np.nan, t_bo=np.nan, p_peak=np.nan, V_bo=np.nan,
               mach=np.nan, pd2V=np.nan, tau_coast=np.nan, tau_r2=np.nan,
               mag_spin=np.nan, mag_ch=None)
    if not fname.exists():
        if verbose:
            print(f"  [BRAK] lot {fno}")
        return rec

    tel = parse_telemetry(fname, verbose=False)
    t = tel.time
    t_ign = detect_ignition(tel)
    rec["t_ign"] = float(t_ign)

    p, ok = extract_spin(tel, t_ign)
    rec["source"] = ("RXO" if tel.raw_columns.get("gyro_ax") is None
                     else "AX") if ok else "mag"

    # okno wybiegu / apogeum
    alt_gps = tel.alt_onboard
    if alt_gps is not None and np.any(np.isfinite(alt_gps)):
        t_apo = float(t[int(np.nanargmax(alt_gps))])
    else:
        t_apo = float(t[-1])

    # Prędkość/wysokość z rekonstrukcji IMU (wiarygodne w fazie napędowej)
    V = Valt = None
    rc = reconstruct_speed(tel, t_ign, t_apo, fno) if ok else None
    if rc is not None:
        tr, sp, al = rc
        V = np.interp(t, tr, sp, left=np.nan, right=np.nan)
        Valt = np.interp(t, tr, al, left=np.nan, right=np.nan)
        rec["V_src"] = "IMU"
    else:
        V = total_speed(tel)
        Valt = alt_gps
        rec["V_src"] = "GPS"

    t_bo = detect_burnout(tel, t_ign)
    if ok:
        ps = _smooth(p, 25)
        if t_bo is None:
            seg = (t > t_ign + 0.6) & (t < t_ign + 6.0)
            t_bo = float(t[np.nanargmax(np.where(seg, ps, -np.inf))])
        rec["t_bo"] = t_bo

        # p_peak: aerodynamiczny pik narastania, Z POMINIĘCIEM szpilki zapłonowej
        # (pierwsze ~0.6 s po zapłonie to transjent, nie spin).
        near = (t > t_ign + 0.6) & (t < t_bo + 0.5)
        rec["p_peak"] = float(np.nanmax(ps[near])) if np.any(near) else np.nan

        # p·d/2V przy równowadze: okno okołowypaleniowe (późny ciąg), V z IMU
        win = ((t > t_bo - 1.0) & (t < t_bo + 0.3) & (V > 20)
               & np.isfinite(p) & np.isfinite(V))
        if np.sum(win) >= 8:
            p_rad = ps[win] * np.pi / 180.0
            pd2v = p_rad * (D_REF / 2.0) / np.maximum(V[win], 1e-3)
            rec["pd2V"] = float(np.nanmedian(pd2v))
            rec["V_bo"] = float(np.nanmedian(V[win]))
            if Valt is not None and np.any(np.isfinite(Valt[win])):
                rec["mach"] = float(np.nanmedian(
                    V[win] / isa_sound_speed(Valt[win])))

        # tłumienie: zanik w wybiegu
        tau, p0, r2 = coast_decay_tau(t, ps, t_bo, t_apo)
        rec["tau_coast"], rec["tau_r2"] = tau, r2

    # kontrola magnetometrem (okno okołowypaleniowe)
    if not np.isnan(rec["t_bo"]):
        ms, mch, _ = mag_spin_check(tel, rec["t_bo"] - 1.0,
                                    min(rec["t_bo"] + 0.5, t_apo))
        rec["mag_spin"], rec["mag_ch"] = ms, mch

    if verbose:
        print(f"  lot {fno:>2} cant={cant:>3} {nose:<5} src={rec['source']:<3} "
              f"p_peak={rec['p_peak']:7.0f}°/s  V={rec['V_bo']:5.0f}  "
              f"M={rec['mach']:4.2f}  p·d/2V={rec['pd2V']:.4f}  "
              f"τ={rec['tau_coast']:4.1f}s  mag={rec['mag_spin']:7.0f}°/s")
    return rec


def fit_cant(records, nose="ostra"):
    """Dopasowanie p·d/2V = slope*cant + a dla danego nosa. Zwraca dict."""
    rows = [r for r in records if r["nose"] == nose and np.isfinite(r["pd2V"])]
    if len(rows) < 2:
        return None
    cant = np.array([r["cant"] for r in rows])
    y = np.array([r["pd2V"] for r in rows])
    A = np.vstack([cant, np.ones_like(cant)]).T
    (slope, inter), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, inter])
    ss = 1.0 - np.sum((y - yhat) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)
    return dict(nose=nose, slope=float(slope), intercept=float(inter),
                r2=float(ss), n=len(rows),
                cants=sorted(set(cant.tolist())))


def save_plots(records, fit_ostra):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plots] pominięto (brak matplotlib: {e})")
        return
    RESULTS.mkdir(exist_ok=True)
    # p·d/2V vs cant
    fig, ax = plt.subplots(figsize=(7, 5))
    for nose, mk, col in (("ostra", "o", "tab:blue"), ("tępa", "s", "tab:red")):
        rr = [r for r in records if r["nose"] == nose and np.isfinite(r["pd2V"])]
        if rr:
            ax.scatter([r["cant"] for r in rr], [r["pd2V"] for r in rr],
                       marker=mk, color=col, label=nose, zorder=3)
            for r in rr:
                ax.annotate(str(r["fno"]), (r["cant"], r["pd2V"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
    if fit_ostra:
        xs = np.linspace(0, max(1.6, *[r["cant"] for r in records]) , 50)
        ax.plot(xs, fit_ostra["slope"] * xs + fit_ostra["intercept"], "--",
                color="tab:blue", alpha=0.7,
                label=f"fit ostra: a={fit_ostra['intercept']:.4f}, "
                      f"nachyl={fit_ostra['slope']:.4f}, R²={fit_ostra['r2']:.2f}")
    ax.set_xlabel("kąt zaklinowania cant [°]")
    ax.set_ylabel("p·d/2V (równowaga okołowypaleniowa)")
    ax.set_title("Bezwymiarowy spin vs cant")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    out = RESULTS / "spin_vs_cant.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  [plots] {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flights", nargs="*", type=int, help="numery lotów")
    ap.add_argument("--nose", choices=["ostra", "tępa"], help="tylko dany nos")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    # loty + konfiguracja z configs.txt
    cfg = {}
    for row in read_config_rows("configs.txt"):
        try:
            fno = int(str(row.get("flight no", "")).strip())
        except (ValueError, TypeError):
            continue
        head = str(row.get("head configuration", "")).strip().lower()
        nose = "tępa" if head.startswith("t") else "ostra"
        try:
            cant = float(str(row.get("cant angle", "")).strip())
        except ValueError:
            cant = np.nan
        analyze = str(row.get("to analyze?", "")).strip().lower() == "yes"
        cfg[fno] = dict(cant=cant, nose=nose, analyze=analyze)

    if args.flights:
        flights = args.flights
    else:
        flights = [f for f, c in cfg.items() if c["analyze"]]
    if args.nose:
        flights = [f for f in flights if cfg.get(f, {}).get("nose") == args.nose]

    print(f"Loty: {flights}\n")
    print("Ekstrakcja spinu (AX/RXO jako °/s, magnetometr = kontrola):")
    records = []
    for fno in flights:
        c = cfg.get(fno, dict(cant=np.nan, nose="ostra"))
        records.append(analyze_flight(fno, c["cant"], c["nose"]))

    fit_o = fit_cant(records, "ostra")
    print("\n" + "=" * 60)
    print("Dopasowanie p·d/2V = nachylenie·cant + a  (ostra)")
    print("=" * 60)
    if fit_o:
        print(f"  nachylenie (≈ −Clδ/Clp): {fit_o['slope']:.4f}")
        print(f"  wyraz wolny a (niesymetria): {fit_o['intercept']:.4f}")
        print(f"  R²={fit_o['r2']:.3f}  n={fit_o['n']}  cant={fit_o['cants']}")
    else:
        print("  za mało punktów")

    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "spin_cant_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    print(f"\nZapisano: {csv_path}")

    if not args.no_plots:
        save_plots(records, fit_o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
