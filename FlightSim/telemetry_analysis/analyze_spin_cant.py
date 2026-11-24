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

Wniosek (patrz compare_roll_model.py)
-------------------------------------
Identyfikowalny jest TYLKO stosunek CLL/Clp (model zaniża go ~4×). Czas
narastania (ostry pik przy wypaleniu) wyklucza obniżanie Clp, więc korekta =
NAPĘD: CLL_scale ≈ 4, Clp_scale ≈ 1. τ z wybiegu NIE jest czystym tłumieniem
(stateczniki napędzają roll także w wybiegu) i służy tu tylko jako diagnostyka.

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
from imu_reconstruction import detect_ignition

G0 = 9.80665

D_REF = 0.070                      # średnica odniesienia [m]
S_REF = np.pi * D_REF ** 2 / 4.0   # pole odniesienia [m²]
IXX_COAST = 0.0028                 # moment bezwładności roll (po wypaleniu) [kg·m²] (yaml)
R_AIR = 287.05
RESULTS = get_data_dir() / "results"

# Przebiegi DATCOM dla modelu roll vs cant
MODEL_RUNS = {0.0: "cant0p000", 0.6: "cant0p600", 1.6: "cant1p600"}


# --------------------------------------------------------------------------
def isa_sound_speed(h_m):
    """Prędkość dźwięku wg ISA (troposfera) [m/s]. h w [m ASL]."""
    T = 288.15 - 0.0065 * np.clip(h_m, 0.0, 20000.0)
    return np.sqrt(1.4 * R_AIR * T)


def isa_rho(h_m):
    """Gęstość powietrza wg ISA [kg/m³]."""
    T = 288.15 - 0.0065 * np.clip(h_m, 0.0, 20000.0)
    p = 101325.0 * (T / 288.15) ** 5.2561
    return p / (R_AIR * T)


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
def axial_speed(tel, t_ign):
    """
    Prędkość osiowa z integracji akcelerometru X (°/s → m/s).

    Akcelerometr mierzy siłę właściwą; w spoczynku acc_x = rzut grawitacji na oś
    (= acc_x_rest). Przy ~stałej orientacji we WCZESNEJ fazie (napęd/wczesny
    wybieg) inercjalne przyspieszenie osiowe = (acc_x − acc_x_rest)·g0, więc
    v_osiowa(t) = ∫ od zapłonu. Wiarygodne tylko na początku lotu — używamy go
    w oknie okołowypaleniowym (tam GPS zaniża V).
    """
    acc = tel.acc_x
    if acc is None:
        return None
    rest = tel.time < (t_ign - 0.5)
    bias = np.nanmean(acc[rest]) if np.any(rest) else 0.0
    a = np.nan_to_num((acc - bias) * G0)
    a[tel.time < t_ign] = 0.0
    dt = np.diff(tel.time, prepend=tel.time[0])
    v = np.cumsum(a * dt)
    v[tel.time < t_ign] = 0.0
    return v


def analyze_flight(fno, cant, nose, verbose=True):
    """Pełna analiza spinu jednego lotu. Zwraca dict metryk (NaN gdy brak)."""
    fname = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    rec = dict(fno=fno, cant=cant, nose=nose, source=None, V_src=None,
               t_ign=np.nan, t_bo=np.nan, p_peak=np.nan, V_bo=np.nan,
               mach=np.nan, pd2V=np.nan, tau_coast=np.nan, tau_r2=np.nan,
               mach_coast=np.nan, clp_meas=np.nan,
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

    # Prędkość osiowa z akcelerometru X (wiarygodna we wczesnej fazie; GPS zaniża
    # V w fazie napędowej). Wysokość do prędkości dźwięku — z GPS, inaczej start.
    V = axial_speed(tel, t_ign)
    if V is not None:
        rec["V_src"] = "accX"
    else:
        V = total_speed(tel)
        rec["V_src"] = "GPS"
    if alt_gps is not None and np.any(np.isfinite(alt_gps)):
        Valt = alt_gps
    else:
        a0 = alt_gps[np.isfinite(alt_gps)] if alt_gps is not None else None
        Valt = np.full_like(t, a0[0] if a0 is not None and len(a0) else 200.0)

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

        # tłumienie: zanik spinu w wybiegu -> stała czasowa τ
        tau, p0, r2 = coast_decay_tau(t, ps, t_bo, t_apo)
        rec["tau_coast"], rec["tau_r2"] = tau, r2

        # zmierzone Clp z zaniku: Ixx·dp/dt = ¼ρV·S·d²·Clp·p  ⇒
        #   Clp = −4·Ixx / (ρ·V·S·d²·τ).  V i wysokość w wybiegu z GPS (tam GPS OK).
        if np.isfinite(tau) and tau > 0:
            Vg = total_speed(tel)
            cwin = (t > t_bo + 0.3) & (t < min(t_bo + 6.0, t_apo)) & np.isfinite(Vg)
            if Vg is not None and np.sum(cwin) > 10:
                Vc = float(np.nanmedian(Vg[cwin]))
                hc = (float(np.nanmedian(alt_gps[cwin]))
                      if alt_gps is not None and np.any(np.isfinite(alt_gps[cwin]))
                      else 500.0)
                if Vc > 20:
                    rho = isa_rho(hc)
                    rec["mach_coast"] = Vc / isa_sound_speed(hc)
                    rec["clp_meas"] = -4.0 * IXX_COAST / (
                        rho * Vc * S_REF * D_REF ** 2 * tau)

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


def load_model_roll():
    """Wczytuje tabele DATCOM (CLL, Clp) dla cant 0/0.6/1.6. Zwraca dict lub {}."""
    import pickle
    base = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(base))               # pkl potrzebuje modułu 'models'
    tabs = {}
    for cant, name in MODEL_RUNS.items():
        p = base / "datcom_runs" / f"rocket_70mm_baseline_{name}" / "aero_table_missile.pkl"
        if p.exists():
            try:
                tabs[cant] = pickle.load(open(p, "rb"))
            except Exception as e:
                print(f"  [model] nie wczytano {name}: {e}")
    return tabs


def model_calibration(records, fit_meas):
    """
    Porównanie z DATCOM i propozycja kalibracji.

    Identyfikowalny jest TYLKO stosunek CLL/Clp (poziom równowagi p·d/2V) — spin
    jest w quasi-równowadze. Czas narastania (ostry pik przy wypaleniu) wyklucza
    OBNIŻANIE Clp (potrzeba szybkiej równowagi), więc niedoszacowanie ~4× trzeba
    złożyć na NAPĘD CLL, nie na tłumienie:

      CLL_scale = p·d/2V(pomiar) / [CLL(cant,M)/Clp(M)](model),  Clp trzymane = model.

    τ z „zaniku" w wybiegu podajemy tylko jako DIAGNOSTYKĘ — nie jest to czyste
    tłumienie (stateczniki napędzają roll także w wybiegu), więc NIE używamy go
    jako Clp_scale (to był błąd wcześniejszej wersji).
    """
    tabs = load_model_roll()
    if len(tabs) < 2 or fit_meas is None:
        return None
    t0 = tabs[0.0]
    i0 = int(np.argmin(np.abs(t0.alpha_table)))     # alpha = 0
    mach = t0.mach_table
    cants_model = sorted(tabs)
    cll_grid = {c: np.abs(tabs[c].CLL_table[i0, :]) for c in cants_model}

    def model_CLL(cant, m):
        col = [np.interp(m, mach, cll_grid[c]) for c in cants_model]
        return float(np.interp(cant, cants_model, col))

    def model_Clp(m):
        return float(np.interp(m, mach, np.abs(t0.Clp_table[i0, :])))

    def model_pd2V(cant, m):
        clp = model_Clp(m)
        return model_CLL(cant, m) / clp if clp > 0 else np.nan

    # CLL_scale per lot (Clp = model): o ile trzeba zwiększyć napęd
    for r in records:
        r["cll_scale"] = np.nan
        if (r["nose"] == "ostra" and r["cant"] > 0
                and np.isfinite(r["pd2V"]) and np.isfinite(r["mach"])):
            mp = model_pd2V(r["cant"], r["mach"])
            if mp and mp > 0:
                r["cll_scale"] = r["pd2V"] / mp
    cs = np.array([r["cll_scale"] for r in records if np.isfinite(r["cll_scale"])])
    cll_scale = float(np.mean(cs)) if len(cs) else np.nan
    cll_std = float(np.std(cs)) if len(cs) else np.nan

    # identyfikowalny czynnik spinu (nachylenie pomiar/model)
    m_rep = float(np.median([r["mach"] for r in records if np.isfinite(r["mach"])]))
    slope_meas = abs(fit_meas["slope"])
    dCLL_dc = float(np.polyfit(np.array(cants_model, float),
                               [model_CLL(c, m_rep) for c in cants_model], 1)[0])
    slope_model = dCLL_dc / model_Clp(m_rep)
    spin_factor = slope_meas / slope_model

    # diagnostyka: efektywne Clp z τ (UWAGA: podtrzymywane napędem — nie Clp_scale)
    clp_m = np.array([abs(r["clp_meas"]) for r in records if np.isfinite(r["clp_meas"])])
    clp_eff = float(np.mean(clp_m)) if len(clp_m) else np.nan

    return dict(m_rep=m_rep, slope_meas=slope_meas, slope_model=slope_model,
                spin_factor=spin_factor, cll_scale=cll_scale, cll_std=cll_std,
                n_cll=len(cs), clp_model=model_Clp(m_rep), clp_eff=clp_eff)


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

    cal = model_calibration(records, fit_o)
    if cal:
        print("\n" + "=" * 60)
        print("Porównanie z DATCOM + propozycja kalibracji (ostra)")
        print("=" * 60)
        print(f"  Mach reprezentatywny (wybieg): {cal['m_rep']:.2f}")
        print(f"  nachylenie p·d/2V vs cant:  pomiar={cal['slope_meas']:.4f}  "
              f"model={cal['slope_model']:.4f}  /deg")
        print(f"  -> stosunek CLL/Clp niedoszacowany x{cal['spin_factor']:.1f} "
              f"(to jest identyfikowalne)")
        print(f"  Podział (z czasu narastania — pik przy wypaleniu, szybka "
              f"równowaga):")
        print(f"    napęd  CLL_scale ≈ {cal['cll_scale']:.1f} ± {cal['cll_std']:.1f}  "
              f"(n={cal['n_cll']})")
        print(f"    tłumienie  Clp_scale ≈ 1  (Clp ~ model; NIE obniżać)")
        print(f"  [diagnostyka] efektywne Clp z τ = {cal['clp_eff']:.2f} vs model "
              f"{cal['clp_model']:.2f} — zaniżone, bo w wybiegu spin jest "
              f"podtrzymywany napędem (to NIE jest Clp_scale)")
        print(f"\n  PROPOZYCJA (zakres: Mach ~0.4–1.0, cant 0–1.6°, ostra):")
        print(f"    CLL_scale ≈ {cal['cll_scale']:.1f}   Clp_scale ≈ 1")
        print(f"    (zweryfikowane czasowo w compare_roll_model.py; "
              f"do finalnej weryfikacji po AMS/RMS)")

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
