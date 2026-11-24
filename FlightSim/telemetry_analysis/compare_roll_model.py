"""
compare_roll_model.py
=====================
Weryfikacja poprawki roll z punktu 1: czy przeskalowanie tłumienia (Clp_scale)
i napędu (CLL_scale) sprawia, że MODELOWY przebieg prędkości obrotowej pokrywa
się ze ZMIERZONYM.

Metoda
------
Całkujemy równanie dynamiki rolki napędzone rzeczywistą obwiednią lotu
(V(t), ρ(t), kąt zaklinowania), z tabelami DATCOM CLL(cant,Mach) i Clp(Mach):

    Ixx·dp/dt = q·S·d·[ CLL(cant,M)·CLL_scale  +  Clp(M)·Clp_scale·(p·d/2V) ]

(q = ½ρV²).  Napęd buduje spin, tłumienie go hamuje; równowaga ∝ V, więc spin
narasta na ciągu i opada w wybiegu wraz z V. Dwie krzywe:
  - PRZED poprawką: CLL_scale = Clp_scale = 1,
  - PO poprawce:    CLL_scale, Clp_scale z analizy (domyślnie 1.0 i 0.25).

Obwiednia V(t), wysokość: z rekonstrukcji IMU (prędkość wiarygodna i na ciągu,
i w wybiegu). Spin zmierzony: kanał AX/RXO w °/s.

Użycie
------
    python telemetry_analysis/compare_roll_model.py 20
    python telemetry_analysis/compare_roll_model.py            # loty 'yes'
    python telemetry_analysis/compare_roll_model.py 22 --clp-scale 0.25 --cll-scale 1.0
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from telemetry_parser import (parse_telemetry, resolve_data_file,
                              read_config_rows, load_flight_config)
from imu_reconstruction import (detect_ignition, calibrate,
                                initial_orientation, reconstruct)
from analyze_spin_cant import (load_model_roll, MODEL_RUNS, isa_rho,
                               isa_sound_speed, extract_spin, detect_burnout,
                               _smooth, D_REF, S_REF, RESULTS)

IXX_FULL = 0.0032      # roll [kg·m²] przed wypaleniem (yaml)
IXX_EMPTY = 0.0028     # roll [kg·m²] po wypaleniu (yaml)


def model_roll_interpolators():
    """CLL(cant,Mach) i Clp(Mach) przy alpha=0, wartości bezwzględne. Zwraca fn."""
    tabs, stale = load_model_roll()
    if stale:
        print("  !!! UWAGA: tabele DATCOM cant przestarzałe (zły LREF) — krzywa "
              "modelu NIEWIARYGODNA (do regeneracji biegów cant).")
    if len(tabs) < 2:
        return None, None, None
    t0 = tabs[0.0]
    i0 = int(np.argmin(np.abs(t0.alpha_table)))
    mach = t0.mach_table
    cants = np.array(sorted(tabs))
    # CLL: siatka |CLL|(cant, mach)
    cll_grid = np.array([np.abs(tabs[c].CLL_table[i0, :]) for c in cants])  # (ncant, nmach)
    clp_vec = np.abs(t0.Clp_table[i0, :])                                   # (nmach,)

    def cll_of(cant, m):
        col = np.array([np.interp(m, mach, cll_grid[j, :]) for j in range(len(cants))])
        return float(np.interp(cant, cants, col))

    def clp_of(m):
        return float(np.interp(m, mach, clp_vec))

    return cll_of, clp_of, (float(mach.min()), float(mach.max()))


def flight_envelope(tel, t_ign, t_end, fno):
    """V(t), alt(t) z rekonstrukcji IMU na siatce tel.time (NaN poza oknem)."""
    cfg = load_flight_config(fno)
    az = cfg["azimuth"] if cfg else 90.0
    el = cfg["elevation"] if cfg else 45.0
    try:
        cal = calibrate(tel, t_rest_end=-1.0)
        q0 = initial_orientation(cal, az, el)
        traj = reconstruct(tel, cal, q0, t_start=t_ign, t_end=t_end)
        V = np.interp(tel.time, traj.time, traj.speed, left=np.nan, right=np.nan)
        alt = np.interp(tel.time, traj.time, traj.alt, left=np.nan, right=np.nan)
        return V, alt
    except Exception as e:
        print(f"  [env] rekonstrukcja nieudana ({e}) — GPS")
        vN = tel.raw_columns.get("vel_north_ms")
        if vN is not None:
            V = np.sqrt(np.nan_to_num(vN) ** 2
                        + np.nan_to_num(tel.raw_columns.get("vel_east_ms")) ** 2
                        + np.nan_to_num(tel.raw_columns.get("vel_down_ms")) ** 2)
        else:
            V = tel.raw_columns.get("vel", tel.vel_onboard)
        return V, tel.alt_onboard


def integrate_roll(t, V, alt, cant, t_ign, t_bo, t_end, cll_of, clp_of,
                   cll_scale, clp_scale):
    """Całkuje p(t) [rad/s] równania rolki przy danej obwiedni i skalach."""
    p = np.zeros_like(t)
    i0 = int(np.searchsorted(t, t_ign))
    iend = int(np.searchsorted(t, t_end))
    for i in range(max(i0, 1), min(iend, len(t))):
        dt = t[i] - t[i - 1]
        Vi = V[i]
        hi = alt[i]
        if not np.isfinite(Vi) or Vi < 15 or not np.isfinite(hi):
            p[i] = p[i - 1]
            continue
        # Ixx: pełne -> puste liniowo na ciągu, potem puste
        if t[i] <= t_bo:
            frac = np.clip((t[i] - t_ign) / max(t_bo - t_ign, 1e-3), 0, 1)
            Ixx = IXX_FULL + (IXX_EMPTY - IXX_FULL) * frac
        else:
            Ixx = IXX_EMPTY
        rho = isa_rho(hi)
        M = Vi / isa_sound_speed(hi)
        q = 0.5 * rho * Vi ** 2
        cll = cll_of(cant, M) * cll_scale
        clp = clp_of(M) * clp_scale
        pd2v = p[i - 1] * (D_REF / 2.0) / Vi
        dpdt = (q * S_REF * D_REF / Ixx) * (cll - clp * pd2v)   # napęd − tłumienie
        p[i] = max(p[i - 1] + dpdt * dt, 0.0)
    return p


def analyze(fno, cant, nose, cll_scale, clp_scale, cll_of, clp_of, verbose=True):
    fname = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    if not fname.exists():
        print(f"  [BRAK] lot {fno}")
        return None
    tel = parse_telemetry(fname, verbose=False)
    t = tel.time
    t_ign = detect_ignition(tel)
    p_meas, ok = extract_spin(tel, t_ign)
    if not ok:
        print(f"  lot {fno}: brak kanału AX/RXO — pomijam")
        return None
    p_meas = _smooth(p_meas, 25)      # usuń szpilkę zapłonową / szum
    alt_gps = tel.alt_onboard
    t_apo = float(t[int(np.nanargmax(alt_gps))]) if (
        alt_gps is not None and np.any(np.isfinite(alt_gps))) else float(t[-1])
    t_bo = detect_burnout(tel, t_ign) or (t_ign + 2.0)

    V, alt = flight_envelope(tel, t_ign, t_apo, fno)

    p_base = integrate_roll(t, V, alt, cant, t_ign, t_bo, t_apo,
                            cll_of, clp_of, 1.0, 1.0)
    p_corr = integrate_roll(t, V, alt, cant, t_ign, t_bo, t_apo,
                            cll_of, clp_of, cll_scale, clp_scale)
    # rad/s -> deg/s
    r2d = 180.0 / np.pi
    return dict(fno=fno, cant=cant, nose=nose, t=t, t_ign=t_ign, t_apo=t_apo,
                p_meas=p_meas, p_base=p_base * r2d, p_corr=p_corr * r2d)


def plot_flight(res, clp_scale, cll_scale):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] brak matplotlib: {e}")
        return
    RESULTS.mkdir(exist_ok=True)
    t = res["t"]
    win = (t > res["t_ign"] - 1.0) & (t < res["t_apo"] + 1.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t[win], res["p_meas"][win], color="k", lw=1.2, label="pomiar (AX/RXO)")
    ax.plot(t[win], res["p_base"][win], "--", color="tab:red", lw=1.5,
            label="model PRZED (Clp×1)")
    ax.plot(t[win], res["p_corr"][win], "-", color="tab:green", lw=1.5,
            label=f"model PO (Clp×{clp_scale:g}, CLL×{cll_scale:g})")
    ax.axvline(res["t_ign"], color="gray", ls=":", lw=0.8)
    ax.set_xlabel("czas [s]"); ax.set_ylabel("prędkość obrotowa roll [°/s]")
    ax.set_title(f"Lot {res['fno']} (cant {res['cant']}°, {res['nose']}) — "
                 f"roll: pomiar vs model")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    out = RESULTS / f"roll_model_vs_test_flight_{res['fno']}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  [plot] {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flights", nargs="*", type=int)
    ap.add_argument("--clp-scale", type=float, default=0.25)
    ap.add_argument("--cll-scale", type=float, default=1.0)
    args = ap.parse_args(argv)

    cll_of, clp_of, mrange = model_roll_interpolators()
    if cll_of is None:
        print("Brak tabel DATCOM roll — nie można porównać.")
        return 1

    cfg = {}
    for row in read_config_rows("configs.txt"):
        try:
            fno = int(str(row.get("flight no", "")).strip())
        except (ValueError, TypeError):
            continue
        head = str(row.get("head configuration", "")).strip().lower()
        try:
            cant = float(str(row.get("cant angle", "")).strip())
        except ValueError:
            cant = np.nan
        cfg[fno] = dict(cant=cant, nose="tępa" if head.startswith("t") else "ostra",
                        analyze=str(row.get("to analyze?", "")).strip().lower() == "yes")

    flights = args.flights or [f for f, c in cfg.items() if c["analyze"]]
    print(f"Clp_scale={args.clp_scale}  CLL_scale={args.cll_scale}  "
          f"(model Mach {mrange[0]:.1f}–{mrange[1]:.1f})\n")
    for fno in flights:
        c = cfg.get(fno, dict(cant=np.nan, nose="ostra"))
        if not np.isfinite(c["cant"]):
            continue
        res = analyze(fno, c["cant"], c["nose"], args.cll_scale, args.clp_scale,
                      cll_of, clp_of)
        if res:
            # proste miary dopasowania w oknie lotu
            t = res["t"]; w = (t > res["t_ign"]) & (t < res["t_apo"])
            pk_meas = np.nanmax(res["p_meas"][w])
            pk_base = np.nanmax(res["p_base"][w])
            pk_corr = np.nanmax(res["p_corr"][w])
            print(f"  lot {fno} cant={c['cant']}: pik  pomiar={pk_meas:.0f}  "
                  f"przed={pk_base:.0f}  po={pk_corr:.0f} °/s")
            plot_flight(res, args.clp_scale, args.cll_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
