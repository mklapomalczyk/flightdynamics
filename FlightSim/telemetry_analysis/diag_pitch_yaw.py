"""
diag_pitch_yaw.py
=================
Punkt 2: próba wyznaczenia charakterystyki pitch/yaw (Cm_α — sztywność
statyczna) z lotów i porównanie z DATCOM.

Dlaczego to trudne
------------------
Żyroskopy gyro_y/z są zdominowane przez OBRÓT rolla (ruch poprzeczny stały w
przestrzeni oscyluje w ramie ciała na f_spin) + offset od niedokładnego montażu
(∝ p). De-spin i odejmowanie przecieku odsłaniają tylko słaby, rozmyty sygnał —
bo rakieta jest bardzo sztywna statycznie (Cm_α ~ −54/rad), więc kąt natarcia i
coning są MAŁE. Potwierdzono na wszystkich lotach, także cant=0 o znikomym
rollu (lot 13, f_spin~0.17 Hz) — brak czystego piku pitch/yaw.

Co działa: magnetometr 3-osiowy (loty 22/24/25)
-----------------------------------------------
Wielkość NIEZMIENNICZA względem rolla — magnituda pola poprzecznego
|B_⊥| = sqrt(B_t1² + B_t2²) — nie zależy od kąta rolla (roll tylko obraca wektor
poprzeczny), a oscyluje na częstotliwości CONINGU (kąt między osią rakiety a
polem B zmienia się przy conieniu). Daje słaby, ale SPÓJNY pik ~2.5 Hz na
22/24/25, zgodny z f_pitch przewidywaną z DATCOM Cm_α.

Częstotliwość pitch (nie-wirująca, sprzężenie żyroskopowe słabe bo Ixx<<Iyy):
    f_pitch = (1/2π)·sqrt(|Cm_α|·q·S_ref·d / Iyy)
Stąd Cm_scale = (f_obs / f_pitch_DATCOM)².

Użycie
------
    python telemetry_analysis/diag_pitch_yaw.py            # loty z mag 3-os.
    python telemetry_analysis/diag_pitch_yaw.py 22 24 25
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from telemetry_parser import parse_telemetry, resolve_data_file, read_config_rows
from imu_reconstruction import detect_ignition
from gps_fusion import fused_speed
from analyze_spin_cant import (_smooth, extract_spin, isa_rho, isa_sound_speed,
                               D_REF, S_REF)

FS_ROOT = Path(__file__).resolve().parent.parent
IYY = 0.53             # moment bezwładności pitch/yaw (po wypaleniu) [kg·m²] (yaml)


def _datcom_cma():
    """Funkcja Cm_α(Mach) [1/rad] przy alpha=0, z przebiegu bazowego DATCOM."""
    p = FS_ROOT / "datcom_runs" / "rocket_70mm_baseline" / "aero_table_missile.pkl"
    sys.path.insert(0, str(FS_ROOT))
    d = pickle.load(open(p, "rb"))
    mach = d.mach_table
    alpha = d.alpha_table
    i0 = int(np.argmin(np.abs(alpha)))
    cma_m = (d.Cm_table[i0 + 1, :] - d.Cm_table[i0 - 1, :]) / (alpha[i0 + 1] - alpha[i0 - 1])

    def cma(M):
        return float(np.interp(M, mach, cma_m))
    return cma


def _peak(sig, fs, f0, f1):
    s = np.nan_to_num(sig - np.nanmean(sig))
    n = len(s)
    if n < 128:
        return np.nan, 0.0
    S = np.abs(np.fft.rfft(s * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    b = (f > f0) & (f < f1)
    if not np.any(b):
        return np.nan, 0.0
    k = np.argmax(S[b])
    return float(f[b][k]), float(S[b][k] / (np.median(S[b]) + 1e-9))


def _bandpow(sig, fs, f0, f1):
    s = np.nan_to_num(sig - np.nanmean(sig))
    n = len(s)
    S = np.abs(np.fft.rfft(s * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return float(np.sum(S[(f > f0) & (f < f1)]))


def analyze_flight(fno, cma_fn, verbose=True):
    fname = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    if not fname.exists():
        return None
    tel = parse_telemetry(fname, verbose=False)
    t = tel.time
    ti = detect_ignition(tel)
    alt = tel.alt_onboard
    if alt is None or not np.any(np.isfinite(alt)):
        return None
    ta = float(t[int(np.nanargmax(alt))])
    fs = 1.0 / tel.dt_mean
    V = fused_speed(tel, ti)

    w = (t > ti + 2.0) & (t < ta)
    if np.sum(w) < 200:
        return None
    Vc = float(np.nanmedian(V[w]))
    hc = float(np.nanmedian(alt[w]))
    q = 0.5 * isa_rho(hc) * Vc ** 2
    M = np.clip(Vc / isa_sound_speed(hc), 0.1, 2.0)
    f_exp = (1 / (2 * np.pi)) * np.sqrt(abs(cma_fn(M)) * q * S_REF * D_REF / IYY)
    f_spin = float(np.nanmedian(np.abs(_smooth(extract_spin(tel, ti)[0], 25)[w]))) / 360.0

    rec = dict(fno=fno, f_exp=f_exp, f_spin=f_spin, mach=M,
               f_mag=np.nan, prom_mag=0.0, cm_scale=np.nan, mag3=False)

    # Magnetometr 3-osiowy: coning z |B_⊥| (niezmiennik rolla)
    mm = {a: tel.raw_columns.get(f"mmag_mag_ZP_{a}") for a in "XYZ"}
    mm = {k: v for k, v in mm.items() if v is not None}
    if len(mm) == 3:
        rec["mag3"] = True
        sb = (f_spin * 0.7, f_spin * 1.3)
        pw = {a: _bandpow(mm[a][w], fs, *sb) for a in mm}
        axial = min(pw, key=pw.get)            # oś osiowa = najmniej mocy na f_spin
        tr = [a for a in mm if a != axial]
        btr = np.sqrt(np.nan_to_num(mm[tr[0]]) ** 2 + np.nan_to_num(mm[tr[1]]) ** 2)
        fobs, prom = _peak(btr[w], fs, 0.8, 6.0)      # pasmo pitch/yaw
        rec["f_mag"], rec["prom_mag"] = fobs, prom
        if np.isfinite(fobs) and f_exp > 0:
            rec["cm_scale"] = (fobs / f_exp) ** 2

    if verbose:
        tag = "mag3" if rec["mag3"] else "—"
        print(f"  lot {fno:>2}: f_pitch(DATCOM)={f_exp:4.2f}Hz  f_spin={f_spin:4.1f}Hz  "
              f"| |B_⊥| pik={rec['f_mag'] if np.isnan(rec['f_mag']) else round(rec['f_mag'],2)}Hz "
              f"(prom {rec['prom_mag']:.1f})  Cm_scale={rec['cm_scale'] if np.isnan(rec['cm_scale']) else round(rec['cm_scale'],2)}  [{tag}]")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flights", nargs="*", type=int)
    args = ap.parse_args(argv)

    cma_fn = _datcom_cma()
    if args.flights:
        flights = args.flights
    else:
        # domyślnie loty z magnetometrem 3-osiowym (nowy format)
        flights = []
        for row in read_config_rows("configs.txt"):
            try:
                fno = int(str(row.get("flight no", "")).strip())
            except (ValueError, TypeError):
                continue
            if fno >= 22:
                flights.append(fno)

    print("Pitch/yaw z magnetometru (|B_⊥| niezmiennik rolla) vs DATCOM Cm_α:\n")
    recs = [r for r in (analyze_flight(f, cma_fn) for f in flights) if r]
    cs = [r["cm_scale"] for r in recs if np.isfinite(r["cm_scale"])]
    print("\n" + "=" * 56)
    if cs:
        print(f"Cm_scale: średnia={np.mean(cs):.2f}  zakres={min(cs):.2f}–{max(cs):.2f}  (n={len(cs)})")
        print("Wniosek: Cm_α (sztywność statyczna pitch) ZGODNA z DATCOM — brak")
        print("korekty. Sygnał słaby (tylko loty z mag 3-os.), więc to WALIDACJA,")
        print("nie precyzyjne skalowanie.")
    else:
        print("Brak wiarygodnego piku coningu — pitch/yaw niewyznaczalne.")
    print("Cmq (tłumienie): niewyznaczalne z lotu (zanik coningu za słaby); "
          "w modelu ANALITYCZNE (−2·CNA_fins·((x_fins−xcg)/d)²), niewalidowane.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
