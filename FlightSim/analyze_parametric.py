"""
analyze_parametric.py
=====================
Porównanie konfiguracji małej rakiety 36mm.

Sweepy:
  A. Silnik — stały impuls całkowity, różny czas pracy
  B. Stateczniki — chord x count (wymaga DATCOM dla każdej konfiguracji)
  C. Cant angle — baseline (0°) vs 0.5°

Wykresy:
  simulation_baseline.png     — 4 wykresy dla baseline
  sweep_motor.png             — zasięg/wysokość/Vmax vs czas pracy silnika
  sweep_fins.png              — zasięg/wysokość/Vmax, linie wg count, markery wg chord
  sweep_cant.png              — porównanie 0° vs 0.5° z odchyleniem bocznym

Uruchomienie z katalogu FlightSim:
    python analyze_parametric.py
"""

import sys, csv, tempfile, os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Folder na wyniki
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

from aero import get_aero_model
from datcom_io.config_reader import load_config

# Import funkcji wizualizacji
try:
    from visualize_rocket import draw_side_view, draw_rear_view
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False
    print("[WARN] visualize_rocket.py niedostepny — wizualizacje pominiete")

def visualize_cfg(cfg, label, out_path):
    """Rysuje geometrie rakiety i zapisuje do pliku."""
    if not HAS_VISUALIZER:
        return
    import matplotlib.pyplot as _plt
    fig, axes = _plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [3, 1]})
    n_fins  = int(cfg.fins[0].count) if cfg.fins else 0
    chord   = cfg.fins[0].root_chord * 1000 if cfg.fins else 0
    cant    = cfg.fins[0].cant_angle if cfg.fins else 0
    fig.suptitle(
        f"Geometria — {label}\n"
        f"L={cfg.body.length*1000:.0f}mm  d={cfg.body.diameter*1000:.0f}mm  "
        f"xcg_ref={cfg.mass.xcg_ref*1000:.0f}mm  "
        f"pletwy: {n_fins}x  chord={chord:.0f}mm  cant={cant}°",
        fontsize=10, fontweight="bold"
    )
    draw_side_view(axes[0], cfg)
    draw_rear_view(axes[1], cfg)
    _plt.tight_layout()
    _plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    _plt.close()
    print(f"   Zapisano wizualizacje: {out_path}")
from datcom_io.rocket_builder import build_mass_model, build_propulsion, build_geometry, build_initial_state
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF
from models.force_logger import ForceLogger
from core.solver6 import run_simulation_6dof
from geo.geographic import GeoModule, MissionConfig

# ============================================================================
# Konfiguracja
# ============================================================================
CASE_NAME    = "rocket_36mm_malarakieta_base"
MISSION_YAML = "missions/mission_01.yaml"
AERO_METHOD  = "missile_datcom"

# Impuls calkowity [N·s]
IMPULSE = 900.0 * 0.3   # = 270 N·s

# Sweep silnika — czasy pracy [s]
MOTOR_TIMES  = [0.300, 0.325, 0.350, 0.370, 0.375, 0.380, 0.400]

# Sweep statecznikow
FIN_CHORDS = [0.140, 0.160, 0.180, 0.200, 0.220]   # [m]
FIN_COUNTS = [4, 6, 8]

# Cant angle sweep
CANT_ANGLES = [0.0, 0.1, 0.25, 0.4, 0.5, 0.7, 1.0]   # [deg]

# ============================================================================
# Setup
# ============================================================================
print(f"Ladowanie: {CASE_NAME}")
cfg_base = load_config(f"configurations/{CASE_NAME}.yaml")
mission  = MissionConfig.from_yaml(MISSION_YAML)
geo      = GeoModule(mission)
atm      = create_atmosphere("ISA")
grv      = create_gravity("constant")

L = cfg_base.body.length
d = cfg_base.body.diameter
print(f"L={L*1000:.0f}mm  d={d*1000:.0f}mm")
print(f"Impuls calkowity: {IMPULSE:.1f} N·s")

def make_yaml(case_base, overrides: dict) -> tuple:
    """
    Tworzy tymczasowy YAML z nadpisanymi parametrami.
    overrides moze zawierac:
      thrust, t_burn, fin_chord, fin_count, cant_angle
    Zwraca (cfg, tmp_path) — tmp_path trzeba usunac po uzyciu.
    """
    import yaml
    with open(f"configurations/{case_base}.yaml",
              encoding='utf-8', errors='replace') as f:
        raw = yaml.safe_load(f.read())

    if "thrust" in overrides and "t_burn" in overrides:
        thrust  = overrides["thrust"]
        t_burn  = overrides["t_burn"]
        raw["propulsion"]["thrust_profile"] = [
            [0.0,   thrust],
            [t_burn, thrust],
            [round(t_burn + 0.001, 4), 0.0],
        ]

    if "fin_chord" in overrides:
        chord = overrides["fin_chord"]
        L_body = float(raw["body"]["length"])
        for fin in raw.get("fins", []):
            fin["root_chord"] = round(float(chord), 5)
            fin["tip_chord"]  = round(float(chord), 5)
            # Koniec statecznika zawsze na koncu rakiety
            fin["position"]   = round(L_body - float(chord), 5)

    if "fin_count" in overrides:
        for fin in raw.get("fins", []):
            fin["count"] = int(overrides["fin_count"])

    if "cant_angle" in overrides:
        for fin in raw.get("fins", []):
            fin["cant_angle"] = round(float(overrides["cant_angle"]), 5)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="_param_")
    os.close(tmp_fd)
    with open(tmp_path, "w", encoding='utf-8') as f:
        yaml.dump(raw, f, allow_unicode=True)
    cfg = load_config(tmp_path)
    return cfg, tmp_path

def run_case(cfg, aero, label="", log_forces=False):
    """Uruchamia symulacje i zwraca slownik wynikow."""
    try:
        mass    = build_mass_model(cfg)
        prop    = build_propulsion(cfg)
        geom    = build_geometry(cfg)
        initial = build_initial_state(mission)

        idx_a0  = np.argmin(np.abs(np.degrees(aero.alpha_table)))
        xcp_ref = float(aero.xcp_table[idx_a0, 0])
        geom.xcp = xcp_ref

        _logger = None
        if log_forces:
            _case_tag = label.replace(" ", "_").replace("=","").replace("°","")
            _logger = ForceLogger(_case_tag, log_dir=str(RESULTS_DIR / "logs"),
                                  enabled=True)

        fm = ForceModel6DOF(
            atmosphere=atm, mass_model=mass, aero_model=aero,
            gravity=grv, geometry=geom, propulsion=prop,
            logger=_logger,
        )
        result = run_simulation_6dof(
            fm, initial, t_max=60., z_ground=0.,
            max_step=0.02, rtol=1e-5, atol=1e-7,
        )
        if _logger is not None:
            _logger.close()

        # Odczytaj FA_x z loggera jesli byl wlaczony
        _FA_x = None
        _t_log = None
        if log_forces:
            import csv as _csv
            import glob as _glob
            _log_files = sorted(
                (_f for _f in (RESULTS_DIR / "logs").glob(
                    f"{label.replace(' ','_').replace('=','').replace('°','')}*.csv")),
                key=lambda p: p.stat().st_mtime
            )
            if _log_files:
                with open(_log_files[-1], newline="", encoding="utf-8") as _f:
                    _rows = list(_csv.DictReader(_f))
                if _rows and "FA_x" in _rows[0]:
                    _t_log  = np.array([float(r["t"])    for r in _rows])
                    _FA_x   = np.array([float(r["FA_x"]) for r in _rows])

        lf = geo.ned_to_lf(result)
        # Czas burnout z profilu ciagu
        if cfg.propulsion and cfg.propulsion.thrust_profile:
            tp = cfg.propulsion.thrust_profile
            t_burnout_est = float(max(t for t, f in tp if f > 0.0))
        else:
            t_burnout_est = 0.3

        # Czas tlumienia alpha — obwiednia lokalnych maksimow
        # t1 = pierwszy lokalny max po burnout, t2 = max gdy amplituda < 0.5*A1
        alpha_deg_arr = np.abs(np.degrees(result.alpha))
        t_arr         = result.t
        idx_start = int(np.searchsorted(t_arr, t_burnout_est + 0.1))
        idx_start = min(idx_start, len(alpha_deg_arr) - 2)
        alpha_post = alpha_deg_arr[idx_start:]
        alpha_max_val = float(alpha_post.max()) if len(alpha_post) > 0 else 0.0

        damp_time = float('nan')
        from scipy.signal import find_peaks as _find_peaks
        if len(alpha_post) > 4 and alpha_max_val > 0.05:
            peaks_rel, _ = _find_peaks(alpha_post, height=0.05)
            peaks_abs    = peaks_rel + idx_start
            if len(peaks_abs) >= 4:
                # Dopasuj eksponente do obwiedni szczytow: A(t) = A0 * exp(-t/tau)
                # log(A) = log(A0) - t/tau — regresja liniowa w log
                t_peaks = t_arr[peaks_abs].astype(float)
                A_peaks = alpha_deg_arr[peaks_abs].astype(float)
                # Zabezpieczenie przed zerowymi wartosciami
                mask = A_peaks > 1e-6
                if np.sum(mask) >= 3:
                    t_pk = t_peaks[mask]
                    A_pk = A_peaks[mask]
                    # Regresja liniowa: log(A) = a + b*t
                    b, a = np.polyfit(t_pk, np.log(A_pk), 1)
                    if b < 0:
                        # tau = -1/b, czas polowicznego zaniku = tau * ln(2)
                        tau = -1.0 / b
                        # damp_time: czas od pierwszego szczytu do A = 0.5*A1
                        t1 = float(t_pk[0])
                        A1 = float(np.exp(a + b * t1))
                        # A(t) = A1 * exp(b*(t-t1)) = 0.5*A1 => t-t1 = ln(0.5)/b
                        damp_time = float(np.log(0.5) / b)
                    else:
                        # Brak tłumienia — użyj metody poprzedniej jako fallback
                        A1 = float(alpha_deg_arr[peaks_abs[0]])
                        t1 = float(t_arr[peaks_abs[0]])
                        for pk in peaks_abs[1:]:
                            if float(alpha_deg_arr[pk]) <= 0.5 * A1:
                                damp_time = float(t_arr[pk]) - t1
                                break

        return {
            "ok":         True,
            "status":     result.status,
            "range_m":    float(lf["x_lf"][-1]),
            "alt_m":      float(lf["alt"].max()),
            "v_max":      float(result.speed.max()),
            "y_max":      float(np.max(np.abs(lf["y_lf"]))),
            "alpha_max":  alpha_max_val,   # max po burnout
            "damp_time":  damp_time,
            "t":          result.t,
            "alt_t":      -result.z,
            "v_t":        result.speed,
            "alpha_t":    np.degrees(result.alpha),
            "x_lf":       lf["x_lf"],
            "FA_x":       _FA_x,
            "t_log":      _t_log,
        }
    except Exception as e:
        print(f"         ! Blad: {e}")
        return {"ok": False, "status": "error"}

# ============================================================================
# A. Baseline — wykresy szczegolowe
# ============================================================================
print("\n" + "="*60)
print("A. Baseline")
print("="*60)

cfg_bl, tmp = make_yaml(CASE_NAME, {})
aero_bl = get_aero_model(CASE_NAME, method=AERO_METHOD)
os.unlink(tmp)

res_bl = run_case(cfg_bl, aero_bl, "baseline")
print(f"   Zasieg={res_bl['range_m']:.0f}m  Alt={res_bl['alt_m']:.0f}m  "
      f"Vmax={res_bl['v_max']:.1f}m/s  status={res_bl['status']}")

if res_bl["ok"]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Baseline — {CASE_NAME}", fontsize=12, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(res_bl["x_lf"]/1000, res_bl["alt_t"], 'b-', lw=2)
    ax.set_xlabel("Zasieg [km]"); ax.set_ylabel("Wysokosc [m]")
    ax.set_title("Wysokosc vs zasieg"); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(res_bl["t"], res_bl["alt_t"], 'b-', lw=2)
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Wysokosc [m]")
    ax.set_title("Wysokosc vs czas"); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(res_bl["t"], res_bl["v_t"], 'r-', lw=2)
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("V [m/s]")
    ax.set_title("Predkosc vs czas"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(res_bl["t"], res_bl["alpha_t"], 'g-', lw=1.5)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Alpha [deg]")
    ax.set_title("Kat natarcia vs czas"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "simulation_baseline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Zapisano: {RESULTS_DIR}/simulation_baseline.png")

# Wizualizacje geometrii
print("\nWizualizacje geometrii...")

# 1. Baseline
visualize_cfg(cfg_bl, "Baseline", RESULTS_DIR / "geometry_baseline.png")

# 2. n=4, chord=140mm
_ov = {"fin_chord": 0.140, "fin_count": 4, "cant_angle": 0.0}
_cfg_geo, _tmp = make_yaml(CASE_NAME, _ov)
visualize_cfg(_cfg_geo, "n=4 chord=140mm", RESULTS_DIR / "geometry_n4_c140.png")
os.unlink(_tmp)

# 3. n=8, chord=220mm
_ov = {"fin_chord": 0.220, "fin_count": 8, "cant_angle": 0.0}
_cfg_geo, _tmp = make_yaml(CASE_NAME, _ov)
visualize_cfg(_cfg_geo, "n=8 chord=220mm", RESULTS_DIR / "geometry_n8_c220.png")
os.unlink(_tmp)

# 4. cant=0.25 deg (baseline geometry)
_ov = {"cant_angle": 0.25}
_cfg_geo, _tmp = make_yaml(CASE_NAME, _ov)
visualize_cfg(_cfg_geo, "cant=0.25°", RESULTS_DIR / "geometry_cant025.png")
os.unlink(_tmp)

# ============================================================================
# B. Sweep silnika
# ============================================================================
print("\n" + "="*60)
print("B. Sweep silnika")
print("="*60)

motor_results = []
n_total = len(MOTOR_TIMES)

for i, t_burn in enumerate(MOTOR_TIMES):
    thrust = IMPULSE / t_burn
    label  = f"T={thrust:.0f}N t={t_burn:.3f}s"
    print(f"[{i+1}/{n_total}] {label}")

    cfg, tmp = make_yaml(CASE_NAME, {"thrust": thrust, "t_burn": t_burn})
    res = run_case(cfg, aero_bl, label, log_forces=True)
    os.unlink(tmp)

    motor_results.append({
        "t_burn":  t_burn,
        "thrust":  thrust,
        "label":   label,
        **{k: res.get(k, float('nan')) for k in
           ["range_m", "alt_m", "v_max", "alpha_max", "y_max", "damp_time", "status", "ok"]},
        "t":      res.get("t"),
        "alt_t":  res.get("alt_t"),
        "v_t":    res.get("v_t"),
        "alpha_t":res.get("alpha_t"),
        "x_lf":   res.get("x_lf"),
    })
    print(f"   Zasieg={res.get('range_m', 'ERR'):.0f}m  "
          f"Alt={res.get('alt_m', 'ERR'):.0f}m  "
          f"Vmax={res.get('v_max', 'ERR'):.1f}m/s  "
          f"status={res.get('status','?')}")

# Wykres sweep silnika
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Sweep silnika — staly impuls 270 N·s", fontsize=12, fontweight='bold')

ts   = [r["t_burn"] for r in motor_results]
cols = ['green' if r.get("status") == "ok" else 'orange' for r in motor_results]

# Kolory punktow wg statusu: ok=zielony, tumbling=pomaranczowy, blowup=czerwony
STATUS_COLORS = {"ok": "#28a745", "tumbling": "#fd7e14", "blowup": "#dc3545",
                 "error": "#6c757d"}

for ax, key, ylabel, title in zip(
    axes,
    ["range_m", "alt_m",  "v_max"],
    ["Zasieg [m]", "Wysokosc max [m]", "V_max [m/s]"],
    ["Zasieg",     "Wysokosc max",     "Predkosc max"],
):
    vals = [r.get(key, float('nan')) for r in motor_results]
    ax.plot(ts, vals, 'b-', lw=1.5, zorder=2, alpha=0.6)
    for j, (t, v, r) in enumerate(zip(ts, vals, motor_results)):
        if not np.isnan(v):
            sc = STATUS_COLORS.get(r.get("status", "ok"), "#28a745")
            ax.plot(t, v, 'o', color=sc, ms=10, zorder=4,
                    label=r.get("status","ok") if j == 0 else "")
            offset = 10 if j % 2 == 0 else -18
            lbl = f"{v:.0f}"
            if r.get("status") != "ok":
                lbl = lbl + chr(10) + "[" + str(r.get("status","?")) + "]"
            ax.annotate(lbl, (t, v), textcoords="offset points",
                        xytext=(0, offset), ha='center', fontsize=7.5)
    ax.set_xlabel("Czas pracy silnika [s]")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3)
    _vv = [r.get(key, float('nan')) for r in motor_results]
    _vv = [v for v in _vv if v == v and not (isinstance(v, float) and v != v)]
    _vv = [v for v in _vv if isinstance(v, (int, float)) and v == v]
    if _vv:
        _lo, _hi = min(_vv), max(_vv)
        _m = max(abs(_hi - _lo) * 0.1, abs(_hi) * 0.1)
        ax.set_ylim(_lo - _m, _hi + _m)


plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "sweep_motor.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Zapisano: {RESULTS_DIR}/sweep_motor.png")

# Trajektorie wszystkich silnikow
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Trajektorie — sweep silnika (staly impuls 270 N·s)",
             fontsize=11, fontweight="bold")
MOTOR_CMAP = plt.cm.viridis(np.linspace(0, 0.85, len(motor_results)))

ax = axes[0]
for r, clr in zip(motor_results, MOTOR_CMAP):
    if r.get("x_lf") is not None:
        ax.plot(np.array(r["x_lf"])/1000, r["alt_t"],
                color=clr, lw=1.5, label=r["label"])
ax.set_xlabel("Zasieg [km]"); ax.set_ylabel("Wysokosc [m]")
ax.set_title("Wysokosc vs zasieg"); ax.set_ylim(bottom=0)
ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes[1]
for r, clr in zip(motor_results, MOTOR_CMAP):
    if r.get("t") is not None:
        ax.plot(r["t"], r["alpha_t"], color=clr, lw=1.2, label=r["label"])
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel("Czas [s]"); ax.set_ylabel("Alpha [deg]")
ax.set_title("Kat natarcia vs czas")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Sila osiowa FA_x z loggera sil
ax = axes[2]
for r, clr in zip(motor_results, MOTOR_CMAP):
    if r.get("FA_x") is not None and r.get("t_log") is not None:
        ax.plot(r["t_log"], -r["FA_x"],   # FA_x jest ujemne (opor), negujemy
                color=clr, lw=1.5, label=r["label"])
    elif r.get("t") is not None:
        # Fallback gdy logger niedostepny
        rho   = 1.225
        q_dyn = 0.5 * rho * np.array(r["v_t"])**2
        FA    = 0.35 * q_dyn * np.pi * (0.036/2)**2
        ax.plot(r["t"], FA, color=clr, lw=1.5, ls='--', label=r["label"]+" (approx)")
ax.set_xlabel("Czas [s]"); ax.set_ylabel("|FA_x| [N]")
ax.set_title("Sila osiowa (opor) vs czas — z loggera")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "motor_trajectories.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Zapisano: {RESULTS_DIR}/motor_trajectories.png")

# Porownanie baseline vs t_burn=0.375s
print("\nPorownanie baseline vs t_burn=0.375s...")
t_opt   = 0.375
thrust_opt = IMPULSE / t_opt
cfg_opt, tmp_opt = make_yaml(CASE_NAME, {"thrust": thrust_opt, "t_burn": t_opt})
res_opt = run_case(cfg_opt, aero_bl, f"t={t_opt}s")
os.unlink(tmp_opt)
print(f"   t={t_opt}s: Zasieg={res_opt.get('range_m','ERR'):.0f}m  "
      f"status={res_opt.get('status','?')}")

if res_bl["ok"] and res_opt["ok"]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Baseline ({IMPULSE/0.3:.0f}N×0.300s) vs optymalny ({thrust_opt:.0f}N×{t_opt}s)\n"
        f"Impuls calkowity: {IMPULSE:.0f} N·s",
        fontsize=11, fontweight="bold"
    )
    LBLS  = [f"Baseline ({IMPULSE/0.3:.0f}N×0.300s)",
             f"t={t_opt}s ({thrust_opt:.0f}N×{t_opt}s)"]
    CLRS  = ["#1f77b4", "#ff7f0e"]
    RESES = [res_bl, res_opt]

    for ax, xs_key, ys_key, xlabel, ylabel, title in [
        (axes[0,0], "x_lf",  "alt_t",  "Zasieg [m]",   "Wysokosc [m]",  "Wysokosc vs zasieg"),
        (axes[0,1], "t",     "alt_t",  "Czas [s]",     "Wysokosc [m]",  "Wysokosc vs czas"),
        (axes[1,0], "t",     "v_t",    "Czas [s]",     "V [m/s]",       "Predkosc vs czas"),
        (axes[1,1], "t",     "alpha_t","Czas [s]",     "Alpha [deg]",   "Kat natarcia vs czas"),
    ]:
        for res, lbl, clr in zip(RESES, LBLS, CLRS):
            ax.plot(res[xs_key], res[ys_key], color=clr, lw=1.8, label=lbl)
        if ys_key == "alt_t":
            ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "comparison_motor_baseline_vs_opt.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {RESULTS_DIR}/comparison_motor_baseline_vs_opt.png")

# ============================================================================
# C. Sweep statecznikow
# ============================================================================
print("\n" + "="*60)
print("C. Sweep statecznikow")
print("="*60)

fin_results = []
n_total = len(FIN_COUNTS) * len(FIN_CHORDS)
n_done  = 0

COLORS = {4: '#1f77b4', 6: '#ff7f0e', 8: '#2ca02c'}
MARKERS = {0.140: 'o', 0.160: 's', 0.180: '^', 0.200: 'D', 0.220: 'v'}

for count in FIN_COUNTS:
    for chord in FIN_CHORDS:
        n_done += 1
        label = f"n={count} c={chord*1000:.0f}mm"
        print(f"[{n_done}/{n_total}] {label}  (DATCOM...)")

        overrides = {"fin_chord": chord, "fin_count": count, "cant_angle": 0.0}
        cfg, tmp  = make_yaml(CASE_NAME, overrides)

        case_tag = f"{CASE_NAME}_n{count}_c{int(chord*1000)}"
        # Zapisz tymczasowy YAML pod nazwa case_tag
        import shutil
        shutil.copy(tmp, f"configurations/{case_tag}.yaml")
        os.unlink(tmp)

        try:
            aero = get_aero_model(case_tag, method=AERO_METHOD, force_rerun=True)
            cfg2, tmp2 = make_yaml(CASE_NAME, overrides)
            res  = run_case(cfg2, aero, label)
            os.unlink(tmp2)
        except Exception as e:
            print(f"         ! Blad DATCOM: {e}")
            res = {"ok": False, "status": "error"}
        finally:
            try:
                os.unlink(f"configurations/{case_tag}.yaml")
            except Exception:
                pass

        fin_results.append({
            "count": count, "chord": chord, "label": label,
            **{k: res.get(k, float('nan')) for k in
               ["range_m", "alt_m", "v_max", "alpha_max", "y_max", "damp_time", "status", "ok"]},
        })
        print(f"   Zasieg={res.get('range_m', 'ERR'):.0f}m  "
              f"Alt={res.get('alt_m', 'ERR'):.0f}m  "
              f"Vmax={res.get('v_max', 'ERR'):.1f}m/s  "
              f"status={res.get('status','?')}")

# Wykres sweep statecznikow
from matplotlib.gridspec import GridSpec as _GS
fig = plt.figure(figsize=(18, 10))
_gs  = _GS(2, 6, figure=fig, hspace=0.4, wspace=0.35)
_ax1 = fig.add_subplot(_gs[0, 0:2])
_ax2 = fig.add_subplot(_gs[0, 2:4])
_ax3 = fig.add_subplot(_gs[0, 4:6])
_ax4 = fig.add_subplot(_gs[1, 1:3])
_ax5 = fig.add_subplot(_gs[1, 3:5])
axes = [_ax1, _ax2, _ax3, _ax4, _ax5]
fig.suptitle("Sweep statecznikow — count x chord", fontsize=12, fontweight='bold')

for ax, key, ylabel, title in zip(
    axes,
    ["range_m",  "alt_m",         "v_max",       "alpha_max",        "damp_time"],
    ["Zasieg [m]", "Wysokosc max [m]", "V_max [m/s]", "Alpha max [deg]", "Czas tlumienia [s]"],
    ["Zasieg",     "Wysokosc max",     "Predkosc max", "Kat natarcia max", "Czas tlumienia alpha"],
):
    for count in FIN_COUNTS:
        recs = [r for r in fin_results if r["count"] == count]
        xs   = [r["chord"]*1000 for r in recs]
        ys   = [r.get(key, float('nan')) for r in recs]
        ax.plot(xs, ys, '-', color=COLORS[count],
                lw=1.5, label=f"n={count}", alpha=0.7, zorder=2)
        for x, y, r in zip(xs, ys, recs):
            m  = MARKERS[r["chord"]]
            ec = STATUS_COLORS.get(r.get("status","ok"), "#28a745")
            ax.plot(x, y, marker=m, color=COLORS[count],
                    ms=9, zorder=3)
    ax.set_xlabel("Cięciwa statecznika [mm]")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _vv = [r.get(key, float('nan')) for r in fin_results]
    _vv = [v for v in _vv if isinstance(v, (int, float)) and v == v]
    if _vv:
        _lo, _hi = min(_vv), max(_vv)
        _m = max(abs(_hi - _lo) * 0.1, abs(_hi) * 0.1)
        ax.set_ylim(_lo - _m, _hi + _m)

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "sweep_fins.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Zapisano: {RESULTS_DIR}/sweep_fins.png")

# ============================================================================
# D. Sweep cant angle
# ============================================================================
print("\n" + "="*60)
print("D. Sweep cant angle")
print("="*60)

cant_results = []
for i, cant in enumerate(CANT_ANGLES):
    label = f"cant={cant}°"
    print(f"[{i+1}/{len(CANT_ANGLES)}] {label}")

    overrides = {"cant_angle": cant}
    cfg, tmp  = make_yaml(CASE_NAME, overrides)
    case_tag  = f"{CASE_NAME}_cant{str(cant).replace('.','p')}"
    import shutil
    shutil.copy(tmp, f"configurations/{case_tag}.yaml")
    os.unlink(tmp)

    try:
        aero = get_aero_model(case_tag, method=AERO_METHOD, force_rerun=True)
        cfg2, tmp2 = make_yaml(CASE_NAME, overrides)
        res  = run_case(cfg2, aero, label)
        os.unlink(tmp2)
    except Exception as e:
        print(f"         ! Blad: {e}")
        res = {"ok": False, "status": "error"}
    finally:
        try:
            os.unlink(f"configurations/{case_tag}.yaml")
        except Exception:
            pass

    cant_results.append({
        "cant": cant, "label": label,
        **{k: res.get(k, float('nan')) for k in
           ["range_m", "alt_m", "v_max", "y_max", "alpha_max",
            "damp_time", "status", "ok"]},
    })
    print(f"   Zasieg={res.get('range_m','ERR'):.0f}m  "
          f"Alt={res.get('alt_m','ERR'):.0f}m  "
          f"Vmax={res.get('v_max','ERR'):.1f}m/s  "
          f"Bok={res.get('y_max','ERR'):.1f}m  "
          f"status={res.get('status','?')}")

# Wykres cant angle
fig, axes_2d = plt.subplots(2, 2, figsize=(12, 9))
axes = list(axes_2d.flat)
fig.suptitle(f"Porownanie konfiguracji z roznym katem zaklinowania\n"
             f"xcg={cfg_base.mass.xcg_ref*1000:.0f}mm  "
             f"cant: {min(CANT_ANGLES)}° — {max(CANT_ANGLES)}°",
             fontsize=12, fontweight='bold')

labels = [r["label"] for r in cant_results]
cols   = ['#1f77b4', '#ff7f0e']

cant_vals_plot = [r["cant"] for r in cant_results]
for ax, key, ylabel, title, fmt in zip(
    axes,
    ["range_m",   "alt_m",          "v_max",       "y_max"],
    ["Zasieg [m]","Wysokosc max [m]","V_max [m/s]", "Odchylenie boczne [m]"],
    ["Zasieg",    "Wysokosc max",    "Predkosc max","Odchylenie boczne"],
    [".0f",       ".1f",            ".1f",          ".1f"],
):
    vals = [r.get(key, float('nan')) for r in cant_results]
    ax.plot(cant_vals_plot, vals, 'b-o', lw=1.5, ms=8, zorder=3)
    # Usun notacje naukowa
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.4g}"))
    # Podpisy z inteligentnym offsetem — naprzemiennie gora/dol
    for j, (x, v, r) in enumerate(zip(cant_vals_plot, vals, cant_results)):
        if not np.isnan(v):
            # Czerwony jesli alpha_max > 10 (tumbling niezdetektowany przez status)
            amax = r.get("alpha_max", 0.)
            if not isinstance(amax, float):
                amax = float('nan')
            if amax > 10.:
                sc = "#dc3545"   # czerwony — tumbling
            elif r.get("status") != "ok":
                sc = STATUS_COLORS.get(r.get("status","ok"), "#28a745")
            else:
                sc = "#28a745"   # zielony — ok
            ax.plot(x, v, 'o', color=sc, ms=10, zorder=4)
            offset = 10 if j % 2 == 0 else -18
            lbl = f"{v:{fmt}}"
            if r.get("status") != "ok":
                lbl = lbl + chr(10) + "[" + str(r.get("status","?")) + "]"
            ax.annotate(lbl, (x, v), textcoords="offset points",
                        xytext=(0, offset), ha='center', fontsize=8)
    ax.set_xlabel("Kat zaklinowania [deg]")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3)
    # Zakres osi Y +/-10% od min/max wartosci
    _vv = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if _vv:
        _lo, _hi = min(_vv), max(_vv)
        _m = max(abs(_hi - _lo) * 0.1, abs(_hi) * 0.1)
        ax.set_ylim(_lo - _m, _hi + _m)

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "sweep_cant.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Zapisano: {RESULTS_DIR}/sweep_cant.png")

# ============================================================================
# Zapis CSV
# ============================================================================
COMMON_FIELDS = ["sweep", "label", "t_burn", "thrust", "count", "chord",
                 "cant", "range_m", "alt_m", "v_max", "y_max", "alpha_max", "damp_time", "status"]

def to_row(sweep_name, r):
    row = {"sweep": sweep_name}
    for fld in COMMON_FIELDS[1:]:
        row[fld] = r.get(fld, float("nan"))
    return row

with open(str(RESULTS_DIR / "parametric_results.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COMMON_FIELDS)
    writer.writeheader()
    for r in motor_results:
        writer.writerow(to_row("motor", r))
    for r in fin_results:
        writer.writerow(to_row("fins", r))
    for r in cant_results:
        writer.writerow(to_row("cant", r))
print(f"Zapisano: {RESULTS_DIR}/parametric_results.csv")
print("\nGotowe.")
