"""
analyze_cant_sweep.py
=====================
Analiza wplywu kata zaklinowania statecznikow na stabilnosc i osiagi rakiety.

Iteruje cant_angle od 0 do 3 stopni (10 przypadkow).
xcg ustalone na 33% dlugosci (SM~4.27 kal).

Uruchomienie z katalogu FlightSim:
    python analyze_cant_sweep.py
"""

import sys, csv, tempfile, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aero import get_aero_model
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_mass_model, build_propulsion, build_geometry, build_initial_state
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from geo.geographic import GeoModule, MissionConfig

# ============================================================================
CASE_NAME    = "rocket_36mm_malarakieta_base"
MISSION_YAML = "missions/mission_01.yaml"
AERO_METHOD  = "missile_datcom"
N_CANT       = 10
CANT_MIN     = 0.0    # stopnie
CANT_MAX     = 3.0    # stopnie
XCG_FRAC     = 0.33   # stale xcg = 33% dlugosci
# ============================================================================

print(f"Ladowanie: {CASE_NAME}")
cfg_base = load_config(f"configurations/{CASE_NAME}.yaml")
mission  = MissionConfig.from_yaml(MISSION_YAML)
geo      = GeoModule(mission)
atm      = create_atmosphere("ISA")
grv      = create_gravity("constant")

L = cfg_base.body.length
d = cfg_base.body.diameter
xcg_fixed = XCG_FRAC * L

print(f"L={L*1000:.0f}mm  d={d*1000:.0f}mm")
print(f"xcg stale = {xcg_fixed*1000:.1f}mm ({XCG_FRAC*100:.0f}%L)")

cant_vals = np.linspace(CANT_MIN, CANT_MAX, N_CANT)

def make_cfg_with_cant_xcg(cfg_base, cant_angle, xcg_new):
    """Zwraca kopie konfiguracji z nowym cant_angle i xcg."""
    import yaml
    yaml_path = f"configurations/{CASE_NAME}.yaml"
    with open(yaml_path, encoding='utf-8', errors='replace') as f:
        raw = yaml.safe_load(f.read())
    # Zmien xcg
    raw["mass"]["xcg_ref"] = round(float(xcg_new), 5)
    if "mass_model" in raw:
        raw["mass_model"]["full"]["xcg"]  = round(float(xcg_new), 5)
        raw["mass_model"]["empty"]["xcg"] = round(float(xcg_new), 5)
    # Zmien cant_angle dla wszystkich zestawow pletw
    if "fins" in raw:
        for fin in raw["fins"]:
            fin["cant_angle"] = round(float(cant_angle), 5)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="_cant_sweep_")
    os.close(tmp_fd)
    with open(tmp_path, "w", encoding='utf-8') as f:
        yaml.dump(raw, f, allow_unicode=True)
    cfg = load_config(tmp_path)
    os.unlink(tmp_path)
    return cfg

# Zaladuj aero raz — zmiana cant_angle wymaga nowego przebiegu DATCOM
# Dlatego ladujemy osobno dla kazdego przypadku
print(f"\nSweep cant_angle: {CANT_MIN}° -> {CANT_MAX}°  (xcg={xcg_fixed*1000:.0f}mm)")
print(f"UWAGA: kazdy przypadek wymaga nowego przebiegu DATCOM\n")

records = []

for i, cant in enumerate(cant_vals):
    print(f"[{i+1:2d}/{N_CANT}] cant={cant:.3f}°")

    x_range = v_max = alt_max = t_flight = p_eq = float('nan')
    tumble  = False

    try:
        cfg  = make_cfg_with_cant_xcg(cfg_base, cant, xcg_fixed)
        mass = build_mass_model(cfg)
        prop = build_propulsion(cfg)
        geom = build_geometry(cfg)

        # Nowy przebieg DATCOM dla kazdego cant_angle
        # Zapisz tymczasowy YAML z nowym cant_angle i wczytaj aero z niego
        import yaml as _yaml
        case_name_cant = f"{CASE_NAME}_cant{cant:.3f}".replace(".", "p")
        cant_yaml_path = f"configurations/{case_name_cant}.yaml"
        with open(f"configurations/{CASE_NAME}.yaml",
                  encoding='utf-8', errors='replace') as _f:
            _raw = _yaml.safe_load(_f.read())
        _raw["mass"]["xcg_ref"] = round(float(xcg_fixed), 5)
        if "mass_model" in _raw:
            _raw["mass_model"]["full"]["xcg"]  = round(float(xcg_fixed), 5)
            _raw["mass_model"]["empty"]["xcg"] = round(float(xcg_fixed), 5)
        if "fins" in _raw:
            for _fin in _raw["fins"]:
                _fin["cant_angle"] = round(float(cant), 5)
        with open(cant_yaml_path, "w", encoding='utf-8') as _f:
            _yaml.dump(_raw, _f, allow_unicode=True)

        aero = get_aero_model(
            case_name_cant,
            method      = AERO_METHOD,
            force_rerun = True,
        )

        # Usun tymczasowy YAML
        try:
            os.unlink(cant_yaml_path)
        except Exception:
            pass

        # xcp z tabeli (alpha=0, Ma=0.3)
        idx_a0  = np.argmin(np.abs(np.degrees(aero.alpha_table)))
        idx_m0  = 0
        xcp_ref = float(aero.xcp_table[idx_a0, idx_m0])
        SM      = (xcp_ref - xcg_fixed) / d
        geom.xcp = xcp_ref

        # Rownowagowa predkosc obrotowa przy Ma=0.3 (V~100 m/s, q~6000 Pa)
        # p_eq = -CLL * q * S * d / (Clp * (d/(2V)) * q * S * d)
        #      = -CLL / (Clp * d/(2V))
        if aero.CLL_table is not None and aero.Clp_table is not None:
            CLL = float(aero.CLL_table[idx_a0, idx_m0])
            Clp = float(aero.Clp_table[idx_a0, idx_m0])
            V_ref = 100.  # m/s przy Ma~0.3
            if abs(Clp) > 1e-10:
                p_eq_rad = -CLL / (Clp * d / (2 * V_ref))
                p_eq     = np.degrees(p_eq_rad)
        else:
            p_eq = 0.0

        print(f"         xcp={xcp_ref*1000:.1f}mm  SM={SM:.2f} kal  "
              f"CLL={CLL if aero.CLL_table is not None else 0.:.5f}  "
              f"p_eq={p_eq:.0f} °/s ({p_eq/360:.1f} obr/s)")

        initial = build_initial_state(mission)
        fm = ForceModel6DOF(
            atmosphere=atm, mass_model=mass, aero_model=aero,
            gravity=grv, geometry=geom, propulsion=prop,
        )
        result = run_simulation_6dof(
            fm, initial, t_max=35., z_ground=0.,
            max_step=0.1, rtol=1e-3, atol=1e-5,
        )

        lf       = geo.ned_to_lf(result)
        x_range  = float(lf["x_lf"][-1])
        alt_max  = float(lf["alt"].max())
        t_flight = float(result.t[-1])
        v_max    = float(result.speed.max())

        # Detekcja tumblingu
        # Tumbling: sqrt(q^2+r^2) > 30 deg/s przez ponad 3s
        omega_qr    = np.sqrt(np.degrees(result.qr)**2 +
                              np.degrees(result.r)**2)
        tumble_mask = omega_qr > 30.
        tumble_time = np.sum(tumble_mask) * (result.t[-1] / len(result.t))
        tumble      = tumble_time > 3.0
        if tumble:
            print(f"         ! Tumbling ({tumble_time:.1f}s)")

    except Exception as e:
        print(f"         ! Blad: {e}")
        SM = float('nan')

    records.append({
        "cant_deg":   cant,
        "SM_cal":     SM if not np.isnan(SM) else float('nan'),
        "p_eq_degs":  p_eq,
        "p_eq_rps":   p_eq/360. if not np.isnan(p_eq) else float('nan'),
        "tumble":     tumble,
        "range_m":    x_range,
        "alt_max_m":  alt_max,
        "t_flight_s": t_flight,
        "v_max_ms":   v_max,
    })

# Zapis CSV
with open("cant_sweep_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
print("\nZapisano: cant_sweep_results.csv")

# ============================================================================
# Wykres
# ============================================================================
ok_recs = [r for r in records if not np.isnan(r["range_m"])]

fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    f"Analiza kata zaklinowania — {CASE_NAME}\n"
    f"xcg={xcg_fixed*1000:.0f}mm ({XCG_FRAC*100:.0f}%L)  "
    f"L={L*1000:.0f}mm  d={d*1000:.0f}mm",
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

def sc_color(r):
    return '#fd7e14' if r["tumble"] else '#28a745'

# 1. p_eq vs cant
ax = fig.add_subplot(gs[0, 0])
if ok_recs:
    xs = [r["cant_deg"] for r in ok_recs]
    ys = [abs(r["p_eq_degs"]) for r in ok_recs]
    cs = [sc_color(r) for r in ok_recs]
    ax.plot(xs, ys, 'b-', lw=1, alpha=0.4)
    ax.scatter(xs, ys, c=cs, s=80, zorder=3)
ax.set_xlabel("Kat zaklinowania [°]")
ax.set_ylabel("p_eq [°/s]")
ax.set_title("Rownowagowa predkosc obrotowa")
ax.grid(alpha=0.3)

# 2. p_eq w obr/s
ax = fig.add_subplot(gs[0, 1])
if ok_recs:
    xs = [r["cant_deg"] for r in ok_recs]
    ys = [abs(r["p_eq_rps"]) for r in ok_recs]
    cs = [sc_color(r) for r in ok_recs]
    ax.plot(xs, ys, 'b-', lw=1, alpha=0.4)
    ax.scatter(xs, ys, c=cs, s=80, zorder=3)
ax.set_xlabel("Kat zaklinowania [°]")
ax.set_ylabel("p_eq [obr/s]")
ax.set_title("Rownowagowa predkosc obrotowa [obr/s]")
ax.grid(alpha=0.3)

# 3. Zasieg
ax = fig.add_subplot(gs[0, 2])
if ok_recs:
    xs = [r["cant_deg"] for r in ok_recs]
    ys = [r["range_m"]/1000 for r in ok_recs]
    cs = [sc_color(r) for r in ok_recs]
    ax.plot(xs, ys, 'b-', lw=1, alpha=0.4)
    ax.scatter(xs, ys, c=cs, s=80, zorder=3)
ax.set_xlabel("Kat zaklinowania [°]")
ax.set_ylabel("Zasieg [km]")
ax.set_title("Zasieg vs kant zaklinowania")
ax.grid(alpha=0.3)

# 4. Wysokosc max
ax = fig.add_subplot(gs[1, 0])
if ok_recs:
    xs = [r["cant_deg"] for r in ok_recs]
    ys = [r["alt_max_m"] for r in ok_recs]
    cs = [sc_color(r) for r in ok_recs]
    ax.plot(xs, ys, 'b-', lw=1, alpha=0.4)
    ax.scatter(xs, ys, c=cs, s=80, zorder=3)
ax.set_xlabel("Kat zaklinowania [°]")
ax.set_ylabel("Wysokosc max [m]")
ax.set_title("Wysokosc max vs kat zaklinowania")
ax.grid(alpha=0.3)

# 5. Czas lotu
ax = fig.add_subplot(gs[1, 1])
if ok_recs:
    xs = [r["cant_deg"] for r in ok_recs]
    ys = [r["t_flight_s"] for r in ok_recs]
    cs = [sc_color(r) for r in ok_recs]
    ax.plot(xs, ys, 'b-', lw=1, alpha=0.4)
    ax.scatter(xs, ys, c=cs, s=80, zorder=3)
ax.set_xlabel("Kat zaklinowania [°]")
ax.set_ylabel("Czas lotu [s]")
ax.set_title("Czas lotu vs kat zaklinowania")
ax.grid(alpha=0.3)

# 6. Tabela
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
col_labels = ["cant\n[°]", "SM\n[kal]", "p_eq\n[obr/s]", "Status",
              "Zasieg\n[m]", "Alt\n[m]"]
rows = []
row_colors = []
for r in records:
    status = "tumbling" if r["tumble"] else "OK"
    rc = '#fff3cd' if r["tumble"] else '#d4edda'
    if np.isnan(r["range_m"]):
        status = "blad"; rc = '#f8d7da'
    rows.append([
        f"{r['cant_deg']:.2f}",
        f"{r['SM_cal']:.2f}" if not np.isnan(r['SM_cal']) else "-",
        f"{abs(r['p_eq_rps']):.1f}" if not np.isnan(r['p_eq_rps']) else "-",
        status,
        f"{r['range_m']:.0f}" if not np.isnan(r['range_m']) else "-",
        f"{r['alt_max_m']:.0f}" if not np.isnan(r['alt_max_m']) else "-",
    ])
    row_colors.append(rc)

tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1.1, 1.25)
for i, rc in enumerate(row_colors):
    for j in range(len(col_labels)):
        tbl[(i+1, j)].set_facecolor(rc)

from matplotlib.patches import Patch
fig.legend(handles=[
    Patch(color='#28a745', label='Stabilna'),
    Patch(color='#fd7e14', label='Tumbling'),
], loc='lower center', ncol=2, fontsize=9)

ax.set_title("Podsumowanie", fontsize=9, fontweight='bold')

plt.savefig("cant_sweep_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Zapisano: cant_sweep_results.png")
print("\nGotowe.")

# Koniec pliku
