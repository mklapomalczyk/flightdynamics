"""
analyze_xcg_sweep.py
====================
Analiza wplywu polozenia xcg na stabilnosc i osiagi rakiety.
Uruchomienie z katalogu FlightSim: python analyze_xcg_sweep.py
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
N_XCG        = 10
XCG_FRAC_MIN = 0.20
XCG_FRAC_MAX = 0.80
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

print(f"Ladowanie modelu aero [{AERO_METHOD}]...")
aero = get_aero_model(CASE_NAME, method=AERO_METHOD)

idx_a0  = np.argmin(np.abs(np.degrees(aero.alpha_table)))
idx_m0  = 0
xcp_ref = float(aero.xcp_table[idx_a0, idx_m0])
print(f"xcp (alpha=0, Ma={aero.mach_table[idx_m0]:.1f}) = {xcp_ref*1000:.1f}mm od nosa")

xcg_fracs = np.linspace(XCG_FRAC_MIN, XCG_FRAC_MAX, N_XCG)
xcg_vals  = xcg_fracs * L
print(f"\nSweep xcg: {xcg_vals[0]*1000:.0f}mm -> {xcg_vals[-1]*1000:.0f}mm")

def make_cfg_with_xcg(cfg_base, xcg_new):
    import yaml
    yaml_path = f"configurations/{CASE_NAME}.yaml"
    with open(yaml_path, encoding='utf-8', errors='replace') as f:
        raw = yaml.safe_load(f.read())
    raw["mass"]["xcg_ref"] = round(float(xcg_new), 5)
    if "mass_model" in raw:
        raw["mass_model"]["full"]["xcg"]  = round(float(xcg_new), 5)
        raw["mass_model"]["empty"]["xcg"] = round(float(xcg_new), 5)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="_xcg_sweep_")
    os.close(tmp_fd)
    with open(tmp_path, "w", encoding='utf-8') as f:
        yaml.dump(raw, f, allow_unicode=True)
    cfg = load_config(tmp_path)
    os.unlink(tmp_path)
    return cfg

records = []

for i, xcg in enumerate(xcg_vals):
    xcg_pct = xcg / L * 100
    SM      = (xcp_ref - xcg) / d
    stable_static  = SM > 0
    stable_dynamic = SM > 1.0

    print(f"[{i+1:2d}/{N_XCG}] xcg={xcg*1000:.1f}mm ({xcg_pct:.0f}%L) "
          f"SM={SM:.2f} kal  {'STABILNA' if stable_static else 'NIESTABILNA'}")

    x_range = v_max = alt_max = t_flight = float('nan')
    tumble  = False

    if stable_static:
        try:
            cfg  = make_cfg_with_xcg(cfg_base, xcg)
            mass = build_mass_model(cfg)

            # --- DIAGNOSTYKA xcg ---
            ms0 = mass.at(0.0)
            ms5 = mass.at(5.0)
            print(f"  CHECK xcg @ t=0s: {ms0.xcg*1000:.2f}mm  "
                  f"t=5s: {ms5.xcg*1000:.2f}mm  "
                  f"(oczekiwane: {xcg*1000:.1f}mm)")

            prop = build_propulsion(cfg)
            geom = build_geometry(cfg)
            geom.xcp = xcp_ref

            initial = build_initial_state(mission)
            fm = ForceModel6DOF(
                atmosphere=atm, mass_model=mass, aero_model=aero,
                gravity=grv, geometry=geom, propulsion=prop,
            )
            # Poluzowane tolerancje — analiza parametryczna, nie wymagamy precyzji
            # solver6 rzuca RuntimeError gdy nie zbiega — lapie wyzej jako Exception
            result = run_simulation_6dof(
                fm, initial, t_max=35., z_ground=0.,
                max_step=0.2, rtol=1e-3, atol=1e-5,
            )
            lf       = geo.ned_to_lf(result)
            x_range  = float(lf["x_lf"][-1])
            alt_max  = float(lf["alt"].max())
            t_flight = float(result.t[-1])
            v_max    = float(result.speed.max())

            alpha_deg   = np.degrees(result.alpha)
            tumble_mask = np.abs(alpha_deg) > 30.
            tumble_time = np.sum(tumble_mask) * (result.t[-1] / len(result.t))
            tumble      = tumble_time > 2.0
            if tumble:
                print(f"         ! Tumbling ({tumble_time:.1f}s)")

        except Exception as e:
            print(f"         ! Blad: {e}")

    records.append({
        "xcg_m": xcg, "xcg_pct": xcg_pct, "SM_cal": SM,
        "stable_s": stable_static, "stable_d": stable_dynamic and not tumble,
        "tumble": tumble,
        "range_m": x_range, "alt_max_m": alt_max,
        "t_flight_s": t_flight, "v_max_ms": v_max,
    })

# Zapis CSV
with open("xcg_sweep_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
print("\nZapisano: xcg_sweep_results.csv")

# ============================================================================
# Wykres
# ============================================================================
stable_s = [r for r in records if r["stable_s"] and not np.isnan(r["range_m"])]
xcg_pcts = [r["xcg_pct"] for r in records]
SMs      = [r["SM_cal"]  for r in records]

fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    f"Analiza polozenia xcg — {CASE_NAME}\n"
    f"xcp={xcp_ref*1000:.1f}mm  L={L*1000:.0f}mm  d={d*1000:.0f}mm",
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. SM vs xcg
ax = fig.add_subplot(gs[0, 0])
bar_colors = ['#28a745' if r["stable_d"] else ('#fd7e14' if r["stable_s"] else '#dc3545')
              for r in records]
ax.bar(xcg_pcts, SMs, color=bar_colors, edgecolor='gray', linewidth=0.5, width=5)
ax.axhline(0,   color='#dc3545', lw=1.5, ls='--')
ax.axhline(1.0, color='#fd7e14', lw=1.0, ls=':')
ax.axhline(2.0, color='#28a745', lw=1.0, ls=':')
ax.set_xlabel("xcg [% dlugosci]"); ax.set_ylabel("SM [kalibry]")
ax.set_title("Zapas statyczny SM")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#28a745', label='Stabilna (SM>1)'),
    Patch(color='#fd7e14', label='Slabo stabilna'),
    Patch(color='#dc3545', label='Niestabilna'),
], fontsize=7)
ax.grid(alpha=0.3)

# 2. Zasieg
ax = fig.add_subplot(gs[0, 1])
if stable_s:
    xs = [r["xcg_pct"] for r in stable_s]
    ys = [r["range_m"]/1000 for r in stable_s]
    cs = ['#fd7e14' if r["tumble"] else '#28a745' for r in stable_s]
    ax.plot(xs, ys, 'b-', lw=1, alpha=0.4)
    ax.scatter(xs, ys, c=cs, s=80, zorder=3)
ax.set_xlabel("xcg [% dlugosci]"); ax.set_ylabel("Zasieg [km]")
ax.set_title("Zasieg vs xcg"); ax.grid(alpha=0.3)

# 3. Wysokosc max
ax = fig.add_subplot(gs[0, 2])
if stable_s:
    xs = [r["xcg_pct"] for r in stable_s]
    ys = [r["alt_max_m"] for r in stable_s]
    cs = ['#fd7e14' if r["tumble"] else '#28a745' for r in stable_s]
    ax.plot(xs, ys, 'b-', lw=1, alpha=0.4)
    ax.scatter(xs, ys, c=cs, s=80, zorder=3)
ax.set_xlabel("xcg [% dlugosci]"); ax.set_ylabel("Wysokosc max [m]")
ax.set_title("Wysokosc max vs xcg"); ax.grid(alpha=0.3)

# 4. Czas lotu
ax = fig.add_subplot(gs[1, 0])
if stable_s:
    xs = [r["xcg_pct"] for r in stable_s]
    ys = [r["t_flight_s"] for r in stable_s]
    cs = ['#fd7e14' if r["tumble"] else '#28a745' for r in stable_s]
    ax.plot(xs, ys, 'b-', lw=1, alpha=0.4)
    ax.scatter(xs, ys, c=cs, s=80, zorder=3)
ax.set_xlabel("xcg [% dlugosci]"); ax.set_ylabel("Czas lotu [s]")
ax.set_title("Czas lotu vs xcg"); ax.grid(alpha=0.3)

# 5. V_max
ax = fig.add_subplot(gs[1, 1])
if stable_s:
    xs = [r["xcg_pct"] for r in stable_s]
    ys = [r["v_max_ms"] for r in stable_s]
    cs = ['#fd7e14' if r["tumble"] else '#28a745' for r in stable_s]
    ax.plot(xs, ys, 'b-', lw=1, alpha=0.4)
    ax.scatter(xs, ys, c=cs, s=80, zorder=3)
ax.set_xlabel("xcg [% dlugosci]"); ax.set_ylabel("V_max [m/s]")
ax.set_title("Predkosc max vs xcg"); ax.grid(alpha=0.3)

# 6. Tabela
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
col_labels = ["xcg\n[%L]", "SM\n[kal]", "Status", "Zasieg\n[m]", "Alt\n[m]"]
rows = []
row_colors = []
for r in records:
    if r["tumble"]:      status = "tumbling"; rc = '#fff3cd'
    elif r["stable_d"]:  status = "OK";       rc = '#d4edda'
    elif r["stable_s"]:  status = "slaba";    rc = '#fff3cd'
    else:                status = "NIESTAB";   rc = '#f8d7da'
    rows.append([
        f"{r['xcg_pct']:.0f}%",
        f"{r['SM_cal']:.2f}",
        status,
        f"{r['range_m']:.0f}" if not np.isnan(r['range_m']) else "-",
        f"{r['alt_max_m']:.0f}" if not np.isnan(r['alt_max_m']) else "-",
    ])
    row_colors.append(rc)

tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.3)
for i, rc in enumerate(row_colors):
    for j in range(len(col_labels)):
        tbl[(i+1, j)].set_facecolor(rc)
ax.set_title("Podsumowanie", fontsize=9, fontweight='bold')

plt.savefig("xcg_sweep_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Zapisano: xcg_sweep_results.png")
print("\nGotowe.")
