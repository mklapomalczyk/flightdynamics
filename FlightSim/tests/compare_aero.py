"""
compare_aero.py — Porównanie modeli: Barrowman vs Missile DATCOM
"""
import sys, numpy as np, matplotlib.pyplot as plt
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

CASE_NAME    = "rocket_36mm_malarakieta_base"
MISSION_YAML = "missions/mission_01.yaml"
CMQ          = -500.0

cfg     = load_config(f"configurations/{CASE_NAME}.yaml")
mass    = build_mass_model(cfg)
prop    = build_propulsion(cfg)
geom    = build_geometry(cfg)
mission = MissionConfig.from_yaml(MISSION_YAML)
geo     = GeoModule(mission)
initial = build_initial_state(mission)
atm     = create_atmosphere("ISA")
grv     = create_gravity("constant")

print("Ladowanie modelu Barrowman...")
aero_bar = get_aero_model(CASE_NAME, Cmq=CMQ, method="barrowman")
print("Ladowanie modelu Missile DATCOM...")
aero_dat = get_aero_model(CASE_NAME, Cmq=CMQ, method="missile_datcom", force_rerun=True)

models = {"Barrowman": aero_bar, "Missile DATCOM": aero_dat}
colors = {"Barrowman": "#4f8ef7", "Missile DATCOM": "#f97316"}
lsmap  = {"Barrowman": "-",       "Missile DATCOM": "--"}

# ---- Symulacje -----------------------------------------------------------
results = {}
for name, aero in models.items():
    print(f"\nSymulacja: {name}")
    fm = ForceModel6DOF(atmosphere=atm, mass_model=mass, aero_model=aero,
                        gravity=grv, geometry=geom, propulsion=prop)
    result = run_simulation_6dof(fm, initial, t_max=120., z_ground=-1., max_step=0.05)
    results[name] = result
    lf = geo.ned_to_lf(result)
    print(f"  Zasieg:   {lf['x_lf'][-1]/1000:.2f} km")
    print(f"  Wys max:  {lf['alt'].max():.0f} m")
    print(f"  Czas:     {result.t[-1]:.1f} s")

# ---- Wykres 1: wspolczynniki vs Mach ------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle(f"Wspolczynniki aerodynamiczne — {CASE_NAME}  (α=0°)", fontsize=12, fontweight='bold')

mach_arr = np.linspace(
    max(aero_bar.mach_table[0], aero_dat.mach_table[0]),
    min(aero_bar.mach_table[-1], aero_dat.mach_table[-1]),
    120
)

def interp_row(aero, alpha_deg):
    idx = np.argmin(np.abs(np.degrees(aero.alpha_table) - alpha_deg))
    CA  = np.interp(mach_arr, aero.mach_table, aero.CA_table[idx, :])
    CN  = np.interp(mach_arr, aero.mach_table, aero.CN_table[idx, :])
    xcp = np.interp(mach_arr, aero.mach_table, aero.xcp_table[idx, :])
    return CA, CN, xcp

CA_b, CN_b, xcp_b = interp_row(aero_bar, 0.)
CA_d, CN_d, xcp_d = interp_row(aero_dat, 0.)

for name, (CA, CN, xcp) in [("Barrowman",(CA_b,CN_b,xcp_b)), ("Missile DATCOM",(CA_d,CN_d,xcp_d))]:
    c, ls = colors[name], lsmap[name]
    axes1[0].plot(mach_arr, CA,  color=c, ls=ls, lw=2, label=name)
    axes1[1].plot(mach_arr, CN,  color=c, ls=ls, lw=2, label=name)
    axes1[2].plot(mach_arr, xcp, color=c, ls=ls, lw=2, label=name)

axes1[0].fill_between(mach_arr, CA_b, CA_d, alpha=0.12, color='gray')

for ax, ttl, yl in zip(axes1,
    ["CA vs Mach", "CN vs Mach (α=0°)", "xcp vs Mach [m od nosa]"],
    ["CA [-]", "CN [-]", "xcp [m]"]):
    ax.axvline(1., color='gray', lw=0.8, ls=':')
    ax.set_xlabel("Mach [-]"); ax.set_ylabel(yl); ax.set_title(ttl)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
fig1.savefig("compare_aero_coefficients.png", dpi=150, bbox_inches="tight")
plt.close(); print("\nZapisano: compare_aero_coefficients.png")

# ---- Wykres 2: CN i CA vs alpha -----------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(f"CN i CA vs α — {CASE_NAME}", fontsize=12, fontweight='bold')

mach_sel   = [0.6, 1.4, 2.0]
mach_clrs  = ['#2dd4bf', '#fb923c', '#a78bfa']

for name, aero in models.items():
    ls = lsmap[name]
    alpha_deg = np.degrees(aero.alpha_table)
    for m, mc in zip(mach_sel, mach_clrs):
        if m < aero.mach_table[0] or m > aero.mach_table[-1]: continue
        j = np.argmin(np.abs(aero.mach_table - m))
        lbl = f"{name}  Ma={m}"
        axes2[0].plot(alpha_deg, aero.CN_table[:, j], color=mc, ls=ls, lw=1.8, label=lbl)
        axes2[1].plot(alpha_deg, aero.CA_table[:, j], color=mc, ls=ls, lw=1.8, label=lbl)

for ax, yl, ttl in zip(axes2, ["CN [-]","CA [-]"], ["CN vs α","CA vs α"]):
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel("α [°]"); ax.set_ylabel(yl); ax.set_title(ttl)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

plt.tight_layout()
fig2.savefig("compare_aero_vs_alpha.png", dpi=150, bbox_inches="tight")
plt.close(); print("Zapisano: compare_aero_vs_alpha.png")

# ---- Wykres 3: trajektorie -----------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5))
fig3.suptitle(f"Trajektorie — El={mission.elevation}° Az={mission.azimuth}°",
              fontsize=12, fontweight='bold')

for name, result in results.items():
    c, ls = colors[name], lsmap[name]
    lf = geo.ned_to_lf(result)
    a_sound = np.array([atm.at(max(0., float(-result.z[i]))).speed_of_sound
                        for i in range(len(result.t))])
    axes3[0].plot(lf["x_lf"]/1000, lf["alt"], color=c, ls=ls, lw=2, label=name)
    axes3[1].plot(result.t, result.speed,       color=c, ls=ls, lw=2, label=name)
    axes3[2].plot(result.t, result.speed/a_sound, color=c, ls=ls, lw=2, label=name)

axes3[0].set(xlabel="Zasieg [km]", ylabel="Wysokosc [m]", title="Plaszczyzna strzalu (LF)")
axes3[0].set_ylim(bottom=0)
axes3[1].set(xlabel="Czas [s]", ylabel="Predkosc [m/s]", title="Predkosc vs czas")
axes3[2].set(xlabel="Czas [s]", ylabel="Mach [-]", title="Liczba Macha vs czas")
axes3[2].axhline(1., color='gray', lw=0.8, ls=':')
for ax in axes3: ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
fig3.savefig("compare_trajectory.png", dpi=150, bbox_inches="tight")
plt.close(); print("Zapisano: compare_trajectory.png")

# ---- Podsumowanie --------------------------------------------------------
lines = ["="*60, "POROWNANIE MODELI AERODYNAMICZNYCH",
         f"Konfiguracja : {CASE_NAME}",
         f"El={mission.elevation}°  Az={mission.azimuth}°  Cmq={CMQ}", "="*60,
         f"{'Parametr':<26} {'Barrowman':>13} {'Missile DATCOM':>15} {'Delta':>8}",
         "-"*62]

metrics = {}
for name, result in results.items():
    lf = geo.ned_to_lf(result)
    a_sound = np.array([atm.at(max(0.,float(-result.z[i]))).speed_of_sound
                        for i in range(len(result.t))])
    metrics[name] = {
        "Zasieg [km]":       lf["x_lf"][-1]/1000,
        "Wys max [m]":       float(lf["alt"].max()),
        "Czas lotu [s]":     float(result.t[-1]),
        "V max [m/s]":       float(result.speed.max()),
        "V upadek [m/s]":    float(result.speed[-1]),
        "Ma max [-]":        float((result.speed/a_sound).max()),
    }

for p in metrics["Barrowman"]:
    vb = metrics["Barrowman"][p]
    vd = metrics["Missile DATCOM"][p]
    dp = (vd-vb)/abs(vb)*100 if abs(vb)>1e-6 else 0.
    lines.append(f"{p:<26} {vb:>13.2f} {vd:>15.2f} {dp:>+7.1f}%")

lines.append("="*60)
summary = "\n".join(lines)
print("\n"+summary)
with open("compare_summary.txt","w",encoding="utf-8") as f: f.write(summary)
print("\nWszystkie pliki gotowe.")
