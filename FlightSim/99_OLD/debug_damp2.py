"""
debug_damp2.py — diagnostyka czasu tlumienia dla n=4, chord=160 vs 200
Uruchom z katalogu FlightSim: python debug_damp2.py
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
from scipy.signal import find_peaks
import tempfile, os, yaml, shutil

CASE_NAME = "rocket_36mm_malarakieta_base"
CONFIGS = [
    {"chord": 0.140, "count": 4, "label": "n=4 c=140mm"},
    {"chord": 0.160, "count": 4, "label": "n=4 c=160mm"},
    {"chord": 0.180, "count": 4, "label": "n=4 c=180mm"},
    {"chord": 0.200, "count": 4, "label": "n=4 c=200mm"},
    {"chord": 0.220, "count": 4, "label": "n=4 c=220mm"},
]
COLORS = plt.cm.viridis(np.linspace(0, 0.85, len(CONFIGS)))

mission = MissionConfig.from_yaml("missions/mission_01.yaml")
geo     = GeoModule(mission)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Diagnostyka czasu tlumienia — n=4, rozne chords",
             fontsize=11, fontweight="bold")

for cfg_d, clr in zip(CONFIGS, COLORS):
    # Zbuduj tymczasowy YAML
    with open(f"configurations/{CASE_NAME}.yaml",
              encoding='utf-8', errors='replace') as f:
        raw = yaml.safe_load(f.read())
    L = float(raw["body"]["length"])
    chord = cfg_d["chord"]
    for fin in raw.get("fins", []):
        fin["root_chord"] = chord
        fin["tip_chord"]  = chord
        fin["count"]      = cfg_d["count"]
        fin["position"]   = round(L - chord, 5)
        fin["cant_angle"] = 0.0
    case_tag = f"{CASE_NAME}_n{cfg_d['count']}_c{int(chord*1000)}"
    shutil.copy(f"configurations/{CASE_NAME}.yaml",
                f"configurations/{case_tag}.yaml")
    with open(f"configurations/{case_tag}.yaml", "w") as f:
        yaml.dump(raw, f)

    aero = get_aero_model(case_tag, method="missile_datcom", force_rerun=False)
    cfg  = load_config(f"configurations/{case_tag}.yaml")
    os.unlink(f"configurations/{case_tag}.yaml")

    mass = build_mass_model(cfg)
    geom = build_geometry(cfg)
    geom.xcp = float(aero.xcp_table[len(aero.alpha_table)//2, 0])
    initial  = build_initial_state(mission)

    fm = ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"),
        mass_model=mass, aero_model=aero,
        gravity=create_gravity("constant"),
        geometry=geom, propulsion=build_propulsion(cfg),
    )
    result = run_simulation_6dof(fm, initial, t_max=60., z_ground=0.,
                                 max_step=0.02, rtol=1e-5, atol=1e-7)

    tp = cfg.propulsion.thrust_profile
    t_burn = float(max(t for t, f in tp if f > 0.0))
    alpha_abs = np.abs(np.degrees(result.alpha))
    t_arr     = result.t
    idx_start = int(np.searchsorted(t_arr, t_burn + 0.1))
    alpha_post = alpha_abs[idx_start:]
    t_post     = t_arr[idx_start:]

    peaks_rel, _ = find_peaks(alpha_post, height=0.05)
    damp_time = float('nan')
    if len(peaks_rel) >= 2:
        A1 = alpha_post[peaks_rel[0]]
        t1 = t_post[peaks_rel[0]]
        for pk in peaks_rel[1:]:
            if alpha_post[pk] <= 0.5 * A1:
                damp_time = t_post[pk] - t1
                break

    print(f"{cfg_d['label']}: damp_time={damp_time:.3f}s  "
          f"A1={alpha_post[peaks_rel[0]]:.3f}°  "
          f"n_peaks={len(peaks_rel)}")

    axes[0].plot(t_post, alpha_post, color=clr, lw=1.2,
                 label=f"{cfg_d['label']} ({damp_time:.2f}s)")
    if len(peaks_rel) > 0:
        axes[0].plot(t_post[peaks_rel], alpha_post[peaks_rel],
                     'o', color=clr, ms=4)

axes[0].axhline(0, color='gray', lw=0.5, ls=':')
axes[0].set_xlabel("Czas [s]"); axes[0].set_ylabel("|Alpha| [deg]")
axes[0].set_title("Obwiednia |alpha| po burnout")
axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

# Wykres obwiedni (laczace szczyty)
for cfg_d, clr in zip(CONFIGS, COLORS):
    axes[1].set_xlabel("Numer szczytu")
    axes[1].set_ylabel("Amplituda [deg]")
    axes[1].set_title("Amplitudy kolejnych szczytow")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("debug_damp2.png", dpi=130, bbox_inches="tight")
plt.close()
print("Zapisano: debug_damp2.png")
