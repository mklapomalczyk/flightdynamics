"""
debug_damp.py — diagnostyka czasu tlumienia alpha
Uruchom z katalogu FlightSim: python debug_damp.py
"""
import sys, numpy as np
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
import matplotlib.pyplot as plt

CASE_NAME = "rocket_36mm_malarakieta_base"
cfg     = load_config(f"configurations/{CASE_NAME}.yaml")
mission = MissionConfig.from_yaml("missions/mission_01.yaml")
geo     = GeoModule(mission)
aero    = get_aero_model(CASE_NAME, method="missile_datcom")
initial = build_initial_state(mission)

mass = build_mass_model(cfg)
prop = build_propulsion(cfg)
geom = build_geometry(cfg)
geom.xcp = float(aero.xcp_table[len(aero.alpha_table)//2, 0])

fm = ForceModel6DOF(
    atmosphere=create_atmosphere("ISA"),
    mass_model=mass, aero_model=aero,
    gravity=create_gravity("constant"),
    geometry=geom, propulsion=prop,
)
result = run_simulation_6dof(fm, initial, t_max=60., z_ground=0.,
                             max_step=0.1, rtol=1e-3, atol=1e-5)

# Burnout
tp = cfg.propulsion.thrust_profile
t_burnout = float(max(t for t, f in tp if f > 0.0))
print(f"t_burnout = {t_burnout}s")

alpha_deg = np.abs(np.degrees(result.alpha))
t_arr     = result.t

idx_start = int(np.searchsorted(t_arr, t_burnout + 0.1))
alpha_post = alpha_deg[idx_start:]
t_post     = t_arr[idx_start:]

print(f"alpha_post: min={alpha_post.min():.4f} max={alpha_post.max():.4f} len={len(alpha_post)}")

# Lokalne maksima
from scipy.signal import find_peaks
peaks_rel, props = find_peaks(alpha_post, height=0.05)
print(f"Liczba szczytow: {len(peaks_rel)}")
if len(peaks_rel) > 0:
    print("Pierwsze 5 szczytow:")
    for pk in peaks_rel[:5]:
        print(f"  t={t_post[pk]:.3f}s  alpha={alpha_post[pk]:.4f}°")

# Oblicz damp_time
damp_time = float('nan')
if len(peaks_rel) >= 2:
    A1 = alpha_post[peaks_rel[0]]
    t1 = t_post[peaks_rel[0]]
    print(f"\nA1={A1:.4f}°  t1={t1:.3f}s  prog=0.5*A1={0.5*A1:.4f}°")
    for pk in peaks_rel[1:]:
        print(f"  pk: t={t_post[pk]:.3f}s  A={alpha_post[pk]:.4f}° {'<= prog' if alpha_post[pk]<=0.5*A1 else ''}")
        if alpha_post[pk] <= 0.5 * A1:
            damp_time = t_post[pk] - t1
            break
print(f"\ndamp_time = {damp_time:.3f}s")

# Wykres
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_post, alpha_post, 'b-', lw=1.2, label='|alpha|')
if len(peaks_rel) > 0:
    ax.plot(t_post[peaks_rel], alpha_post[peaks_rel],
            'ro', ms=6, label='szczyty')
    ax.axhline(0.5*alpha_post[peaks_rel[0]], color='orange',
               ls='--', lw=1, label='0.5*A1')
ax.set_xlabel("Czas [s]"); ax.set_ylabel("|Alpha| [deg]")
ax.set_title(f"Diagnostyka tlumienia — damp_time={damp_time:.3f}s")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("debug_damp.png", dpi=120)
plt.close()
print("Zapisano: debug_damp.png")

# Test nowej metody - dopasowanie eksponenty
print("\n=== Nowa metoda — dopasowanie eksponenty ===")
from scipy.signal import find_peaks as _fp
idx_s = int(np.searchsorted(t_arr, t_burnout + 0.1))
a_post = alpha_deg[idx_s:]
t_post2 = t_arr[idx_s:]
peaks_r, _ = _fp(a_post, height=0.05)

if len(peaks_r) >= 4:
    t_pk = t_post2[peaks_r].astype(float)
    A_pk = a_post[peaks_r].astype(float)
    mask = A_pk > 1e-6
    b, a_coef = np.polyfit(t_pk[mask], np.log(A_pk[mask]), 1)
    tau = -1.0/b if b < 0 else float('nan')
    damp_exp = np.log(0.5)/b if b < 0 else float('nan')
    print(f"tau={tau:.2f}s  damp_time(exp)={damp_exp:.2f}s  b={b:.4f}")

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(t_post2, a_post, 'b-', lw=1, alpha=0.6)
    ax2.plot(t_pk, A_pk, 'ro', ms=5, label='szczyty')
    t_fit = np.linspace(t_pk[0], t_pk[-1], 200)
    ax2.plot(t_fit, np.exp(a_coef + b*t_fit), 'g--', lw=2, label=f'exp fit (tau={tau:.1f}s)')
    ax2.axhline(0.5*A_pk[0], color='orange', ls='--', lw=1, label='0.5*A1')
    ax2.set_xlabel("Czas [s]"); ax2.set_ylabel("|Alpha| [deg]")
    ax2.set_title(f"Fit eksponenty — damp_time={damp_exp:.2f}s")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("debug_damp_exp.png", dpi=120)
    plt.close()
    print("Zapisano: debug_damp_exp.png")
