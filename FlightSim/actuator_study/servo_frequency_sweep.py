"""
servo_frequency_sweep.py
========================
Frequency sweep of a 2nd-order actuator: fixed amplitude sinusoidal input,
increasing omega. Measures steady-state output amplitude.

Output: gain (A_out/A_in) and phase vs omega — a numerical Bode diagram
that includes the effect of rate and position limits.

    python servo_frequency_sweep.py
    python servo_frequency_sweep.py --show
"""

import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# USER PARAMETERS — edit these
# ============================================================

AMPLITUDE = 5.0       # input amplitude [deg]

# Frequency sweep range
OMEGA_MIN = 1.0       # [rad/s]
OMEGA_MAX = 200.0     # [rad/s]
N_POINTS  = 40        # number of frequencies to test

# Actuator parameters
WN   = 60.0           # natural frequency [rad/s]
ZETA = 0.7            # damping ratio [-]

# Limits
RATE_LIMIT = 400.0    # [deg/s]
POS_LIMIT  = 15.0     # [deg]

# ============================================================


def run_one(omega, amp):
    T = 2.0 * np.pi / omega
    n_settle = max(int(5 * 2 * ZETA / omega * omega) + 3, 6)
    t_max = n_settle * T
    t_measure = t_max - 3 * T

    def deriv(t, x):
        delta, delta_dot = x
        c = amp * np.sin(omega * t)
        ddot = WN**2 * (c - delta) - 2.0 * ZETA * WN * delta_dot

        if delta_dot > RATE_LIMIT:
            ddot = min(ddot, 0.0)
        elif delta_dot < -RATE_LIMIT:
            ddot = max(ddot, 0.0)

        if delta >= POS_LIMIT and delta_dot > 0.0:
            delta_dot = 0.0
            ddot = min(ddot, 0.0)
        elif delta <= -POS_LIMIT and delta_dot < 0.0:
            delta_dot = 0.0
            ddot = max(ddot, 0.0)

        return [delta_dot, ddot]

    sol = solve_ivp(deriv, (0.0, t_max), [0.0, 0.0],
                    method="RK45", rtol=1e-6, atol=1e-8,
                    max_step=T / 40)

    mask = sol.t >= t_measure
    delta_m = sol.y[0][mask]
    t_m = sol.t[mask]

    a_out = (np.max(delta_m) - np.min(delta_m)) / 2.0

    # phase: find time shift of output peak vs input peak
    cmd_m = amp * np.sin(omega * t_m)
    i_cmd_peak = np.argmax(cmd_m)
    i_out_peak = np.argmax(delta_m)
    dt_lag = t_m[i_out_peak] - t_m[i_cmd_peak]
    # normalize to [-T/2, 0]
    dt_lag = dt_lag % T
    if dt_lag > T / 2:
        dt_lag -= T
    phase_deg = dt_lag / T * 360.0

    return a_out, phase_deg


omegas = np.geomspace(OMEGA_MIN, OMEGA_MAX, N_POINTS)
gains = np.zeros(N_POINTS)
phases = np.zeros(N_POINTS)

print(f"Sweeping {N_POINTS} frequencies from {OMEGA_MIN} to {OMEGA_MAX} rad/s ...")
for i, w in enumerate(omegas):
    a_out, ph = run_one(w, AMPLITUDE)
    gains[i] = a_out / AMPLITUDE
    phases[i] = ph

print("Done.")

# Analytical (no limits) for comparison
r = omegas / WN
gain_analytical = 1.0 / np.sqrt((1 - r**2)**2 + (2 * ZETA * r)**2)
phase_analytical = -np.degrees(np.arctan2(2 * ZETA * r, 1 - r**2))

# ---- Plots ---- #
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax[0].plot(omegas, gains, "o-", color="tab:blue", ms=4, lw=1.5,
           label="numerical (with limits)")
ax[0].plot(omegas, gain_analytical, "k--", lw=1, label="analytical (no limits)")
ax[0].axhline(1.0, color="gray", ls=":", lw=0.8)
ax[0].axhline(1.0 / np.sqrt(2), color="tab:red", ls=":", lw=0.8,
              label="-3 dB")
ax[0].axvline(WN, color="tab:orange", ls=":", lw=0.8, label=f"wn={WN}")
ax[0].set_ylabel("gain  A_out / A_in")
ax[0].set_xscale("log")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, which="both")
ax[0].set_title(f"Frequency response: wn={WN}, zeta={ZETA}, A={AMPLITUDE} deg, "
                f"rate_lim={RATE_LIMIT} deg/s, pos_lim={POS_LIMIT} deg")

ax[1].plot(omegas, phases, "o-", color="tab:blue", ms=4, lw=1.5,
           label="numerical")
ax[1].plot(omegas, phase_analytical, "k--", lw=1, label="analytical")
ax[1].axvline(WN, color="tab:orange", ls=":", lw=0.8)
ax[1].set_ylabel("phase lag [deg]")
ax[1].set_xlabel("omega [rad/s]")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("servo_frequency_sweep.png", dpi=130, bbox_inches="tight")
print(f"Saved: servo_frequency_sweep.png")

if "--show" in sys.argv:
    plt.show()
plt.close(fig)
