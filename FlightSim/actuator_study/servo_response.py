"""
servo_response.py
=================
Standalone study of a 2nd-order actuator driven by a sinusoidal input.
No dependency on the flight model — pure ODE integration + plots.

Edit the parameters below and run:
    python servo_response.py
    python servo_response.py --show
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

# Input signal: delta_cmd(t) = AMPLITUDE * sin(OMEGA * t)
AMPLITUDE = 5.0       # [deg]
OMEGA     = 10.0      # [rad/s]

# Actuator 2nd order: delta_ddot = wn^2*(cmd - delta) - 2*zeta*wn*delta_dot
WN   = 60.0           # natural frequency [rad/s]
ZETA = 0.7            # damping ratio [-]

# Limits
RATE_LIMIT = 400.0    # max actuator rate [deg/s]
POS_LIMIT  = 15.0     # max actuator deflection [deg]

# Simulation
T_MAX = 2.0           # [s]
DT    = 0.001         # output step [s]

# ============================================================


def cmd(t):
    return AMPLITUDE * np.sin(OMEGA * t)


def derivatives(t, x):
    delta, delta_dot = x
    c = cmd(t)

    ddot = WN**2 * (c - delta) - 2.0 * ZETA * WN * delta_dot

    # rate limit
    if delta_dot > RATE_LIMIT:
        ddot = min(ddot, 0.0)
    elif delta_dot < -RATE_LIMIT:
        ddot = max(ddot, 0.0)

    # position limit
    if delta >= POS_LIMIT and delta_dot > 0.0:
        delta_dot = 0.0
        ddot = min(ddot, 0.0)
    elif delta <= -POS_LIMIT and delta_dot < 0.0:
        delta_dot = 0.0
        ddot = max(ddot, 0.0)

    return [delta_dot, ddot]


t_eval = np.arange(0.0, T_MAX, DT)
sol = solve_ivp(derivatives, (0.0, T_MAX), [0.0, 0.0],
                t_eval=t_eval, max_step=DT, method="RK45")

t = sol.t
delta = sol.y[0]
delta_dot = sol.y[1]
command = np.array([cmd(ti) for ti in t])

# ---- Plots ---- #
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

ax[0].plot(t, command, "k--", lw=1.2, label="command")
ax[0].plot(t, delta, "tab:blue", lw=1.5, label="actuator")
ax[0].set_ylabel("deflection [deg]")
ax[0].legend()
ax[0].grid(alpha=0.3)
ax[0].set_title(f"2nd-order servo: wn={WN}, zeta={ZETA}, "
                f"rate_lim={RATE_LIMIT} deg/s, pos_lim={POS_LIMIT} deg\n"
                f"input: {AMPLITUDE} deg @ {OMEGA} rad/s")

ax[1].plot(t, delta_dot, "tab:orange", lw=1.5)
ax[1].axhline(RATE_LIMIT, color="r", ls=":", lw=1, label=f"+/- {RATE_LIMIT} deg/s")
ax[1].axhline(-RATE_LIMIT, color="r", ls=":", lw=1)
ax[1].set_ylabel("rate [deg/s]")
ax[1].legend()
ax[1].grid(alpha=0.3)

# phase lag and gain
ax[2].plot(t, command - delta, "tab:green", lw=1.5)
ax[2].set_ylabel("error [deg]")
ax[2].set_xlabel("time [s]")
ax[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("servo_response.png", dpi=130, bbox_inches="tight")
print(f"Saved: servo_response.png")

if "--show" in sys.argv:
    plt.show()
plt.close(fig)
