import numpy as np
import matplotlib.pyplot as plt
from aero import get_aero_model

aero = get_aero_model("rocket_70mm_baseline", method="missile_datcom", force_rerun=True)

if aero.Cmq_table is not None:
    alpha_deg = np.degrees(aero.alpha_table)
    mach      = aero.mach_table

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cmq(α, Mach) — obliczone z dwóch przebiegów DATCOM", fontweight='bold')

    # Cmq vs Mach dla wybranych alpha
    colors = ['#4f8ef7', '#f97316', '#34d399']#, '#a78bfa', '#f43f5e']
    for i, a in enumerate([-10, 0, 5]):
        idx = np.argmin(np.abs(alpha_deg - a))
        axes[0].plot(mach, aero.Cmq_table[idx, :],
                     color=colors[i % len(colors)], lw=2, label=f'α={a}°')
    axes[0].axhline(0, color='gray', lw=0.5, ls=':')
    axes[0].axvline(1., color='gray', lw=0.8, ls='--', alpha=0.5)
    axes[0].set_xlabel("Mach [-]"); axes[0].set_ylabel("Cmq [-]")
    axes[0].set_title("Cmq vs Mach"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Cmq vs alpha dla wybranych Mach
    mach_sel = [mach[0], mach[len(mach)//2], mach[-1]]
    for m, c in zip(mach_sel, ['#4f8ef7', '#f97316', '#34d399']):
        j = np.argmin(np.abs(mach - m))
        axes[1].plot(alpha_deg, aero.Cmq_table[:, j],
                     color=c, lw=2, label=f'Ma={m:.1f}')
    axes[1].axhline(0, color='gray', lw=0.5, ls=':')
    axes[1].set_xlabel("α [°]"); axes[1].set_ylabel("Cmq [-]")
    axes[1].set_title("Cmq vs α"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("cmq_table.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("Brak tabeli Cmq — użyto stałej wartości:", aero.Cmq)