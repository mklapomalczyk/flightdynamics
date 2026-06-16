"""
visualize_rocket.py
====================
Wizualizator geometrii rakiety z pliku YAML.
Generuje rzut boczny i widok od tyłu z wymiarami.

Użycie:
    python visualize_rocket.py
    python visualize_rocket.py rocket_70mm_baseline
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from datcom_io.config_reader import load_config


def build_profile(cfg):
    """Buduje profil boczny (x, r) od nosa do ogona."""
    d    = cfg.body.diameter
    r    = d / 2.0
    l    = cfg.body.length
    l_N  = cfg.body.nose.length
    bt   = cfg.body.boattail
    l_bt = bt.length   if bt else 0.0
    r_bt = bt.diameter / 2.0 if bt else r
    l_cy = l - l_N - l_bt

    # Nos — ogiva lub stożek
    n = 80
    x_nose = np.linspace(0, l_N, n)
    nose_type = cfg.body.nose.type.lower()
    if nose_type == "ogive":
        r_nose = r * np.sqrt(np.maximum(1 - (1 - x_nose / l_N) ** 2, 0))
    else:  # cone lub inne
        r_nose = r * (x_nose / l_N)

    # Walec
    x_cyl = np.array([l_N, l_N + l_cy])
    r_cyl = np.array([r, r])

    # Boattail
    if l_bt > 0:
        x_bt = np.array([l_N + l_cy, l_N + l_cy + l_bt])
        r_bt_arr = np.array([r, r_bt])
        xs = np.concatenate([x_nose, x_cyl, x_bt])
        rs = np.concatenate([r_nose, r_cyl, r_bt_arr])
    else:
        xs = np.concatenate([x_nose, x_cyl])
        rs = np.concatenate([r_nose, r_cyl])

    return xs, rs, {
        "l": l, "r": r, "l_N": l_N,
        "l_cy": l_cy, "l_bt": l_bt, "r_bt": r_bt if bt else r,
    }


def draw_side_view(ax, cfg):
    """Rysuje rzut boczny."""
    xs, rs, dims = build_profile(cfg)
    d = cfg.body.diameter
    l = cfg.body.length

    # Wypełnienie korpusu
    ax.fill_between(xs,  rs, -rs, color='#c8d8f0', alpha=0.85, zorder=2)
    # Kontur górny i dolny
    ax.plot(xs,  rs, 'k-', lw=1.5, zorder=3)
    ax.plot(xs, -rs, 'k-', lw=1.5, zorder=3)
    # Zamknięcie ogona
    ax.plot([xs[-1], xs[-1]], [-rs[-1], rs[-1]], 'k-', lw=1.5, zorder=3)

    # Płetwy
    for fin in cfg.fins:
        _draw_fin_side(ax, fin, dims["r"], color='#5080b0')

    # XCG
    xcg = cfg.mass.xcg_ref
    ax.axvline(xcg, color='red', lw=1.2, ls='--', zorder=4, alpha=0.8)
    ax.text(xcg + 0.01, dims["r"] * 1.15, f"xcg={xcg:.3f}m",
            color='red', fontsize=8, va='bottom')

    # Oś rakiety
    ax.axhline(0, color='gray', lw=0.6, ls=':', zorder=1)

    # Wymiary — długość całkowita
    _dim_arrow(ax, 0, l, -dims["r"] * 1.6, f"L = {l*1000:.0f} mm")

    # Nos
    _dim_arrow(ax, 0, dims["l_N"], -dims["r"] * 2.1, f"l_N = {dims['l_N']*1000:.0f} mm")

    # Średnica — strzałka pionowa
    xd = dims["l_N"] + dims["l_cy"] * 0.4
    ax.annotate("", xy=(xd, dims["r"]), xytext=(xd, -dims["r"]),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
    ax.text(xd + 0.015, 0, f"d = {d*1000:.0f} mm",
            ha='left', va='center', fontsize=8, color='#444')

    # Boattail
    if dims["l_bt"] > 0:
        x_bt_start = dims["l_N"] + dims["l_cy"]
        ax.text(x_bt_start + dims["l_bt"] / 2,
                -dims["r_bt"] * 1.8,
                f"boattail\nd={dims['r_bt']*2*1000:.0f}mm",
                ha='center', fontsize=7, color='#555')

    # Wymiary płetwy
    if cfg.fins:
        import math
        fin = cfg.fins[0]
        sweep_x = fin.span * math.tan(math.radians(fin.sweep_le))
        xle_r = fin.position
        xte_r = xle_r + fin.root_chord
        xle_t = xle_r + sweep_x
        xte_t = xle_t + fin.tip_chord
        r_fin = dims["r"]

        # Cięciwa — jedna etykieta C= gdy Cr==Ct, dwie gdy różne
        ax.annotate("", xy=(xte_r, r_fin*0.6), xytext=(xle_r, r_fin*0.6),
                    arrowprops=dict(arrowstyle="<->", color="#888", lw=0.9))
        if abs(fin.root_chord - fin.tip_chord) < 1e-4:
            ax.text((xle_r+xte_r)/2, r_fin*0.42,
                    f"C={fin.root_chord*1000:.0f}mm",
                    ha="center", fontsize=7, color="#666")
        else:
            ax.text((xle_r+xte_r)/2, r_fin*0.42,
                    f"Cr={fin.root_chord*1000:.0f}mm",
                    ha="center", fontsize=7, color="#666")
            ax.annotate("", xy=(xte_t, r_fin+fin.span*1.05), xytext=(xle_t, r_fin+fin.span*1.05),
                        arrowprops=dict(arrowstyle="<->", color="#888", lw=0.9))
            ax.text((xle_t+xte_t)/2, r_fin+fin.span*1.17,
                    f"Ct={fin.tip_chord*1000:.0f}mm",
                    ha="center", fontsize=7, color="#666")

        # Kąt skosu — tylko gdy != 0
        if abs(fin.sweep_le) > 0.5:
            ax.text(xle_r + sweep_x*0.5, r_fin + fin.span*0.55,
                    f"Λ={fin.sweep_le:.0f}°",
                    ha="center", fontsize=7, color="#5080b0", style="italic")


    ax.set_aspect('equal')
    ax.set_xlim(-l * 0.05, l * 1.12)
    ax.set_ylim(-dims["r"] * 2.8, dims["r"] * 2.6)
    ax.set_xlabel("x od nosa [m]", fontsize=9)
    ax.set_ylabel("r [m]", fontsize=9)
    ax.set_title("Rzut boczny", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)


def _draw_fin_side(ax, fin, r_body, color):
    """Rysuje jedną płetwę w rzucie bocznym na podstawie sweep_le."""
    import math
    sweep_deg = getattr(fin, 'sweep_le', 0.0)
    sweep_x   = fin.span * math.tan(math.radians(sweep_deg))

    xle_r = fin.position                  # LE nasady
    xte_r = xle_r + fin.root_chord        # TE nasady
    xle_t = xle_r + sweep_x              # LE końcówki
    xte_t = xle_t + fin.tip_chord        # TE końcówki

    xs = [xle_r, xle_t, xte_t, xte_r, xle_r]
    # Górna płetwa
    ys = [r_body, r_body + fin.span, r_body + fin.span, r_body, r_body]
    ax.fill(xs,  ys,           color=color, alpha=0.7, zorder=2)
    ax.plot(xs,  ys,           'k-', lw=1.2, zorder=3)
    # Dolna płetwa (symetria)
    ax.fill(xs, [-y for y in ys], color=color, alpha=0.7, zorder=2)
    ax.plot(xs, [-y for y in ys], 'k-', lw=1.2, zorder=3)


def _dim_arrow(ax, x0, x1, y, label):
    """Strzałka wymiaru poziomego."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=1.0))
    ax.text((x0 + x1) / 2, y - abs(y) * 0.08, label,
            ha='center', va='top', fontsize=8, color='#333')


def draw_rear_view(ax, cfg):
    """Rysuje widok od tyłu (przekrój)."""
    fins = cfg.fins
    r    = cfg.body.diameter / 2.0
    bt   = cfg.body.boattail
    r_bt = bt.diameter / 2.0 if bt else r

    # Obrys korpusu (zewnętrzny)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(r * np.cos(theta), r * np.sin(theta),
            color='#c8d8f0', zorder=2, alpha=0.85)
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', lw=1.5, zorder=3)

    # Obrys boattail (jeśli jest)
    if bt and r_bt < r:
        ax.fill(r_bt * np.cos(theta), r_bt * np.sin(theta),
                color='#9ab0d0', zorder=3, alpha=0.9)
        ax.plot(r_bt * np.cos(theta), r_bt * np.sin(theta), 'k--', lw=1.0, zorder=4)

    # Otwór dyszy (przybliżony)
    r_nozzle = r_bt * 0.55
    ax.fill(r_nozzle * np.cos(theta), r_nozzle * np.sin(theta),
            color='#444', zorder=5, alpha=0.8)

    # Płetwy — rzut od tyłu
    for fin in fins:
        _draw_fin_rear(ax, fin, r, cfg.body.diameter)

    # Osie
    ax.axhline(0, color='gray', lw=0.5, ls=':', zorder=1)
    ax.axvline(0, color='gray', lw=0.5, ls=':', zorder=1)

    # Wymiar rozpiętości
    if fins:
        r_tip = r + fins[0].span
        ax.annotate("", xy=(r_tip, 0), xytext=(-r_tip, 0),
                    arrowprops=dict(arrowstyle='<->', color='#333', lw=1.0))
        ax.text(0, -r_tip * 1.12,
                f"rozpiętość = {r_tip*2*1000:.0f} mm",
                ha='center', va='top', fontsize=8, color='#333')


    lim = (r + (fins[0].span if fins else 0)) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel("y [m]", fontsize=9)
    ax.set_ylabel("z [m]", fontsize=9)
    ax.set_title("Widok od tyłu", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)


def _draw_fin_rear(ax, fin, r_body, d_body):
    """Rysuje płetwę w widoku od tyłu."""
    count = int(fin.count)
    span  = fin.span
    # Szerokość płetwy w widoku od tyłu — przybliżenie prostokątem
    w_avg = (fin.root_chord + fin.tip_chord) / 2.0 * 0.25  # grubość wizualna

    for i in range(count):
        angle = i * 2 * np.pi / count
        # Środek płetwy
        cx = np.cos(angle) * (r_body + span / 2)
        cy = np.sin(angle) * (r_body + span / 2)
        # Prostokąt prostopadle do promienia
        perp = np.array([-np.sin(angle), np.cos(angle)])
        radial = np.array([np.cos(angle), np.sin(angle)])

        corners = [
            cx * np.array([1,0]) + cy * np.array([0,1]) + w_avg * perp - span/2 * radial,
            cx * np.array([1,0]) + cy * np.array([0,1]) - w_avg * perp - span/2 * radial,
            cx * np.array([1,0]) + cy * np.array([0,1]) - w_avg * perp + span/2 * radial,
            cx * np.array([1,0]) + cy * np.array([0,1]) + w_avg * perp + span/2 * radial,
        ]
        # Uproszczone — użyj patch prostokątny wzdłuż promienia
        xs = [r_body * np.cos(angle), (r_body + span) * np.cos(angle)]
        ys = [r_body * np.sin(angle), (r_body + span) * np.sin(angle)]
        lw = getattr(fin, "thickness", 0.002) * 300

        ax.plot(xs, ys, color='#5080b0', lw=lw * 20,
                solid_capstyle='butt', zorder=2, alpha=0.7)
        ax.plot(xs, ys, 'k-', lw=1.0, zorder=3)


def main():
    case_name = sys.argv[1] if len(sys.argv) > 1 else "rocket_70mm_baseline"
    yaml_path = Path("configurations") / f"{case_name}.yaml"

    print(f"Ładowanie: {yaml_path}")
    cfg = load_config(yaml_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(
        f"Geometria rakiety — {case_name}\n"
        f"L={cfg.body.length*1000:.0f}mm  "
        f"d={cfg.body.diameter*1000:.0f}mm  "
        f"l_N={cfg.body.nose.length*1000:.0f}mm  "
        f"nos={cfg.body.nose.type}  "
        f"płetwy={int(cfg.fins[0].count) if cfg.fins else 0}×",
        fontsize=10, fontweight='bold'
    )

    draw_side_view(axes[0], cfg)
    draw_rear_view(axes[1], cfg)

    plt.tight_layout()
    out = f"rocket_geometry_{case_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out}")


if __name__ == "__main__":
    main()
