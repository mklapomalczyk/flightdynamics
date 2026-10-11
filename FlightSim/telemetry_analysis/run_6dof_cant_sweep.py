"""
run_6dof_cant_sweep.py
========================
Uruchamia PRAWDZIWY symulator 6DOF (core/solver6.run_simulation_6dof,
ten sam pipeline co MAIN.py) dla kilku wartosci kata zaklinowania
(cant_angle) pletw i porownuje apogeum z danymi polowymi.

Nie modyfikuje MAIN.py ani configurations/*.yaml — kat cant_angle z
YAML jest nadpisywany TYLKO w pamieci (RocketConfig wczytany przez
load_config(), potem zmieniany na kopii per-run).

MissileDATCOM.exe jest binarka Windows i nie da sie jej tu ponownie
odpalic (brak Wine) — wiec nie mozna "od zera" przeliczyc DATCOM dla
nowego cant_angle. Zamiast tego:

  1. Wczytujemy cache aero_table_missile.pkl (DATCOM policzony dla
     RZECZYWISTEGO cant_angle z YAML — sprawdzane i wypisywane).
  2. Statyczny wklad cant_angle w opor osiowy liczymy analitycznie wg
     Fleemana (ta sama formula CN_fin co w datcom_io/barrowman.py,
     uzyta przy alpha=cant_angle — sila normalna pletwy odchylonej o
     staly kat, zrzutowana na os ciala: dCA = CN_fin(cant) * sin(cant)).
     To NIE zalezy od predkosci obrotowej — czysto geometryczny,
     staly efekt zaklinowania (zgodnie z prośbą: "Drag itself will be
     increased due to deflection, but its static and not connected to
     the rolling rate").
  3. Od tabeli CA z DATCOM odejmujemy wklad juz "wbudowany" (przy
     cant_angle z YAML) i dodajemy wklad dla docelowego cant_angle z
     przebiegu sweepu. Dla cant_angle == YAML, korekta = 0 (tozsamosc
     z cache).
  4. Opcjonalnie (--spin-drag): dodaje DRUGI, dynamiczny czlon —
     ZALOZENIE/HIPOTEZA, nie zwalidowany model — opor wywolany
     wirowaniem, wyliczony z bilansu mocy: moc rozpraszana przez
     tlumienie toczenia (Clp*p, juz liczone w force_model6.py) jest
     scigana z energii translacyjnej: P_roll = |MA_roll_damp * p|,
     D_spin = P_roll / V, dodawane jako dodatkowe du/dt w podklasie
     ForceModel6DOF (oryginalny plik forces/force_model6.py NIE jest
     modyfikowany).

Uzycie:
    python run_6dof_cant_sweep.py
    python run_6dof_cant_sweep.py --cants 0.0 0.6 1.2 1.6
    python run_6dof_cant_sweep.py --spin-drag
"""

import sys
import csv
import copy
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir
from run_all_flights import read_configs, check_data_exists

from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_mass_model, build_propulsion, build_geometry, build_initial_state
from datcom_io.barrowman import FleemanCalculator
from core.state6 import State6DOF, IDX_U
from core.solver6 import run_simulation_6dof
from core.quaternion import aero_angles
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from forces.force_model6 import ForceModel6DOF
from geo.geographic import MissionConfig
from aero import get_aero_model


# --------------------------------------------------------------------------
def cant_axial_increment(cfg, mach_grid, cant_override_deg=None):
    """Suma po wszystkich zestawach pletw: CN_fin(cant) * sin(cant), jako
    funkcja Mach. cant_override_deg=None -> uzyj fin.cant_angle z YAML
    (czyli to, co JUZ jest wbudowane w cache DATCOM)."""
    calc = FleemanCalculator(cfg)
    delta = np.zeros(len(mach_grid), dtype=float)
    for fin in cfg.fins:
        cant_deg = fin.cant_angle if cant_override_deg is None else cant_override_deg
        cant_rad = np.radians(cant_deg)
        for j, m in enumerate(mach_grid):
            CN_fin, _ = calc._CN_single_fin(fin, cant_rad, float(m))
            delta[j] += CN_fin * np.sin(cant_rad)
    return delta


def build_cant_aero_model(aero_model, cfg, target_cant_deg):
    """Kopia aero_model z CA_table skorygowana o roznice statycznego wkladu
    cant_angle (docelowy minus ten, juz wbudowany przez DATCOM)."""
    mach_grid = aero_model.mach_table
    baseline_delta = cant_axial_increment(cfg, mach_grid, cant_override_deg=None)
    target_delta   = cant_axial_increment(cfg, mach_grid, cant_override_deg=target_cant_deg)
    correction = target_delta - baseline_delta   # (n_mach,)

    aero_new = copy.deepcopy(aero_model)
    aero_new.CA_table = aero_model.CA_table + correction[None, :]
    return aero_new, correction


# --------------------------------------------------------------------------
class SpinDragForceModel6DOF(ForceModel6DOF):
    """Podklasa testowa — dodaje HIPOTETYCZNY opor wywolany wirowaniem,
    z bilansu mocy tlumienia toczenia (Clp). Nie modyfikuje oryginalnego
    force_model6.py; dziala tylko gdy spin_drag_enabled=True."""

    def __init__(self, *args, spin_drag_enabled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.spin_drag_enabled = spin_drag_enabled

    def derivatives(self, t, x):
        dxdt = super().derivatives(t, x)
        if not self.spin_drag_enabled:
            return dxdt

        state = State6DOF.from_numpy(x[:13])
        if not self.launcher.is_disabled() and len(x) > 13:
            rail_dist = float(x[13])
        else:
            rail_dist = 0.0
        if self.launcher.on_rail(rail_dist):
            return dxdt

        u, v, w, p = state.u, state.v, state.w, state.p
        speed = float(np.sqrt(u**2 + v**2 + w**2))
        if speed < 1.0:
            return dxdt

        ms  = self.mass_model.at(t)
        atm = self.atmosphere.at(-state.z)
        mach  = atm.mach(speed)
        q_dyn = 0.5 * atm.density * speed**2
        alpha, beta = aero_angles(u, v, w)

        Clp_rad = 0.0
        if hasattr(self.aero, 'Clp_table') and self.aero.Clp_table is not None:
            Clp_rad = self.aero._interp(self.aero.Clp_table, alpha, mach)
        elif hasattr(self.aero, 'Clp'):
            Clp_rad = float(self.aero.Clp)

        MA_roll_damp = (Clp_rad * (p * self.geom.d_ref / (2.0 * speed)) *
                         q_dyn * self.geom.S_ref * self.geom.d_ref)
        P_roll = abs(MA_roll_damp * p)
        D_spin = P_roll / speed

        dxdt[IDX_U] -= D_spin / ms.mass
        return dxdt


# --------------------------------------------------------------------------
def actual_apogee_mean(base, nose):
    rows = read_configs(base)
    flights = [r["fno"] for r in rows if r["nose"] == nose and check_data_exists(base, r["fno"])]
    csv_path = Path(base) / "results" / "trajectory_closure_summary.csv"
    vals = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["flight_no"]) in flights:
                    vals.append(float(row["h_apo_actual_m"]))
    return (float(np.mean(vals)), flights) if vals else (None, flights)


# --------------------------------------------------------------------------
def run_one(case_name, cant_deg, spin_drag):
    mission = MissionConfig.from_yaml(str(Path(get_data_dir()).parent / "missions" / "mission_01.yaml"))
    cfg  = load_config(str(Path(get_data_dir()).parent / "configurations" / f"{case_name}.yaml"))
    mass = build_mass_model(cfg)
    prop = build_propulsion(cfg)
    geom = build_geometry(cfg)

    aero_model = get_aero_model(case_name, method="missile_datcom", force_rerun=False)
    aero_cant, correction = build_cant_aero_model(aero_model, cfg, cant_deg)

    atm      = create_atmosphere("ISA")
    launcher = LauncherConfig(L_rail=3.0)

    force_model = SpinDragForceModel6DOF(
        atmosphere=atm, mass_model=mass, aero_model=aero_cant,
        gravity=create_gravity("constant"), geometry=geom,
        propulsion=prop, launcher=launcher, spin_drag_enabled=spin_drag,
    )

    initial_state = build_initial_state(mission)
    result = run_simulation_6dof(force_model, initial_state, t_max=100, dt_output=0.01)

    h = -result.z
    i_apo = int(np.argmax(h))
    return dict(
        cant_deg=cant_deg, h_apo=float(h[i_apo]), t_apo=float(result.t[i_apo]),
        V_max=float(np.max(result.speed)), status=result.status,
        correction_mean=float(np.mean(correction)),
    )


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="6DOF sweep po cant_angle (statyczny opor + opcjonalny opor od wirowania)")
    parser.add_argument("--case", default="rocket_70mm_baseline")
    parser.add_argument("--nose", default="ostra")
    parser.add_argument("--cants", nargs="*", type=float, default=[0.0, 0.6, 1.2, 1.6])
    parser.add_argument("--spin-drag", action="store_true",
                         help="dodaj hipotetyczny opor od wirowania (bilans mocy Clp)")
    args = parser.parse_args()

    base = get_data_dir()
    yaml_path = Path(base).parent / "configurations" / f"{args.case}.yaml"
    cfg_check = load_config(str(yaml_path))
    yaml_cants = [f"{fin.name}={fin.cant_angle:.3f}deg" for fin in cfg_check.fins]
    print(f"Konfiguracja: {args.case}  (cant_angle z YAML, juz wbudowany w cache DATCOM: {yaml_cants})")
    print(f"Tryb opor od wirowania: {'WLACZONY (hipoteza, niezwalidowana)' if args.spin_drag else 'wylaczony'}")

    h_actual, flights = actual_apogee_mean(base, args.nose)
    if h_actual is not None:
        print(f"Rzeczywiste apogeum GPS (mean, nos '{args.nose}', loty {flights}): {h_actual:.1f} m")
    else:
        print(f"Brak danych rzeczywistych apogeum dla nosa '{args.nose}'.")

    rows = []
    print(f"\n{'cant [deg]':>10} {'h_apo [m]':>10} {'t_apo [s]':>10} {'V_max [m/s]':>12} {'status':>10}")
    for cant in args.cants:
        r = run_one(args.case, cant, args.spin_drag)
        rows.append(r)
        print(f"{r['cant_deg']:10.2f} {r['h_apo']:10.1f} {r['t_apo']:10.2f} {r['V_max']:12.1f} {r['status']:>10}")

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "spindrag" if args.spin_drag else "static"
    csv_path = out_dir / f"cant_sweep_{args.case}_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cant_deg", "h_apo", "t_apo", "V_max", "status", "correction_mean"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nZapisano: {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    cants = [r["cant_deg"] for r in rows]
    h_apos = [r["h_apo"] for r in rows]
    ax.plot(cants, h_apos, 'o-', ms=8, lw=1.5, color='tab:blue',
            label=f"6DOF predykcja ({tag})")
    if h_actual is not None:
        ax.axhline(h_actual, color='tab:red', ls='--', lw=1.5,
                   label=f"rzeczywiste apogeum GPS (mean, '{args.nose}')")
    ax.set_xlabel("Cant angle [deg]")
    ax.set_ylabel("Apogeum [m]")
    ax.set_title(f"6DOF: apogeum vs cant_angle — {args.case} ({tag})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / f"cant_sweep_{args.case}_{tag}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
