import sys, pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIGS_DIR  = PROJECT_ROOT / "configurations"
RUNS_DIR     = PROJECT_ROOT / "datcom_runs"
sys.path.insert(0, str(PROJECT_ROOT))
from models.aerodynamics import TableAero

def get_aero_model(case_name, Cmq=-20.0, force_rerun=False, method="barrowman", compute_cmq=True):
    run_dir = RUNS_DIR / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = CONFIGS_DIR / f"{case_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Brak pliku konfiguracji: {yaml_path}")
    from datcom_io.config_reader import load_config
    cfg = load_config(yaml_path)
    if method == "barrowman":
        return _get_barrowman(cfg, case_name, run_dir, Cmq, force_rerun)
    elif method == "missile_datcom":
        return _get_missile_datcom(cfg, case_name, run_dir, Cmq, force_rerun, compute_cmq)
    else:
        raise ValueError(f"Nieznana metoda: '{method}'. Uzyj 'barrowman' lub 'missile_datcom'.")

def _get_barrowman(cfg, case_name, run_dir, Cmq, force_rerun):
    pkl_path = run_dir / "aero_table_barrowman.pkl"
    if pkl_path.exists() and not force_rerun:
        print(f"[aero] Barrowman: laduje cache {pkl_path.name}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    print(f"[aero] Barrowman: obliczam dla {case_name}")
    from datcom_io.barrowman_aero import get_aero_barrowman
    aero = get_aero_barrowman(case_name, Cmq=Cmq, force_rerun=force_rerun)
    with open(pkl_path, "wb") as f:
        pickle.dump(aero, f)
    print(f"[aero] Zapisano cache: {pkl_path.name}")
    return aero

def _get_missile_datcom(cfg, case_name, run_dir, Cmq, force_rerun, compute_cmq=True):
    pkl_path      = run_dir / "aero_table_missile.pkl"
    out_path      = run_dir / "datcom.out"
    out_body_path = run_dir / "datcom_body.out"

    if pkl_path.exists() and not force_rerun:
        print(f"[aero] Missile DATCOM: laduje cache {pkl_path.name}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    if out_path.exists() and not force_rerun and (not compute_cmq or out_body_path.exists()):
        print(f"[aero] Missile DATCOM: parsuje istniejace pliki out")
        return _parse_and_cache(out_path, pkl_path, Cmq,
                                out_body_path if compute_cmq else None,
                                None, cfg)
    from datcom_io.missile_datcom_generator import generate_missile_datcom_input
    from datcom_io.missile_datcom_runner    import run_missile_datcom

    # Przebieg 1: pełna konfiguracja (kadłub + płetwy)
    inp_path = run_dir / "for005.dat"
    print(f"[aero] Missile DATCOM: przebieg 1 — pełna konfiguracja")
    generate_missile_datcom_input(cfg, inp_path, body_only=False)
    run_missile_datcom(inp_path, run_dir)

    # Przebieg 2: tylko kadłub (do obliczenia Cmq) — opcjonalny
    if compute_cmq:
        inp_body_path = run_dir / "for005_body.dat"
        print(f"[aero] Missile DATCOM: przebieg 2 — tylko kadłub (Cmq)")
        generate_missile_datcom_input(cfg, inp_body_path, body_only=True)
        run_missile_datcom(inp_body_path, run_dir,
                           output_filename="datcom_body.out")

    return _parse_and_cache(out_path, pkl_path, Cmq,
                            out_body_path if compute_cmq else None,
                            None, cfg)
def _parse_and_cache(out_path, pkl_path, Cmq, out_body_path=None,
                      out_roll_path=None, cfg=None):
    import numpy as np
    from datcom_io.missile_datcom_reader import parse_missile_datcom_output, missile_datcom_to_table_aero

    # Pełna konfiguracja
    result = parse_missile_datcom_output(out_path)
    table  = missile_datcom_to_table_aero(result)
    alpha_rad = np.deg2rad(table["alpha_deg"])

    # Oblicz tabelę Cmq i Clp z dwóch przebiegów
    Cmq_table = None
    Clp_table = None
    if out_body_path is not None and cfg is not None:
        if not out_body_path.exists():
            print(f"[aero] Brak {out_body_path.name} — Cmq/Clp nie zostaną obliczone z geometrii")
        else:
            result_tables = _compute_cmq_table(
                result_full = result,
                out_body    = out_body_path,
                cfg         = cfg,
            )
            if result_tables is not None:
                Cmq_table, Clp_table = result_tables
                print(f"[aero] Cmq(alpha,Mach) obliczone z dwóch przebiegów DATCOM")
                print(f"       Cmq @ alpha=0, Mach[0]: {Cmq_table[len(alpha_rad)//2, 0]:.2f}")

    # Wczytaj Clp z przebiegu roll (nadpisuje analityczne jeśli dostępne)
    if out_roll_path is not None and Path(out_roll_path).exists():
        from datcom_io.missile_datcom_reader import parse_missile_datcom_output, missile_datcom_to_table_aero
        try:
            result_roll = parse_missile_datcom_output(out_roll_path)
            table_roll  = missile_datcom_to_table_aero(result_roll)
            if np.any(table_roll["CLLP"] != 0):
                Clp_table = table_roll["CLLP"]
                print(f"[aero] Clp(alpha,Mach) z przebiegu $RLLO")
                print(f"       Clp @ alpha=0, Mach[0]: {Clp_table[len(alpha_rad)//2, 0]:.4f}")
            else:
                print(f"[aero] Ostrzeżenie: CLLP=0 w datcom_roll.out — brak $RLLO w outputcie")
        except Exception as e:
            print(f"[aero] Ostrzeżenie: nie można wczytać roll run: {e}")

    # CLL z tabeli DATCOM (z cant_angle w $FINSET)
    CLL_table = table.get("CLL")
    if CLL_table is not None and np.any(CLL_table != 0):
        print(f"[aero] CLL(alpha,Mach) wczytane z DATCOM (cant_angle)")

    aero = TableAero(
        alpha_table = alpha_rad,
        mach_table  = table["mach"],
        CA_table    = table["CA"],
        CN_table    = table["CN"],
        Cm_table    = table["CM"],
        xcp_table   = table["XCP"],
        Cmq         = Cmq,
        Cmq_table   = Cmq_table,
        Clp_table   = Clp_table,
        CYB_table   = table.get("CYB"),
        CLL_table   = CLL_table,
        xcg_ref     = float(table["xcg"]),   # xcg uzyte w DATCOM do obliczenia Cm_table
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(aero, f)
    print(f"[aero] Zapisano cache: {pkl_path.name}")
    return aero


def _compute_cmq_table(result_full, out_body, cfg):
    """
    Oblicza tabelę Cmq(alpha, Mach) z dwóch przebiegów DATCOM.

    Metoda:
      CNA_fins = CNA_full - CNA_body   [1/deg]
      x_fins   = środek cięciwy płetw wzdłuż osi [m od nosa]
      xcg      = xcg z pliku DATCOM
      lref     = lref z pliku DATCOM

      Cmq = -2 * CNA_fins_rad * ((x_fins - xcg) / lref)²
            * lref / d_ref

    gdzie CNA_fins_rad = CNA_fins * (180/pi) — przeliczenie z 1/deg na 1/rad.

    Returns
    -------
    np.ndarray (N_alpha, N_mach) lub None
    """
    import numpy as np
    from datcom_io.missile_datcom_reader import parse_missile_datcom_output, missile_datcom_to_table_aero

    try:
        result_body = parse_missile_datcom_output(out_body)
        if not result_body.cases:
            return None
        table_body = missile_datcom_to_table_aero(result_body)
    except Exception as e:
        print(f"[aero] Ostrzeżenie: nie można wczytać body-only run: {e}")
        return None

    table_full = missile_datcom_to_table_aero(result_full)

    # Wspólna siatka alpha/Mach — musi być identyczna
    alpha_full = table_full["alpha_deg"]
    mach_full  = table_full["mach"]
    alpha_body = table_body["alpha_deg"]
    mach_body  = table_body["mach"]

    # Interpoluj CNA_body na siatkę pełnej konfiguracji
    # (body-only może mieć mniej punktów Mach)
    # Interpoluj CNA_body na siatkę pełnej konfiguracji
    # (body-only może mieć inną liczbę punktów Mach)
    same_grid = (len(alpha_full) == len(alpha_body) and
                 len(mach_full)  == len(mach_body)  and
                 np.allclose(alpha_full, alpha_body, atol=0.1) and
                 np.allclose(mach_full,  mach_body,  atol=0.01))
    if same_grid:
        CNA_body = table_body["CNA"]
    else:
        print(f"[aero] Interpolacja CNA_body: "
              f"({len(alpha_body)}α, {len(mach_body)}M) → "
              f"({len(alpha_full)}α, {len(mach_full)}M)")
        # Interpolacja + ekstrapolacja przez powtórzenie wartości brzegowej
        # fill_value=None (ekstrapolacja liniowa) daje asymetrię dla dużych α
        # Zamiast tego: przytnij Mach do zakresu body przed interpolacją
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (np.deg2rad(alpha_body), mach_body),
            table_body["CNA"],
            method="linear", bounds_error=False,
            fill_value=None,   # ekstrapolacja liniowa wewnątrz alpha
        )
        # Przytnij Mach do zakresu dostępnego w body-only
        mach_clipped = np.clip(mach_full, mach_body[0], mach_body[-1])
        pts = np.array([[np.deg2rad(a), m]
                        for a in alpha_full for m in mach_clipped])
        CNA_body = interp(pts).reshape(len(alpha_full), len(mach_full))
        if not np.allclose(mach_clipped, mach_full):
            n_clipped = np.sum(mach_full > mach_body[-1])
            print(f"[aero] CNA_body: {n_clipped} punktów Mach poza zakresem "
                  f"body-only ({mach_body[-1]:.2f}) — użyto wartości brzegowej")

    CNA_full = table_full["CNA"]   # [1/deg]
    CNA_fins = CNA_full - CNA_body  # [1/deg]

    # Pozycja aerodynamiczna płetw
    xcg  = table_full["xcg"]
    lref = table_full["lref"]
    d    = cfg.body.diameter

    # x_fins = pozycja środka cięciwy pierwszego zestawu płetw
    if cfg.fins:
        fin   = cfg.fins[0]
        import math
        sweep_x = fin.span * math.tan(math.radians(fin.sweep_le))
        xle_t   = fin.position + sweep_x
        xte_r   = fin.position + fin.root_chord
        xte_t   = xle_t + fin.tip_chord
        # Środek MAC wzdłuż osi
        x_fins  = (fin.position + xte_r) / 2.0 * 0.5 + (xle_t + xte_t) / 2.0 * 0.5
    else:
        return None

    # arm = (x_fins - xcg) / lref — znormalizowane
    arm = (x_fins - xcg) / lref

    # CNA_fins [1/deg] → [1/rad]
    CNA_fins_rad = CNA_fins * (180.0 / np.pi)

    # Cmq = -2 * CNA_fins_rad * ((x_fins - xcg) / lref)² * lref/d
    Cmq_table = -2.0 * CNA_fins_rad * (arm ** 2) * (lref / d)

    # Clp = -n_fins * CNA_fins_rad * (r_fin_mid / d)² * (d / lref)
    # r_fin_mid = r_body + span/2  [m]
    r_fin_mid = cfg.body.diameter / 2.0 + fin.span / 2.0
    Clp_table = (-int(fin.count) * CNA_fins_rad *
                 (r_fin_mid / d) ** 2 *
                 (d / lref))

    print(f"[aero] Clp(alpha,Mach) obliczone analitycznie z CNA_fins")
    print(f"       Clp @ alpha=0, Mach[0]: {Clp_table[len(Clp_table)//2, 0]:.4f}")

    return Cmq_table, Clp_table
