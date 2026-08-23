import os
import sys, pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIGS_DIR  = PROJECT_ROOT / "configurations"
RUNS_DIR     = PROJECT_ROOT / "datcom_runs"
sys.path.insert(0, str(PROJECT_ROOT))
from models.aerodynamics import TableAero

def get_aero_model(case_name, Cmq=-20.0, force_rerun=False, method="barrowman", compute_cmq=True,
                    compute_roll=False):
    # compute_roll domyslnie False: $RLLO ROLLQ=1.0,$ (missile_datcom_generator.py)
    # NIE dziala z ta wersja Missile DATCOM (Rev 3/99) -- zweryfikowane na
    # locie 19: przebieg z tym namelist produkuje IDENTYCZNA strukture
    # outputu co zwykly przebieg pelnej konfiguracji (te same sekcje
    # STATIC AERODYNAMICS/DERIVATIVES PER DEGREE z CNA/CMA/CYB/CLNB/CLLB),
    # BEZ zadnej sekcji roll-damping/CLLP -- DATCOM po cichu ignoruje ten
    # namelist (bledna nazwa zmiennej lub $RLLO nie istnieje w tej wersji).
    # Zostawiono `compute_roll=True` jako opcje (kod generatora/parsera
    # gotowy, gdyby ktos zweryfikowal poprawna skladnie z podrecznika
    # Missile DATCOM-97 -- nie mozna tego sprawdzic w tym sandboxie, brak
    # renderera PDF), ale domyslnie wylaczone, zeby nie marnowac trzeciego
    # przebiegu DATCOM ktory i tak nic nie daje.
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
        return _get_missile_datcom(cfg, case_name, run_dir, Cmq, force_rerun, compute_cmq, compute_roll)
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

def _get_missile_datcom(cfg, case_name, run_dir, Cmq, force_rerun, compute_cmq=True,
                         compute_roll=True):
    pkl_path      = run_dir / "aero_table_missile.pkl"
    out_path      = run_dir / "datcom.out"
    out_body_path = run_dir / "datcom_body.out"
    out_roll_path = run_dir / "datcom_roll.out"

    if pkl_path.exists() and not force_rerun:
        print(f"[aero] Missile DATCOM: laduje cache {pkl_path.name}")
        with open(pkl_path, "rb") as f:
            cached = pickle.load(f)
        _check_moment_reference(cached, cfg, pkl_path)
        return cached

    if (out_path.exists() and not force_rerun
            and (not compute_cmq or out_body_path.exists())
            and (not compute_roll or out_roll_path.exists())):
        print(f"[aero] Missile DATCOM: parsuje istniejace pliki out")
        return _parse_and_cache(out_path, pkl_path, Cmq,
                                out_body_path if compute_cmq else None,
                                out_roll_path if compute_roll else None, cfg)
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

    # Przebieg 3: pelna konfiguracja + $RLLO (prawdziwy Clp z DATCOM) --
    # opcjonalny. Bez tego Clp_table jest ZAWSZE analitycznym przyblizeniem
    # z CNA_fins (patrz _compute_cmq_table), NIGDY prawdziwym tlumieniem
    # obrotowym z DATCOM -- mimo ze czytnik (missile_datcom_reader.py) i
    # _parse_and_cache juz obsluguja wczytanie $RLLO CLLP, ta sciezka nigdy
    # nie byla wywolywana w pipeline uzywanym do walidacji lotow. Znaleziono
    # podczas analizy lotu 19: telemetria pokazuje predkosc obrotowa
    # nasycona na suficie czujnika (2000 deg/s) przez ~8s i nadal
    # 1300-2000 deg/s w fazie balistycznej, znaczaco powyzej modelu
    # (~850-900 deg/s) -- analityczny Clp jest glownym podejrzanym.
    if compute_roll:
        inp_roll_path = run_dir / "for005_roll.dat"
        print(f"[aero] Missile DATCOM: przebieg 3 — pelna konfiguracja + $RLLO (Clp)")
        generate_missile_datcom_input(cfg, inp_roll_path, body_only=False, roll_only=True)
        run_missile_datcom(inp_roll_path, run_dir,
                           output_filename="datcom_roll.out")

    return _parse_and_cache(out_path, pkl_path, Cmq,
                            out_body_path if compute_cmq else None,
                            out_roll_path if compute_roll else None, cfg)
def _check_moment_reference(aero_obj, cfg, src):
    """
    Pilnuje, ze wspolczynniki momentu sa znormalizowane przez SREDNICE — czyli
    przez te sama dlugosc, ktora model podstawia licząc moment (geom.d_ref).
    Chroni przed cichym uzyciem cache sprzed zmiany LREF (bylo: dlugosc kadluba,
    czyli momenty 18.3x za male dla 70mm).
    """
    if cfg is None:
        return
    # Furtka WYLACZNIE do zrobienia migawki "przed" (compare_validation.py
    # --save-baseline): pozwala jeszcze raz policzyc walidacje na STARYCH
    # danych, zeby bylo z czym porownac wynik po poprawce. Wlacza sie tylko
    # jawnie zmienna srodowiskowa i glosno o sobie mowi — nigdy nie jest
    # domyslna, bo blad jest cichy i wart 18x w momencie.
    if os.environ.get("FLIGHTSIM_ALLOW_STALE_LREF") == "1":
        lref_dbg = getattr(aero_obj, "lref_ref", None)
        print(f"[aero] *** UWAGA: FLIGHTSIM_ALLOW_STALE_LREF=1 — pomijam kontrole "
              f"dlugosci odniesienia dla {getattr(src, 'name', src)} "
              f"(lref={lref_dbg}). Wyniki maja momenty pochylajace/odchylajace "
              f"~18x za male. Uzywac TYLKO do migawki 'przed'.")
        return
    d_body = float(cfg.body.diameter)
    lref = getattr(aero_obj, "lref_ref", None)
    if lref is None:
        raise ValueError(
            f"\n{src}: cache nie zawiera informacji o dlugosci odniesienia "
            f"(pochodzi sprzed poprawki LREF).\n"
            f"Usun ten plik i przelicz aero od nowa (force_rerun=True), "
            f"albo uruchom: python check_moment_reference.py")
    if abs(lref - d_body) > 0.05 * max(d_body, 1e-9):
        raise ValueError(
            f"\n{src}: wspolczynniki znormalizowane przez LREF={lref} m, a model "
            f"mnozy przez srednice={d_body} m ({lref/d_body:.1f}x rozbieznosci).\n"
            f"Przegeneruj deck i uruchom DATCOM ponownie (force_rerun=True).")


def _parse_and_cache(out_path, pkl_path, Cmq, out_body_path=None,
                      out_roll_path=None, cfg=None):
    import numpy as np
    from datcom_io.missile_datcom_reader import parse_missile_datcom_output, missile_datcom_to_table_aero

    # Pełna konfiguracja
    result = parse_missile_datcom_output(out_path)
    table  = missile_datcom_to_table_aero(result)
    alpha_rad = np.deg2rad(table["alpha_deg"])

    # --- Kontrola dlugosci odniesienia momentu ---------------------------- #
    # DATCOM zwraca CM = M/(q*SREF*LREF); model liczy moment jako
    # q*S_ref*d_ref*Cm z d_ref = SREDNICA. Obie wielkosci MUSZA byc te same,
    # inaczej momenty sa przeskalowane o LREF/srednica (dla 70mm bylo to 18.3x).
    # Generator ustawia teraz LREF = srednica, ale STARE pliki datcom*.out
    # powstaly przy LREF = dlugosc kadluba i po cichu dawalyby zle momenty —
    # dlatego sprawdzamy to przy kazdym parsowaniu.
    if (cfg is not None and result.cases
            and os.environ.get("FLIGHTSIM_ALLOW_STALE_LREF") != "1"):
        lref_out = float(result.cases[0].lref)
        d_body = float(cfg.body.diameter)
        if abs(lref_out - d_body) > 0.05 * max(d_body, 1e-9):
            raise ValueError(
                f"\n{out_path}: LREF z DATCOM = {lref_out} m, a srednica kadluba "
                f"= {d_body} m (stosunek {lref_out/d_body:.1f}x).\n"
                f"Model mnozy wspolczynniki momentu przez srednice, wiec ten plik "
                f"dalby momenty {lref_out/d_body:.1f}x za duze/male.\n"
                f"To najpewniej WYNIK ZE STAREGO DECKA (LREF = dlugosc kadluba). "
                f"Przegeneruj i uruchom DATCOM ponownie:\n"
                f"    python MAIN.py            (albo run_control_datcom.py --run)\n"
                f"Szczegoly: python check_moment_reference.py")

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
        CA_base_table = table.get("CA_base"),  # opor denny -> korekta powered/coast
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
      d        = srednica kadluba (JEDYNA dlugosc odniesienia tutaj)

      Cmq = -2 * CNA_fins_rad * ((x_fins - xcg) / d)²
      Clp = -n_fins * CNA_fins_rad * (r_fin_mid / d)²

    gdzie CNA_fins_rad = CNA_fins * (180/pi) — przeliczenie z 1/deg na 1/rad.

    LREF z pliku DATCOM celowo NIE wystepuje w tych wzorach — model normalizuje
    tlumienie srednica, wiec mieszanie obu dlugosci bylo zrodlem bledu 18.4x
    (patrz komentarz przy wzorach nizej).

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

    # CNA_fins [1/deg] → [1/rad]
    CNA_fins_rad = CNA_fins * (180.0 / np.pi)

    # ---- Normalizacja: WSZYSTKO przez SREDNICE, nie przez lref ----------- #
    # Model liczy tlumienie jako Cmq * (q*d_ref/(2V)) * q_dyn*S_ref*d_ref
    # (models/aerodynamics.py:205, forces/force_model6.py:261), czyli oczekuje
    # pochodnych znormalizowanych PRZEZ SREDNICE. Wyprowadzenie:
    #
    #   pitch: dM = -(q*S_fin*CNa*(q_rate*arm/V))*arm,  q_hat = q_rate*d/(2V)
    #          => Cmq = -2 * CNA_fins * ((x_fins - xcg)/d)^2
    #   roll:  dL = -(q*S_fin*CNa_1fin*(p*r/V))*r,      p_hat = p*d/(2V)
    #          => Clp = -2 * n * CNa_1fin * (r/d)^2
    #          a poniewaz CNA_fins to wklad CALEGO zestawu w sile normalna,
    #          w ukladzie krzyzowym nosza ja 2 z 4 pletw (CNa_1fin ~ CNA_fins/2),
    #          podczas gdy toczenie tlumia wszystkie 4:
    #          => Clp = -2 * 4 * (CNA_fins/2) * (r/d)^2 = -n * CNA_fins * (r/d)^2
    #
    # Wczesniej byly tu czynniki (lref/d) i (d/lref) z lref WCZYTANYM Z PLIKU
    # DATCOM. Przy LREF = dlugosc kadluba dawaly Cmq i Clp 18.4x za male; po
    # poprawce LREF (lref == d) wychodza poprawnie, ale tylko PRZYPADKIEM —
    # kazda przyszla zmiana LREF po cichu przeskalowalaby tlumienie.
    # Dlatego lref nie wystepuje juz w tych wzorach w ogole.
    arm_d = (x_fins - xcg) / d
    Cmq_table = -2.0 * CNA_fins_rad * (arm_d ** 2)

    # r_fin_mid = r_body + span/2  [m]
    r_fin_mid = cfg.body.diameter / 2.0 + fin.span / 2.0
    Clp_table = -int(fin.count) * CNA_fins_rad * (r_fin_mid / d) ** 2

    print(f"[aero] Clp(alpha,Mach) obliczone analitycznie z CNA_fins")
    print(f"       Clp @ alpha=0, Mach[0]: {Clp_table[len(Clp_table)//2, 0]:.4f}")

    return Cmq_table, Clp_table
