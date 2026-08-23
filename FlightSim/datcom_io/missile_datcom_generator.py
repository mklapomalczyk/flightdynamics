"""
datcom_io/missile_datcom_generator.py
======================================
Generator pliku wejściowego dla Missile DATCOM (Rev 3/99).

Jednostki: metry (DIM M na początku pliku).
Format: namelisty Fortran ($NAMELIST ... ,$)

Obsługuje:
  - $FLTCON  — warunki lotu (Mach, alpha)
  - $REFQ    — dane referencyjne (XCG, SREF, LREF)
  - $AXIBOD  — geometria kadłuba osiowosymetrycznego
  - $FINSETn — do 2 zestawów płetw (n=1,2)
  - $DEFLCT  — kąty wychylenia płetw (domyślnie 0)
"""

from __future__ import annotations
import math
import os
from pathlib import Path
from datcom_io.config_reader import RocketConfig

# Szerokosc karty (kolumn) dla starego, sztywnego formatu Fortran uzywanego
# przez Missile DATCOM. Linie dluzsze sa po cichu ucinane/blednie parsowane
# przez czytnik DATCOM-a (objaw: "** BLANK CARD - IGNORED" / "MISSING
# NAMELIST TERMINATION ADDED" w datcom.out, mimo poprawnej skladni pliku
# .inp) -- stad koniecznosc dzielenia dlugich tablic (np. ALPHA z szerokim
# zakresem katow) na kolejne karty kontynuacji.
DATCOM_CARD_WIDTH = 80


def _wrap_namelist_array(var_name: str, values, value_fmt, indent: str,
                          max_width: int = DATCOM_CARD_WIDTH) -> list[str]:
    """
    Formatuje tablice namelist Fortran (np. ALPHA=1.,2.,3.,...) na jedna
    lub wiecej fizycznych linii, dzielac dlugie tablice na karty
    kontynuacji wg formatu Missile DATCOM:

        NALPHA=20., ALPHA=0.,2.,4.,...,18.,20.,
        ALPHA(12)=22.,24.,...,52.,

    tj. kontynuacja powtarza nazwe zmiennej z indeksem (1-based) pierwszej
    wartosci na tej karcie w nawiasie, zamiast proby kontynuowania listy
    bez etykiety (co dla dlugich linii DATCOM po prostu ucina/gubi).

    Nie dodaje koncowego "$" (terminator namelist) -- to robi wywolujacy
    dla ostatniej zmiennej w danym bloku $NAMELIST.
    """
    tokens = [f"{value_fmt(v)}," for v in values]
    lines = []
    i = 0
    n = len(values)
    while i < n:
        prefix = f"{indent}{var_name}=" if i == 0 else f"{indent}{var_name}({i + 1})="
        line = prefix
        started = False
        while i < n:
            tok = tokens[i]
            if started and len(line) + len(tok) > max_width:
                break
            line += tok
            i += 1
            started = True
        lines.append(line)
    return lines


def _effective_fin_sets(cfg: RocketConfig, include_controls: bool = True):
    """
    Laczy pletwy pasywne (cfg.fins) i powierzchnie sterowe (cfg.control_surfaces)
    w jedna liste zestawow $FINSETn.

    DATCOM nie zna pojecia "powierzchnia sterowa" — kazdy zestaw paneli to
    $FINSETn, a sterowanie zadaje sie przez DELTAn. Zestawy MUSZA byc
    uporzadkowane od nosa do ogona, wiec canardy (x~0.15) staja sie $FINSET1,
    a pletwy ogonowe (x~1.16) $FINSET2.

    Zwraca (lista [(fin_like, is_control)], lista 1-based indeksow sterowych).
    Indeksy zwracamy jawnie, zeby nigdzie nie zakladac na sztywno, ze
    "canardy to zestaw 1".
    """
    items = [(f, False) for f in cfg.fins]
    if include_controls:
        items += [(cs, True) for cs in getattr(cfg, "control_surfaces", [])]

    # DATCOM przyjmuje maksymalnie 4 zestawy ($FINSET1..4).
    items.sort(key=lambda it: float(it[0].position))
    items = items[:4]

    ctrl_indices = [i for i, (_, is_ctrl) in enumerate(items, start=1) if is_ctrl]
    return items, ctrl_indices


def generate_missile_datcom_input(
    cfg: RocketConfig,
    output_path: str | Path,
    mach_list:  list[float] | None = None,
    alpha_list: list[float] | None = None,
    xcg_override: float | None = None,
    body_only:    bool = False,
    roll_only:    bool = False,
    power_on:     bool = False,
    include_controls: bool = True,
    emit_gam:     bool = False,
    delta_cases:  list[dict] | None = None,
) -> Path:
    """
    Generuje plik .inp dla Missile DATCOM na podstawie RocketConfig.

    Parameters
    ----------
    cfg : RocketConfig
        Konfiguracja rakiety z YAML.
    output_path : str | Path
        Ścieżka do pliku wyjściowego.
    mach_list : list[float], optional
        Lista liczb Macha. Domyślnie z cfg.flight_conditions.
    alpha_list : list[float], optional
        Lista kątów natarcia [deg]. Domyślnie symetryczna.
    xcg_override : float, optional
        Nadpisuje XCG z YAML [m].

    Returns
    -------
    Path  Ścieżka do wygenerowanego pliku.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mach_list is None:
        mach_list = cfg.flight_conditions.mach
    if alpha_list is None:
        if cfg.flight_conditions.alpha:
            alpha_list = cfg.flight_conditions.alpha
        else:
            alpha_list = [-16., -8., 0., 8., 16.]

    xcg = xcg_override if xcg_override is not None else cfg.mass.xcg_ref
    d   = cfg.body.diameter
    l   = cfg.body.length
    S   = 3.14159265 * (d / 2)**2   # pole przekroju
    l_N = cfg.body.nose.length

    lines = []

    # Jednostki
    lines.append("DIM M")
    lines.append("")

    # --- $FLTCON ----------------------------------------------------------
    # MACH/ALPHA dzielone na karty kontynuacji (ALPHA(n)=...) gdy tablica
    # jest za dluga na jedna linie -- patrz _wrap_namelist_array powyzej
    # (DATCOM po cichu ucina/gubi wartosci na zbyt dlugich liniach zamiast
    # zglosic blad, co dawalo NALPHA niezgodne z rzeczywista lista i
    # "MISSING NAMELIST TERMINATION ADDED" w datcom.out).
    nmach = len(mach_list)
    nalpha = len(alpha_list)

    lines.append(f" $FLTCON  NALPHA={nalpha}.,NMACH={nmach}.,")

    mach_lines = _wrap_namelist_array("MACH", mach_list, lambda m: f"{m:.4f}", "          ")
    lines.extend(mach_lines)

    alpha_lines = _wrap_namelist_array("ALPHA", alpha_list, lambda a: f"{a:.1f}", "          ")
    alpha_lines[-1] += "$"
    lines.extend(alpha_lines)
    lines.append("")

    # --- $REFQ ------------------------------------------------------------
    # LREF = SREDNICA kadluba, nie jego dlugosc.
    #
    # DATCOM zwraca CM jako wielkosc bezwymiarowa: CM = M / (q*SREF*LREF), wiec
    # zeby odzyskac moment trzeba pomnozyc przez TE SAMA LREF. Model 6DOF liczy
    # moment jako q_dyn*S_ref*d_ref*Cm z d_ref = SREDNICA (models/aerodynamics.py),
    # czyli oczekuje wspolczynnikow znormalizowanych przez srednice — to zarazem
    # DOMYSLNA wartosc LREF wg podrecznika ($REFQ: "Default is maximum body
    # diameter") i standardowa konwencja pociskowa.
    #
    # Wczesniej podawano tu dlugosc kadluba (1.285 m przy srednicy 0.070 m), wiec
    # DATCOM normalizowal przez 1.285, a model mnozyl przez 0.070 — momenty
    # pochylajacy i odchylajacy wychodzily ~18x za male. Zweryfikowane dwiema
    # niezaleznymi drogami (moment z XCP: sila x ramie, kontra moment z CM*LREF),
    # patrz check_moment_reference.py.
    #
    # UWAGA: XCP jest raportowany W JEDNOSTKACH LREF, wiec czytnik
    # (xcp_m = xcg - xcp_cal*lref) automatycznie pozostaje spojny.
    #
    # FLIGHTSIM_LREF_MODE=length odtwarza STARE (bledne) zachowanie. Istnieje
    # wylacznie po to, zeby dalo sie policzyc walidacje "przed" i porownac ja z
    # "po" (compare_validation.py) bez cofania sie w gicie. Do normalnej pracy
    # NIE uzywac — dlatego jest glosne ostrzezenie i dlatego aero.py i tak
    # zazada FLIGHTSIM_ALLOW_STALE_LREF=1, zeby taki wynik wpuscic do modelu.
    lref = d
    if os.environ.get("FLIGHTSIM_LREF_MODE", "").lower() == "length":
        lref = l
        print(f"[MissileDatcom] UWAGA: FLIGHTSIM_LREF_MODE=length — LREF="
              f"{lref:.4f} m (dlugosc kadluba). To STARA, BLEDNA normalizacja, "
              f"tylko do porownan walidacyjnych.")
    lines.append(f" $REFQ    XCG={xcg:.4f},")
    lines.append(f"          SREF={S:.7f},")
    lines.append(f"          LREF={lref:.4f},")
    lines.append(f"          RHR=0.,$")
    lines.append("")

    # --- $AXIBOD ----------------------------------------------------------
    nose_type = cfg.body.nose.type.upper()
    # Missile DATCOM akceptuje: OGIVE, CONE, POWER, HAACK, KARMAN
    if nose_type not in ("OGIVE", "CONE", "POWER", "HAACK", "KARMAN"):
        nose_type = "OGIVE"

    l_cyl = l - l_N
    # Sekcja ogonowa (stożek): jeśli brak danych, zakładamy cylindryczny
    bt = cfg.body.boattail
    if bt is not None and bt.length > 0:
        l_aft  = bt.length
        d_aft  = bt.diameter
        taft   = bt.type.upper() if hasattr(bt, 'type') else "CONE"
        if taft not in ("CONE", "OGIVE"):
            taft = "CONE"
        l_cyl  = l - l_N - l_aft
        d_exit = 0.0   # coast: den zamkniety (pelny opor denny)
    else:
        l_aft  = 0.0
        d_aft  = d
        taft   = "CONE"
        l_cyl  = l - l_N
        d_exit = 0.0

    # Power-on: srednica wylotu dyszy (DEXIT>0) — DATCOM liczy opor denny
    # tylko na pierscieniu (den minus wylot), bo plomien wypelnia srodek.
    # Wartosc z konfiguracji (propulsion.nozzle_diameter, dotad ignorowana).
    # Coast (power_on=False) zostaje przy DEXIT=0 (pelny opor denny).
    if power_on:
        nozzle_d = getattr(getattr(cfg, "propulsion", None), "nozzle_diameter", 0.0) or 0.0
        d_exit = min(float(nozzle_d), d_aft)

    lines.append(f" $AXIBOD  X0=0.00,")
    lines.append(f"          TNOSE={nose_type},")
    lines.append(f"          LNOSE={l_N:.4f},")
    lines.append(f"          DNOSE={d:.4f},")
    lines.append(f"          BNOSE=0.0,")
    lines.append(f"          TRUNC=.FALSE.,")
    lines.append(f"          LCENTR={l_cyl:.4f},")
    lines.append(f"          DCENTR={d:.4f},")
    lines.append(f"          TAFT={taft},")
    lines.append(f"          LAFT={l_aft:.4f},")
    lines.append(f"          DAFT={d_aft:.4f},")
    lines.append(f"          DEXIT={d_exit:.4f},$")
    lines.append("")

    # --- $FINSETn + $DEFLCT --------------------------------------- #
    if not body_only:
        deflect_lines = []
        eff_sets, ctrl_indices = _effective_fin_sets(cfg, include_controls=include_controls)

        for i, (fin, _is_ctrl) in enumerate(eff_sets, start=1):
            # XLE — pozycja krawędzi natarcia
            # XLE_tip = XLE_root + span * tan(sweep_le)
            xle_root = fin.position
            sweep_deg = getattr(fin, 'sweep_le', 0.0)
            xle_tip  = fin.position + fin.span * math.tan(math.radians(sweep_deg))
    
            # SSPAN — rozpiętości od osi (0 = nasada)
            r = d / 2.0
            sspan_root = r
            sspan_tip  = r + fin.span
    
            # PHIF — kąty obrotu paneli
            npanel = int(fin.count)
            phif   = [360.0 / npanel * j for j in range(npanel)]
            phif_str = ",".join(f"{p:.1f}" for p in phif)
    
            # Grubość profilu
            tc = getattr(fin, 'thickness_ratio', 0.05)
            zupper_root = tc * fin.root_chord / 2.0 / fin.root_chord
            zupper_tip  = tc * fin.tip_chord  / 2.0 / fin.tip_chord if fin.tip_chord > 0 else zupper_root
    
            lines.append(f" $FINSET{i} XLE={xle_root:.4f},{xle_tip:.4f},NPANEL={npanel}.,")
            lines.append(f"          SSPAN={sspan_root:.4f},{sspan_tip:.4f},")
            lines.append(f"          CHORD={fin.root_chord:.4f},{fin.tip_chord:.4f},")
            lines.append(f"          LER=2*0.0,")
            lines.append(f"          PHIF={phif_str},")
            # GAM (diedra) — USUNIETE. Wczesniej kat zaklinowania byl zapisywany
            # DWUKROTNIE: jako GAM= ORAZ jako DELTA= w $DEFLCT, wiec DATCOM
            # widzial ~2x zamierzone zaklinowanie. Wedlug podrecznika $DEFLCT
            # ustala "the incidence angle for each panel in each fin set" — to
            # wlasciwe miejsce na zaklinowanie. Zostawiamy je wylacznie w DELTA.
            # emit_gam=True odtwarza stare (bledne) zachowanie do porownania,
            # patrz tests/test_gam_baseline_compare.py.
            if emit_gam and abs(getattr(fin, "cant_angle", 0.0)) > 0.001:
                gam_str = ",".join([f"{fin.cant_angle:.4f}"] * npanel)
                lines.append(f"          GAM={gam_str},")
            lines.append(f"          SECTYP=HEX,")
            lines.append(f"          ZUPPER={zupper_root:.6f},{zupper_tip:.6f},")
            lines.append(f"          LMAXU=0.4,0.4,")
            lines.append(f"          LFLATU=0.2,0.2,$")
            lines.append("")
    
            # DELTA = zaklinowanie (cant) — wychylenie sterowania dokladane
            # osobno per przypadek (patrz delta_cases nizej).
            cant = float(getattr(fin, "cant_angle", 0.0))
            delta_str = ",".join([f"{cant:.4f}"] * npanel)
            deflect_lines.append(f"          DELTA{i}={delta_str},")

        # --- $DEFLCT ----------------------------------------------------------
        # XHINGE musi miec tyle wpisow ile jest zestawow pletw — liczone z TEJ
        # SAMEJ listy co $FINSETn. Wczesniej brane z osobnego cfg.fins[:2], co
        # przy dolozeniu canardow dawalo niezgodna liczbe wpisow i blad parsowania
        # $DEFLCT po stronie DATCOM.
        if eff_sets:
            xhinge_str = ",".join(
                f"{fin.position + fin.root_chord * 0.75:.4f}"
                for fin, _ in eff_sets
            )
            if deflect_lines:
                lines.append(" $DEFLCT  " + deflect_lines[0].strip())
                for dl in deflect_lines[1:]:
                    lines.append("          " + dl.strip())
                lines.append(f"          XHINGE={xhinge_str},$")
        lines.append("")

    # $RLLO — roll rate derivatives (Clp)
    if roll_only:
        lines.append(" $RLLO   ROLLQ=1.0,$")
        lines.append("")
    lines.append("PART")

    # --- Sweep wychylen: przypadki "stacked" (SAVE / NEXT CASE) ------------ #
    # Podrecznik Missile DATCOM, rozdz. 3.3 + Figure 16: karta SAVE zachowuje
    # namelisty poprzedniego przypadku, wiec kolejne przypadki podaja TYLKO
    # $DEFLCT. Jeden przebieg DATCOM zamiast N — i geometria jest z definicji
    # identyczna we wszystkich przypadkach (nie moga sie "rozjechac").
    if delta_cases and not body_only:
        n_panels_of = {i: int(f.count) for i, (f, _) in enumerate(eff_sets, start=1)}
        cant_of     = {i: float(getattr(f, "cant_angle", 0.0))
                       for i, (f, _) in enumerate(eff_sets, start=1)}
        xhinge_str = ",".join(
            f"{fin.position + fin.root_chord * 0.75:.4f}" for fin, _ in eff_sets
        )
        # SAVE MUSI byc w KAZDYM przypadku, nie tylko w pierwszym.
        # Podrecznik, rozdz. 3.2.2: "The SAVE card saves namelist inputs from one
        # case to the following case BUT NOT FOR THE ENTIRE RUN" oraz
        # "If a SAVE control card is not present in a case, all previous case
        # inputs are deleted."
        # Z jednym SAVE geometria docierala tylko do przypadku 2, a od 3. w gore
        # DATCOM kasowal wejscia — w wyniku plik zawieral 2 przypadki zamiast 34.
        lines.append("SAVE")
        lines.append("NEXT CASE")
        for case in delta_cases:
            label = str(case.get("label", "DEFLECTION CASE"))[:60]
            per_set = case.get("delta", {})     # {idx_zestawu: [delta per panel]}
            lines.append(f"CASEID {label}")
            # $DEFLCT zostal zachowany przez SAVE z poprzedniego przypadku, a
            # podrecznik (3.2.2) wymaga: "When changing a namelist that has been
            # saved, the namelist must first be deleted using the delete control
            # card." Bez tego DATCOM wywala sie w trakcie przebiegu (obserwowany
            # kod powrotu 0xC00000A1) — deck konczyl sie na 2 przypadkach.
            lines.append("DELETE DEFLCT")
            dl_lines = []
            for idx in sorted(n_panels_of):
                npan = n_panels_of[idx]
                ctrl = per_set.get(idx, [0.0] * npan)
                if len(ctrl) != npan:
                    raise ValueError(
                        f"delta_cases: zestaw {idx} ma {npan} paneli, podano {len(ctrl)}")
                # Calkowita incydencja panelu = zaklinowanie + wychylenie sterowania
                tot = [cant_of[idx] + float(c) for c in ctrl]
                dl_lines.append(f"DELTA{idx}=" + ",".join(f"{v:.4f}" for v in tot) + ",")
            lines.append(" $DEFLCT  " + dl_lines[0])
            for dl in dl_lines[1:]:
                lines.append("          " + dl)
            lines.append(f"          XHINGE={xhinge_str},$")
            lines.append("PART")
            lines.append("SAVE")      # patrz komentarz wyzej — konieczne w kazdym przypadku
            lines.append("NEXT CASE")
    else:
        lines.append("NEXT CASE")

    output_path.write_text("\n".join(lines), encoding="ascii")
    print(f"[MissileDatcom] Wygenerowano: {output_path}")
    return output_path



