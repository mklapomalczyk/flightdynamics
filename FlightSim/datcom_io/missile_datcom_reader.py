"""
datcom_io/missile_datcom_reader.py
====================================
Parser pliku wyjściowego Missile DATCOM (Rev 3/99).
Parsuje sekcję "STATIC AERODYNAMICS FOR BODY-FIN SET 1 AND 2".

XCP: X-C.P. w kalibrach od XCG (ujemne = za XCG)
     xcp_m = XCG + X-C.P. * LREF
"""

from __future__ import annotations
import re
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MissileDatcomAero:
    mach:   float
    alpha:  np.ndarray
    CN:     np.ndarray
    CM:     np.ndarray
    CA:     np.ndarray
    XCP:    np.ndarray   # [m od nosa]
    CNA:    np.ndarray   # CNα [1/deg] z DERIVATIVES
    CYB:    np.ndarray   # CYβ [1/deg] z DERIVATIVES
    CLLP:   np.ndarray   # Clp [1/deg] roll damping z $RLLO
    CLL:    np.ndarray   # rolling moment coefficient [-] z cant_angle
    xcg:    float
    lref:   float
    sref:   float
    CA_base: np.ndarray = field(default_factory=lambda: np.array([]))  # opor denny [-]


@dataclass
class MissileDatcomResult:
    cases: list = field(default_factory=list)

    @property
    def mach_list(self):
        return [c.mach for c in self.cases]

    def get_case(self, mach):
        for c in self.cases:
            if abs(c.mach - mach) < 0.01:
                return c
        return None


def parse_missile_datcom_output(output_path) -> MissileDatcomResult:
    lines = Path(output_path).read_text(encoding="ascii", errors="replace").splitlines()
    result = MissileDatcomResult()

    i = 0
    while i < len(lines):
        if "STATIC AERODYNAMICS FOR BODY" not in lines[i]:
            i += 1
            continue

        # Czytaj warunki lotu
        mach = xcg = sref = lref = None
        for j in range(i+1, min(i+25, len(lines))):
            l = lines[j]
            m = re.search(r"MACH NO\s*=\s*([\d.]+)", l)
            if m: mach = float(m.group(1))
            m = re.search(r"MOMENT CENTER\s*=\s*([\d.]+)", l)
            if m: xcg = float(m.group(1))
            m = re.search(r"REF AREA\s*=\s*([\d.E+\-]+)", l)
            if m: sref = float(m.group(1))
            m = re.search(r"REF LENGTH\s*=\s*([\d.]+)", l)
            if m: lref = float(m.group(1))
            if all(v is not None for v in [mach, xcg, sref, lref]):
                break

        if None in [mach, xcg, sref, lref]:
            i += 1; continue

        cn_data, xcp_data, cna_data, cyb_data, cllp_data, cll_data = [], [], [], [], [], []

        # Znajdź koniec tej sekcji (następny nagłówek STATIC AERODYNAMICS)
        next_section = len(lines)
        for ns in range(i+1, len(lines)):
            if "STATIC AERODYNAMICS FOR BODY" in lines[ns]:
                next_section = ns
                break

        k = i + 1
        # Szukaj tabeli longitudinalnej — tylko do końca tej sekcji
        while k < min(next_section, len(lines)):
            if "----- LONGITUDINAL -----" in lines[k]:
                k += 2
                # Pomiń ewentualne puste linie przed danymi
                while k < len(lines) and not lines[k].strip():
                    k += 1
                while k < len(lines):
                    stripped = lines[k].strip()
                    if not stripped: k += 1; break
                    parts = stripped.split()
                    if len(parts) >= 4:
                        try:
                            cll = float(parts[6]) if len(parts) >= 7 else 0.0
                            cn_data.append((float(parts[0]), float(parts[1]),
                                            float(parts[2]), float(parts[3])))
                            cll_data.append((float(parts[0]), cll))
                        except ValueError:
                            pass
                    k += 1
                break
            k += 1

        # Szukaj tabeli XCP — tylko do końca tej sekcji
        while k < min(next_section, len(lines)):
            if "X-C.P." in lines[k] and "ALPHA" in lines[k]:
                k += 1
                # Pomiń ewentualne puste linie przed danymi
                while k < len(lines) and not lines[k].strip():
                    k += 1
                while k < len(lines):
                    stripped = lines[k].strip()
                    if not stripped or "X-C.P. MEAS" in stripped: break
                    parts = stripped.split()
                    if len(parts) >= 5:
                        try:
                            # X-C.P. w REF. LENGTHS od XCG
                            # NEG. AFT = ujemne oznacza za XCG (ku ogonowi)
                            # xcp_m = xcg - xcp_cal * lref
                            xcp_cal = float(parts[4])
                            xcp_m   = xcg - xcp_cal * lref
                            xcp_data.append((float(parts[0]), xcp_m))
                        except ValueError:
                            pass
                    k += 1
                break
            k += 1

        # Szukaj sekcji DERIVATIVES (CNA per degree)
        # Może być w tej samej sekcji lub w kolejnej sekcji z tym samym Mach
        kd = i + 1
        while kd < len(lines):
            l_kd = lines[kd]
            # Zatrzymaj się gdy napotkamy nowy Mach (inna liczba niż bieżąca)
            m_new = re.search(r"MACH NO\s*=\s*([\d.]+)", l_kd)
            if m_new and abs(float(m_new.group(1)) - mach) > 0.01:
                break
            if "DERIVATIVES (PER DEGREE)" in l_kd:
                kd += 2  # skip nagłówek ALPHA CNA CMA ...
                while kd < len(lines) and not lines[kd].strip():
                    kd += 1
                while kd < len(lines):
                    stripped = lines[kd].strip()
                    if not stripped or "PANEL" in stripped: break
                    parts = stripped.split()
                    if len(parts) >= 3:
                        try:
                            cna_data.append((float(parts[0]), float(parts[1])))
                            if len(parts) >= 4:
                                cyb_data.append((float(parts[0]), float(parts[3])))
                        except ValueError:
                            pass
                    kd += 1
                break
            kd += 1

        # Szukaj sekcji ROLL RATE DERIVATIVES (Clp) — z $RLLO
        kr = i + 1
        while kr < next_section:
            if "ROLL RATE DERIVATIVES" in lines[kr]:
                kr += 2
                while kr < len(lines) and not lines[kr].strip():
                    kr += 1
                while kr < len(lines):
                    stripped = lines[kr].strip()
                    if not stripped: break
                    parts = stripped.split()
                    if len(parts) >= 2:
                        try:
                            cllp_data.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass
                    kr += 1
                break
            kr += 1

        if cn_data and len(cn_data) == len(xcp_data):
            result.cases.append(MissileDatcomAero(
                mach  = mach,
                alpha = np.array([r[0] for r in cn_data]),
                CN    = np.array([r[1] for r in cn_data]),
                CM    = np.array([r[2] for r in cn_data]),
                CA    = np.array([r[3] for r in cn_data]),
                XCP   = np.array([r[1] for r in xcp_data]),
                CNA   = np.array([r[1] for r in cna_data]) if len(cna_data) == len(cn_data) else np.zeros(len(cn_data)),
                CLLP  = np.array([r[1] for r in cllp_data]) if len(cllp_data) == len(cn_data) else np.zeros(len(cn_data)),
                CYB   = np.array([r[1] for r in cyb_data])  if len(cyb_data)  == len(cn_data) else np.zeros(len(cn_data)),
                CLL   = np.array([r[1] for r in cll_data])  if len(cll_data)  == len(cn_data) else np.zeros(len(cn_data)),
                xcg=xcg, lref=lref, sref=sref,
            ))
        elif not cn_data and (cna_data or cllp_data) and result.cases:
            # Sekcja zawiera tylko DERIVATIVES bez CN/CM/CA
            # Szukaj przypadku z tym samym Mach i dołącz CNA/CLLP
            for case in result.cases:
                if abs(case.mach - mach) < 0.01 and len(case.alpha) > 0:
                    if cna_data and len(cna_data) == len(case.alpha):
                        case.CNA  = np.array([r[1] for r in cna_data])
                    if cllp_data and len(cllp_data) == len(case.alpha):
                        case.CLLP = np.array([r[1] for r in cllp_data])
                    break
        i = k

    # --- Drugi przebieg: opor denny (CA-BASE) z sekcji rozbicia CA -------
    # DATCOM drukuje rozbicie oporu osiowego w tabeli:
    #   ALPHA  CA-FRIC  CA-PRES/WAVE  CA-BASE  CA-PROT  CA-SEP  CA-ALP
    # (sekcja "BODY ALONE PARTIAL OUTPUT", jedna na Mach). CA-BASE to
    # skladowa denna calkowitego CA — potrzebna do modelu z napędem
    # (plomien silnika wypelnia den, opor denny ~0 podczas spalania) vs
    # bez napedu (pelny opor denny podczas lotu balistycznego). CA-PROT i
    # CA-SEP bywaja puste, wiec CA-BASE to zawsze 4. kolumna liczbowa
    # (parts[3]).
    base_by_mach = {}   # mach -> list[(alpha, CA_base)]
    cur_mach = None
    bi = 0
    while bi < len(lines):
        m = re.search(r"MACH NO\s*=\s*([\d.]+)", lines[bi])
        if m:
            cur_mach = float(m.group(1))
        if "CA-FRIC" in lines[bi] and "CA-BASE" in lines[bi] and cur_mach is not None:
            rows = []
            bk = bi + 1
            while bk < len(lines):
                s = lines[bk].strip()
                if not s:
                    if rows:
                        break
                    bk += 1
                    continue
                parts = s.split()
                if len(parts) >= 4:
                    try:
                        rows.append((float(parts[0]), float(parts[3])))
                    except ValueError:
                        break
                else:
                    break
                bk += 1
            if rows:
                base_by_mach.setdefault(round(cur_mach, 4), rows)
            bi = bk
            continue
        bi += 1

    for c in result.cases:
        rows = base_by_mach.get(round(c.mach, 4))
        if rows:
            a_b = np.array([r[0] for r in rows])
            v_b = np.array([r[1] for r in rows])
            # dopasuj do siatki alpha przypadku (zwykle identyczna)
            if len(a_b) == len(c.alpha) and np.allclose(a_b, c.alpha, atol=0.1):
                c.CA_base = v_b
            else:
                c.CA_base = np.interp(c.alpha, a_b, v_b)
        else:
            c.CA_base = np.zeros(len(c.alpha))

    # Usuń duplikaty, posortuj
    seen, unique = set(), []
    for c in sorted(result.cases, key=lambda x: x.mach):
        if c.mach not in seen:
            seen.add(c.mach); unique.append(c)
    result.cases = unique

    if not result.cases:
        print(f"[MissileDatcom] Ostrzezenie: brak danych w {output_path}")
    else:
        print(f"[MissileDatcom] Wczytano {len(result.cases)} przypadkow Mach: {result.mach_list}")
    return result


def missile_datcom_to_table_aero(result: MissileDatcomResult) -> dict:
    if not result.cases:
        raise ValueError("MissileDatcomResult jest pusty")
    machs  = np.array([c.mach for c in result.cases])
    alphas = result.cases[0].alpha
    n_m, n_a = len(machs), len(alphas)
    CN_t = np.zeros((n_a, n_m))
    CA_t = np.zeros((n_a, n_m))
    XCP_t= np.zeros((n_a, n_m))
    CM_t = np.zeros((n_a, n_m))
    CNA_t  = np.zeros((n_a, n_m))
    CLLP_t = np.zeros((n_a, n_m))
    CYB_t  = np.zeros((n_a, n_m))
    CLL_t  = np.zeros((n_a, n_m))
    CAB_t  = np.zeros((n_a, n_m))
    for j, c in enumerate(result.cases):
        CN_t[:,j]=c.CN; CA_t[:,j]=c.CA; XCP_t[:,j]=c.XCP; CM_t[:,j]=c.CM
        CNA_t[:,j]=c.CNA; CLLP_t[:,j]=c.CLLP; CYB_t[:,j]=c.CYB; CLL_t[:,j]=c.CLL
        if getattr(c, "CA_base", None) is not None and len(c.CA_base) == n_a:
            CAB_t[:,j] = c.CA_base
    return dict(alpha_deg=alphas, mach=machs, CN=CN_t, CA=CA_t,
                XCP=XCP_t, CM=CM_t, CNA=CNA_t, CLLP=CLLP_t, CYB=CYB_t, CLL=CLL_t,
                CA_base=CAB_t,
                xcg=result.cases[0].xcg,
                lref=result.cases[0].lref, sref=result.cases[0].sref)
