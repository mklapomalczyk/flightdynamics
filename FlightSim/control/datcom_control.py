"""
control/datcom_control.py
=========================
Budowa tablic pochodnych sterowania z wyjscia DATCOM (deck skladany
SAVE / NEXT CASE, sweep wychylen — patrz run_control_datcom.py).

Sweep, a nie pojedyncza roznica skonczona, pozwala ZWERYFIKOWAC liniowosc
zamiast ja zalozyc: dopasowujemy prosta w zakresie liniowym i raportujemy
R^2 oraz maksymalne odchylenie na calym sweepie.

Przypisanie przypadkow DATCOM do kanalow robimy po etykietach CASEID
(generator zapisuje np. "D_PITCH DELTA=-4.0"), a nie po kolejnosci — kolejnosc
byloby cichym zalozeniem, ktore rozjedzie sie przy pierwszej zmianie sweepa.
Fallback pozycyjny istnieje, ale glosno ostrzega.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .derivatives import PANEL_PATTERNS, ControlDerivTable, fit_derivative_from_sweep

# Wspolczynnik DATCOM -> pochodna, ktora z niego liczymy.
# Kanal -> (nazwa pola w ControlDerivTable, klucz wspolczynnika DATCOM)
CHANNEL_COEFFS: Dict[str, List[Tuple[str, str]]] = {
    "d_pitch": [("Cm_delta", "CM"), ("CN_delta", "CN")],
    "d_yaw":   [("Cn_delta", "CLN"), ("CY_delta", "CY")],
    "d_roll":  [("Cl_delta", "CLL")],
}

_CASEID_RE = re.compile(r"CASEID\s+(D_[A-Z]+)\s+DELTA\s*=\s*([-+]?\d+(?:\.\d+)?)",
                        re.IGNORECASE)
# Ta sama etykieta, ale juz bez slowa CASEID — tak przepisuje ja DATCOM
# w naglowku kazdej strony wynikow.
_LABEL_RE = re.compile(r"(D_[A-Z]+)\s+DELTA\s*=\s*([-+]?\d+(?:\.\d+)?)",
                       re.IGNORECASE)


def _coeff_grid(group, key: str, alpha: np.ndarray, mach: np.ndarray) -> np.ndarray:
    """Wyciaga (n_alpha, n_mach) dla danego wspolczynnika z grupy przypadkow."""
    out = np.zeros((len(alpha), len(mach)))
    for j, M in enumerate(mach):
        case = group.get_case(M)
        if case is None:
            continue
        vals = np.asarray(getattr(case, key, np.zeros(0)), dtype=float)
        a_c = np.asarray(case.alpha, dtype=float)
        if vals.size == 0 or a_c.size == 0:
            continue
        if vals.shape == alpha.shape and np.allclose(a_c, alpha, atol=1e-6):
            out[:, j] = vals
        else:
            out[:, j] = np.interp(alpha, a_c, vals)
    return out


def parse_sweep_labels(out_path) -> Optional[List[Tuple[str, float]]]:
    """
    Etykiety (kanal, wychylenie) z kart CASEID w wyjsciu DATCOM.
    Zwraca None gdy DATCOM ich nie przepisal — wtedy trzeba fallbacku.
    """
    txt = Path(out_path).read_text(encoding="ascii", errors="replace")
    seen, labels = set(), []
    for m in _CASEID_RE.finditer(txt):
        chan = m.group(1).lower()
        delta = float(m.group(2))
        key = (chan, delta)
        if key in seen:          # CASEID pojawia sie w naglowku kazdej strony
            continue
        seen.add(key)
        labels.append(key)
    return labels or None


def build_control_derivatives(out_path,
                              sweep_deg: Optional[List[float]] = None,
                              linear_range_deg: float = 6.0,
                              S_ref: float = 0.0, d_ref: float = 0.0,
                              verbose: bool = True) -> ControlDerivTable:
    """
    Buduje ControlDerivTable z pliku wyjsciowego sweepa wychylen.

    Uklad pliku (patrz run_control_datcom.py):
      przypadek 1        — geometria, wychylenie 0 (baza)
      przypadki 2..N     — wzorzec x wychylenie, wg PANEL_PATTERNS
    """
    from datcom_io.missile_datcom_reader import parse_missile_datcom_cases

    paths = [out_path] if isinstance(out_path, (str, Path)) else list(out_path)

    groups, base = [], None
    for p in paths:
        gs = parse_missile_datcom_cases(p)
        if not gs:
            continue
        # Pierwszy przypadek kazdego pliku to geometria przy wychyleniu 0.
        # Przy podziale sweepa na kilka przebiegow baza powtarza sie —
        # bierzemy pierwsza, reszta sluzy jako kontrola spojnosci.
        if base is None:
            base = gs[0]
        groups.extend(gs[1:])

    if base is None:
        raise ValueError(f"{paths}: nie udalo sie odczytac zadnego przypadku DATCOM")

    # Etykieta CASEID czytana Z KAZDEGO PRZYPADKU (naglowek strony), a nie z
    # echa wejscia na poczatku pliku. Przy urwanym przebiegu echo zawiera
    # wszystkie 33 karty, a wynikow jest mniej — wiazanie po echu dawaloby
    # ciche przesuniecie przypisania.
    # Odrzucamy przypadki NIEKOMPLETNE (mniej Machow niz baza). Przy crashu
    # DATCOM ostatni przypadek bywa urwany w polowie — jego brakujace Machy
    # weszlyby do dopasowania jako zera i po cichu zepsuly nachylenie.
    n_mach_base = len(base.cases)
    labels, kept, dropped = [], [], []
    for g in groups:
        lbl = g.cases[0].case_label if g.cases else ""
        m = _CASEID_RE.match("CASEID " + lbl) or _LABEL_RE.match(lbl)
        if not m:
            continue
        if len(g.cases) < n_mach_base:
            dropped.append((lbl, len(g.cases)))
            continue
        labels.append((m.group(1).lower(), float(m.group(2))))
        kept.append(g)
    if dropped and verbose:
        for lbl, n in dropped:
            print(f"[ctrl] POMIJAM niekompletny przypadek '{lbl}' "
                  f"({n}/{n_mach_base} Machow) — prawdopodobnie urwany przebieg")
    if kept:
        groups = kept
    n_lab = len(labels)

    def _diag() -> str:
        """Diagnostyka — najczestsza przyczyna to urwany deck skladany."""
        exp = (len(PANEL_PATTERNS) * len(sweep_deg) + 1) if sweep_deg else None
        s = (f"\n  przypadkow DATCOM w pliku : {len(groups)}"
             f"\n  etykiet CASEID odczytanych: {n_lab}")
        if exp:
            s += f"\n  oczekiwano                : {exp} (1 bazowy + wzorce x sweep)"
        if len(groups) <= 2:
            s += ("\n  PRAWDOPODOBNA PRZYCZYNA: karta SAVE musi wystapic w KAZDYM"
                  "\n  przypadku, nie tylko w pierwszym (podrecznik 3.2.2: 'saves"
                  "\n  namelist inputs from one case to the following case but not"
                  "\n  for the entire run'). Bez tego DATCOM kasuje geometrie po"
                  "\n  drugim przypadku. Przegeneruj deck aktualnym generatorem:"
                  "\n      python run_control_datcom.py ctrl --run")
        return s

    if len(groups) < 2:
        raise ValueError(f"{out_path}: sweep wychylen wymaga wielu przypadkow." + _diag())

    sweeps = groups

    # --- przypisanie przypadkow do (kanal, wychylenie) --------------------- #
    if labels is not None and len(labels) == len(sweeps):
        if verbose:
            print(f"[ctrl] przypisanie po CASEID ({len(labels)} przypadkow)")
    else:
        if labels and len(labels) != len(sweeps):
            raise ValueError(
                f"{out_path}: liczba etykiet CASEID ({len(labels)}) nie zgadza "
                f"sie z liczba przypadkow ({len(sweeps)})." + _diag())
        if sweep_deg is None:
            raise ValueError(
                "Brak czytelnych etykiet CASEID i brak sweep_deg — nie da sie "
                "bezpiecznie przypisac przypadkow do kanalow." + _diag())
        labels = [(ch, d) for ch in PANEL_PATTERNS for d in sweep_deg]
        if len(labels) != len(sweeps):
            raise ValueError(
                f"{out_path}: fallback pozycyjny nie pasuje." + _diag())
        print("[ctrl] OSTRZEZENIE: brak etykiet CASEID — przypisanie POZYCYJNE. "
              "Sprawdz, czy kolejnosc w decku odpowiada PANEL_PATTERNS x sweep.")

    # --- wspolna siatka --------------------------------------------------- #
    alpha_deg = np.asarray(base.cases[0].alpha, dtype=float)
    mach = np.asarray(sorted(c.mach for c in base.cases), dtype=float)
    n_a, n_m = len(alpha_deg), len(mach)

    # --- zbierz sweep per kanal ------------------------------------------- #
    per_channel: Dict[str, Dict[str, List]] = {
        ch: {"delta": [], "coeff": {}} for ch in PANEL_PATTERNS}
    for ch in PANEL_PATTERNS:
        for _field, key in CHANNEL_COEFFS[ch]:
            per_channel[ch]["coeff"][key] = []

    # wartosc bazowa (delta = 0) — wspolna dla wszystkich kanalow
    base_grids = {key: _coeff_grid(base, key, alpha_deg, mach)
                  for key in {k for ch in CHANNEL_COEFFS for _f, k in CHANNEL_COEFFS[ch]}}

    for (ch, delta), grp in zip(labels, sweeps):
        if ch not in per_channel:
            continue
        per_channel[ch]["delta"].append(delta)
        for _field, key in CHANNEL_COEFFS[ch]:
            per_channel[ch]["coeff"][key].append(_coeff_grid(grp, key, alpha_deg, mach))

    # --- dopasowanie pochodnych ------------------------------------------- #
    tables = {name: np.zeros((n_a, n_m)) for name in
              ("Cm_delta", "Cn_delta", "Cl_delta", "CN_delta", "CY_delta")}
    fit_info: Dict[str, dict] = {}

    for ch, data in per_channel.items():
        d = np.asarray(data["delta"], dtype=float)
        if d.size < 2:
            print(f"[ctrl] OSTRZEZENIE: kanal {ch} ma {d.size} punktow — pomijam")
            continue
        order = np.argsort(d)
        d_sorted = d[order]

        for field, key in CHANNEL_COEFFS[ch]:
            stack = np.stack(data["coeff"][key], axis=0)[order]   # (n_delta,n_a,n_m)
            # wstaw punkt bazowy delta=0, jesli sweep go nie zawiera
            if not np.any(np.isclose(d_sorted, 0.0)):
                d_use = np.concatenate([d_sorted, [0.0]])
                stack = np.concatenate([stack, base_grids[key][None, ...]], axis=0)
                o2 = np.argsort(d_use)
                d_use, stack = d_use[o2], stack[o2]
            else:
                d_use = d_sorted

            worst_r2, worst_dev, all_r2 = 1.0, 0.0, []
            worst_cell = (None, None)
            for ia in range(n_a):
                for im in range(n_m):
                    slope, info = fit_derivative_from_sweep(
                        d_use, stack[:, ia, im], linear_range_deg)
                    tables[field][ia, im] = slope
                    all_r2.append(info["r2"])
                    if info["r2"] < worst_r2:
                        worst_r2 = info["r2"]
                        worst_cell = (float(alpha_deg[ia]), float(mach[im]))
                    worst_dev = max(worst_dev, info["max_dev_full_sweep"])
            fit_info[field] = {"worst_r2": worst_r2,
                               "median_r2": float(np.median(all_r2)),
                               "worst_cell_alpha_mach": worst_cell,
                               "max_dev_full_sweep": worst_dev,
                               "n_delta": int(len(d_use)),
                               "delta_range_deg": [float(d_use.min()), float(d_use.max())]}
            if verbose:
                wc = fit_info[field]["worst_cell_alpha_mach"]
                print(f"[ctrl] {field:9s} z {key:4s}: R^2 mediana="
                      f"{fit_info[field]['median_r2']:.4f} min={worst_r2:.4f}"
                      f" @(alpha={wc[0]}, M={wc[1]})  max_dev={worst_dev:.5f}")

    tab = ControlDerivTable(
        alpha_rad=np.deg2rad(alpha_deg), mach=mach,
        xcg_ref=float(base.cases[0].xcg),
        S_ref=S_ref or float(base.cases[0].sref),
        d_ref=d_ref or float(base.cases[0].lref),
        fit_info=fit_info, **tables)
    return tab
