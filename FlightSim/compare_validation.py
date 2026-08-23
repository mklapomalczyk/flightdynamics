"""
compare_validation.py
=====================
Porownanie wynikow walidacji PRZED i PO zmianie w modelu.

Walidacja per lot zyje w telemetry_analysis/analyze_per_flight_6dof.py i
zapisuje field_test_data/results/per_flight_6dof_per_nose.csv. Ten skrypt
pozwala zrobic z tego pliku migawke, a potem porownac z nia nowy wynik —
zeby bylo widac, co dokladnie zmiana w modelu poprawila, a co pogorszyla.

Dla konkretnie poprawki LREF nie rob tego recznie — jest gotowy skrypt, ktory
robi oba przebiegi i porownanie jednym poleceniem (i sam ustawia zmienne
srodowiskowe, wiec nie zalezy od shella):

    python run_lref_validation_compare.py

Recznie, dla dowolnej innej zmiany w modelu:
    # 1. przed zmiana (albo tuz po pobraniu starych wynikow)
    python telemetry_analysis/analyze_per_flight_6dof.py
    python compare_validation.py --save-baseline przed_LREF

    # 2. po zmianie w modelu i przeliczeniu DATCOM
    python telemetry_analysis/analyze_per_flight_6dof.py
    python compare_validation.py --vs przed_LREF

Migawki leza w field_test_data/results/validation_snapshots/.

Uzycie:
    python compare_validation.py --list
    python compare_validation.py --save-baseline <nazwa>
    python compare_validation.py --vs <nazwa>
    python compare_validation.py --vs <nazwa> --plot
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "field_test_data" / "results"
CSV_NAME = "per_flight_6dof_per_nose.csv"
SNAP_DIR = RESULTS / "validation_snapshots"

# Analiza per lot zapisuje DWA warianty predykcji:
#   'adjusted'       — bez wiatru,
#   'adjusted_wind'  — z profilem wiatru.
# Do crossrange liczy sie WYLACZNIE wariant z wiatrem: bez wiatru model nie ma
# czym znosic rakiety w bok, wiec crossrange wychodzi ~0 niezaleznie od tego,
# co zmienimy w aerodynamice. Porownywanie kolumny bez wiatru pokazywalo wiec
# "brak zmian" tam, gdzie zmiana jest najwieksza.
VARIANTS = {
    "adjusted":      "bez wiatru",
    "adjusted_wind": "z wiatrem",
}


def metrics_for(variant: str):
    """(kolumna_modelu, kolumna_pomiaru, etykieta, jednostka)"""
    v = variant
    return [
        (f"h_apo_pred_{v}",             "h_apo_actual",            "apogeum",    "m"),
        (f"v_max_pred_{v}",             "v_max_actual",            "Vmax",       "m/s"),
        (f"downrange_apo_pred_{v}_m",   "downrange_apo_actual_m",  "downrange",  "m"),
        (f"crossrange_apo_pred_{v}_m",  "crossrange_apo_actual_m", "crossrange", "m"),
    ]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "?"
    except Exception:
        return "?"


def meta_path(name: str) -> Path:
    return SNAP_DIR / f"{name}.meta.json"


def read_meta(name: str) -> dict:
    p = meta_path(name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return {int(r["fno"]): r for r in csv.DictReader(f)}


def fnum(row, key):
    try:
        v = float(row.get(key, ""))
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def err_stats(data, pred_key, act_key):
    """Bledy bezwzgledne model-pomiar dla wszystkich lotow z danymi."""
    out = {}
    for fno, row in data.items():
        p, a = fnum(row, pred_key), fnum(row, act_key)
        if np.isfinite(p) and np.isfinite(a):
            out[fno] = (p, a, p - a)
    return out


def cmd_list():
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    snaps = sorted(SNAP_DIR.glob("*.csv"))
    if not snaps:
        print(f"Brak migawek w {SNAP_DIR}")
        return 0
    print(f"Migawki w {SNAP_DIR}:")
    for s in snaps:
        d = read_csv(s)
        m = read_meta(s.stem)
        prov = (f"  {m.get('saved_at','?')}  commit {m.get('git_commit','?')}"
                if m else "  (bez metadanych — zapisana starsza wersja skryptu)")
        print(f"  {s.stem:<28} {len(d) if d else 0} lotow{prov}")
    return 0


def cmd_save(name: str):
    src = RESULTS / CSV_NAME
    if not src.exists():
        print(f"Brak {src}\nUruchom najpierw: "
              f"python telemetry_analysis/analyze_per_flight_6dof.py")
        return 1
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    dst = SNAP_DIR / f"{name}.csv"
    shutil.copy2(src, dst)
    # Metadane sa tu po to, zeby dalo sie ODPOWIEDZIEC "jaki model wyprodukowal
    # ta migawke". Bez nich porownanie dwoch identycznych plikow wyglada jak
    # "zmiana nic nie dala", a naprawde znaczy "migawka zostala zrobiona juz po
    # zmianie modelu" — czego z samego CSV nie da sie odroznic.
    meta = {"saved_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_head(),
            "source_sha": file_sha(src),
            "source_mtime": src.stat().st_mtime,
            "n_flights": len(read_csv(dst))}
    meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Zapisano migawke: {dst}  ({meta['n_flights']} lotow)")
    print(f"  commit {meta['git_commit']}   sha CSV {meta['source_sha']}")
    print(f"\nTeraz: zmien model, przelicz walidacje ponownie")
    print(f"       (python telemetry_analysis/analyze_per_flight_6dof.py),")
    print(f"       dopiero potem: python compare_validation.py --vs {name}")
    return 0


def cmd_compare(name: str, do_plot: bool, exclude: set):
    snap = SNAP_DIR / f"{name}.csv"
    src = RESULTS / CSV_NAME
    base = read_csv(snap)
    curr = read_csv(src)
    if base is None:
        print(f"Brak migawki '{name}'. Dostepne:"); cmd_list(); return 1
    if curr is None:
        print(f"Brak {src}\nUruchom najpierw: "
              f"python telemetry_analysis/analyze_per_flight_6dof.py")
        return 1

    # --- czy w ogole jest co porownywac? --------------------------------- #
    # Najczestszy blad uzycia: migawka zapisana JUZ PO zmianie modelu, albo
    # walidacja nieprzeliczona po zmianie. W obu wypadkach oba pliki sa te same
    # i wykres pokazuje pary identycznych slupkow, co czytа sie jak "zmiana nic
    # nie zmienila". To NIE jest wynik — to brak wyniku, wiec przerywamy.
    meta = read_meta(name)
    if file_sha(snap) == file_sha(src):
        print("=" * 84)
        print("PRZERWANO: migawka i aktualny wynik to DOKLADNIE TEN SAM plik.")
        print("=" * 84)
        print(f"  migawka        : {snap}")
        print(f"  aktualny wynik : {src}")
        print(f"  sha (oba)      : {file_sha(snap)}")
        if meta:
            print(f"  migawka zapisana: {meta.get('saved_at','?')}  "
                  f"commit {meta.get('git_commit','?')}")
        print("\nNie ma tu zadnej zmiany do pokazania. Prawdopodobna przyczyna:")
        print("  a) migawka zostala zapisana JUZ PO zmianie modelu — wtedy")
        print("     'przed' i 'po' opisuja ten sam model. Trzeba wrocic do")
        print("     starej wersji modelu, przeliczyc walidacje i zapisac")
        print("     migawke jeszcze raz;")
        print("  b) walidacja nie zostala przeliczona po zmianie modelu —")
        print("     uruchom: python telemetry_analysis/analyze_per_flight_6dof.py")
        print("     i dopiero potem porownanie;")
        print("  c) zmiana byla sterowana zmienna srodowiskowa")
        print("     (FLIGHTSIM_LREF_MODE), ktora nie dotarla do Pythona — w")
        print("     PowerShell `set` to alias Set-Variable i NIE tworzy zmiennej")
        print("     srodowiskowej (`$env:NAZWA=\"...\"` tworzy).")
        print("     Sprawdz:  python check_lref_mode.py")
        print("     Najprosciej pominac shell i uruchomic calosc jednym")
        print("     poleceniem:  python run_lref_validation_compare.py")
        return 2

    if meta.get("source_mtime") and src.stat().st_mtime <= meta["source_mtime"] + 1:
        print("UWAGA: plik z aktualnym wynikiem nie byl modyfikowany od czasu")
        print("       zapisania migawki — walidacja moze byc nieprzeliczona.\n")

    print("=" * 84)
    if meta:
        print(f"migawka '{name}': {meta.get('saved_at','?')}, "
              f"commit {meta.get('git_commit','?')}   |   teraz: commit {git_head()}")
    print(f"WALIDACJA: '{name}' (przed)   vs   aktualny wynik (po)")
    if exclude:
        print(f"WYKLUCZONE LOTY: {sorted(exclude)}")
    print("=" * 84)

    all_summaries = {}
    for variant, vlabel in VARIANTS.items():
        metrics = metrics_for(variant)
        if not any(k in (next(iter(curr.values())) or {}) for k, _, _, _ in metrics):
            print(f"\n[wariant '{variant}' ({vlabel}) — brak kolumn w CSV, pomijam]")
            continue
        print(f"\n{'#' * 84}")
        print(f"# WARIANT: {variant}  ({vlabel})")
        print(f"{'#' * 84}")
        all_summaries[variant] = _compare_one(base, curr, metrics, exclude)

    print("\n" + "=" * 84)
    print("PODSUMOWANIE ZBIORCZE (mniej = lepiej)")
    print("=" * 84)
    # RMSE sam w sobie nie wystarcza: potrafi urosnac przez JEDEN lot z bledem
    # znaku, jednoczesnie ukrywajac to, ze systematyczne przesuniecie zmalalo.
    # Dlatego rozbijamy blad na BIAS (przesuniecie, jednakowe dla wszystkich
    # lotow) i ROZRZUT (to, czego bias nie tlumaczy) — RMSE^2 = bias^2 + rozrzut^2.
    print("  BIAS    = sredni blad (znak mowi, w ktora strone model sie myli)")
    print("  ROZRZUT = odchylenie std bledu (blad NIE-systematyczny)")
    for variant, summary in all_summaries.items():
        print(f"\n  wariant '{variant}' ({VARIANTS[variant]}):")
        print(f"  {'metryka':<12} {'n':>3} {'RMSE':>17} {'BIAS':>19} "
              f"{'ROZRZUT':>17}   ocena")
        for label, unit, r0, r1, b0, b1, s0, s1, n in summary:
            chg = (r1 - r0) / r0 * 100.0 if r0 > 1e-12 else np.nan
            verdict = ("IDENTYCZNE" if abs(r1 - r0) < 1e-9 else
                       "POPRAWA" if chg < -1 else
                       "POGORSZENIE" if chg > 1 else "bez zmian")
            print(f"  {label:<12} {n:>3} {r0:>7.1f}->{r1:<8.1f} "
                  f"{b0:>+8.1f}->{b1:<+9.1f} {s0:>7.1f}->{s1:<8.1f}   {verdict}")
    print("=" * 84)

    if do_plot:
        for variant in all_summaries:
            _plot(base, curr, metrics_for(variant), name, variant, exclude)
    return 0


def _compare_one(base, curr, metrics, exclude=frozenset()):
    """Tabela per lot + statystyki dla jednego wariantu."""
    summary = []
    for pred_key, act_key, label, unit in metrics:
        eb = err_stats(base, pred_key, act_key)
        ec = err_stats(curr, pred_key, act_key)
        common = [f for f in sorted(set(eb) & set(ec)) if f not in exclude]
        if not common:
            print(f"\n{label}: brak wspolnych lotow z danymi — pomijam")
            continue

        print(f"\n{label} [{unit}]")
        print(f"  {'lot':>4} {'pomiar':>10} {'przed':>10} {'po':>10} "
              f"{'blad przed':>11} {'blad po':>10} {'zmiana':>9}")
        for fno in common:
            p0, a, e0 = eb[fno]
            p1, _, e1 = ec[fno]
            better = abs(e1) < abs(e0)
            mark = "lepiej" if better else ("gorzej" if abs(e1) > abs(e0) else "=")
            print(f"  {fno:>4} {a:>10.1f} {p0:>10.1f} {p1:>10.1f} "
                  f"{e0:>+11.1f} {e1:>+10.1f} {mark:>9}")

        e0 = np.array([eb[f][2] for f in common])
        e1 = np.array([ec[f][2] for f in common])
        rmse0, rmse1 = float(np.sqrt((e0 ** 2).mean())), float(np.sqrt((e1 ** 2).mean()))
        bias0, bias1 = float(e0.mean()), float(e1.mean())
        sd0 = float(e0.std(ddof=1)) if len(e0) > 1 else 0.0
        sd1 = float(e1.std(ddof=1)) if len(e1) > 1 else 0.0
        chg = (rmse1 - rmse0) / rmse0 * 100.0 if rmse0 > 1e-12 else np.nan
        print(f"  {'RAZEM':>4} {'':>10} {'':>10} {'':>10} "
              f"RMSE {rmse0:>7.1f} -> {rmse1:<7.1f} ({chg:+.1f}%)")
        print(f"  {'':>4} {'':>10} {'':>10} {'':>10} "
              f"BIAS {bias0:>+7.1f} -> {bias1:<+7.1f}   "
              f"ROZRZUT {sd0:.1f} -> {sd1:.1f}")
        # Systematyczne niedoszacowanie/przeszacowanie widac dopiero po znakach:
        # same |bledy| tego nie pokazuja, a to najczesciej wlasnie one mowia,
        # czy model reaguje za slabo, czy za mocno.
        s0 = "".join("+" if v > 0 else "-" for v in e0)
        s1 = "".join("+" if v > 0 else "-" for v in e1)
        if s0 != s1:
            print(f"  {'':>4} znaki bledu: przed [{s0}]  ->  po [{s1}]")
        summary.append((label, unit, rmse0, rmse1, bias0, bias1, sd0, sd1, len(common)))
    return summary


def _plot(base, curr, metrics, name, variant, exclude=frozenset()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (pred_key, act_key, label, unit) in zip(axes, metrics):
        eb, ec = err_stats(base, pred_key, act_key), err_stats(curr, pred_key, act_key)
        common = [f for f in sorted(set(eb) & set(ec)) if f not in exclude]
        if not common:
            ax.set_visible(False); continue
        x = np.arange(len(common)); w = 0.38
        ax.bar(x - w/2, [eb[f][2] for f in common], w, label=f"przed ({name})",
               color="tab:gray")
        ax.bar(x + w/2, [ec[f][2] for f in common], w, label="po",
               color="tab:blue")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([str(f) for f in common], fontsize=8)
        ax.set_xlabel("lot"); ax.set_ylabel(f"blad model-pomiar [{unit}]")
        ax.set_title(label); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.suptitle(f"Wplyw zmiany modelu na walidacje  ('{name}' -> aktualny) "
                 f"— wariant {variant} ({VARIANTS.get(variant, '')})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = RESULTS / f"validation_compare_{name}_{variant}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano wykres: {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-baseline", metavar="NAZWA")
    ap.add_argument("--vs", metavar="NAZWA")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--plot", action="store_true")
    # Lot 14 domyslnie poza statystyka: jego symulacja NIE DZIALA (apogeum
    # ~190 m przy pomiarze 2142 m, downrange ~100 m przy 3024 m). Blad rzedu
    # -2000 m dominuje RMSE apogeum i downrange w OBU przebiegach, wiec
    # wliczanie go opisuje zepsuty przebieg, a nie skutek zmiany w modelu.
    # --exclude "" wlacza go z powrotem.
    ap.add_argument("--exclude", default="14", metavar="LOTY",
                    help="numery lotow pominietych w statystyce, po przecinku "
                         "(domyslnie 14 — symulacja tego lotu nie dziala; "
                         "--exclude \"\" nie pomija nic)")
    a = ap.parse_args()

    excl = {int(x) for x in a.exclude.replace(";", ",").split(",") if x.strip()}

    if a.list:
        return cmd_list()
    if a.save_baseline:
        return cmd_save(a.save_baseline)
    if a.vs:
        return cmd_compare(a.vs, a.plot, excl)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
