# Notatka — Punkt 2: walidacja pitch/yaw (Cm_α, Cmq) z lotów

Kontynuacja walidacji modelu 6DOF pod AMS/RMS (po pkt 1 = roll). Cel: wyznaczyć
z lotów sztywność statyczną pitch (Cm_α) i tłumienie (Cmq), porównać z DATCOM.

## 1. Problem: żyroskopy pitch/yaw nie dają sygnału

- `gyro_y`/`gyro_z` są zdominowane przez **obrót rolla**: ruch poprzeczny stały
  w przestrzeni oscyluje w ramie ciała na częstotliwości spinu f_spin. Do tego
  **offset od niedokładnego montażu** (przechył osi żyroskopu przecieka dużą
  prędkość rolla jako składową ∝ p).
- Sprawdzono **wszystkie loty** (9–25, w tym cant=0 o znikomym rollu — lot 13,
  f_spin ≈ 0.17 Hz, gdzie pitch/yaw NIE jest maskowany). De-spin (rama
  nie-wirująca) i de-leaking (odjęcie składowej ∝ p = korekta montażu) **nie
  odsłaniają czystego piku** — prominencje 2–3, rozrzucone.
- **Wniosek:** to nie montaż/przeciek jest problemem (usuwalne, nic nie zmienia)
  — ruch pitch/yaw jest **realnie mały**, bo rakieta jest bardzo sztywna
  statycznie (Cm_α ≈ −54/rad → znikomy kąt natarcia → słaby coning). Zgodne z
  powrotem rakiety w całości (brak koziołkowania).

## 2. Co zadziałało: magnetometr 3-osiowy (loty 22/24/25)

Wielkość **niezmiennicza względem rolla**: magnituda pola poprzecznego
`|B_⊥| = sqrt(B_t1² + B_t2²)`. Roll tylko obraca wektor poprzeczny — nie zmienia
jego długości — więc |B_⊥| **nie zależy od kąta rolla**, a **oscyluje na
częstotliwości CONINGU** (kąt między osią rakiety a polem B zmienia się przy
conieniu). To izoluje pitch/yaw bez de-spin.

Częstotliwość pitch (nie-wirująca; sprzężenie żyroskopowe słabe, bo Ixx ≪ Iyy):

> f_pitch = (1/2π)·sqrt(|Cm_α|·q·S_ref·d / Iyy)

**Wynik — pik |B_⊥| vs f_pitch z DATCOM:**

| lot | f_obs | f_pitch(DATCOM) | Cm_scale = (f_obs/f_exp)² |
|---|---|---|---|
| 22 | 2.50 Hz | 2.59 | 0.93 |
| 24 | 2.47 Hz | 2.60 | 0.90 |
| 25 | 2.70 Hz | 2.49 | 1.18 |

> **Cm_scale ≈ 1.0 (0.90–1.18), n=3**

Spójny pik ~2.5 Hz na trzech niezależnych lotach, dokładnie na przewidywanej
f_pitch — przypadek jest mało prawdopodobny, więc sygnał realny (choć słaby;
spektrogram pokazuje rozmyte pasmo, nie ostry grzbiet).

## 3. Wynik

- **Cm_α (sztywność statyczna pitch): ZGODNA z DATCOM, Cm_scale ≈ 1.0.** Brak
  korekty. To **walidacja** (brak dowodu na błąd modelu), nie precyzyjne
  skalowanie — sygnał słaby, tylko 3 loty z mag 3-os.
- **Cmq (tłumienie pitch): NIEWYZNACZALNE z lotu** (zanik coningu za słaby/
  zaszumiony na obwiednię). W modelu jest **ANALITYCZNE**:
  `Cmq = −2·CNA_fins·((x_fins − xcg)/d)²` (z DATCOM tylko CNA_fins = przebieg
  pełny − kadłub; struktura tłumienia analityczna). Ta sama konstrukcja co Clp
  rolla — „podejrzana", ale bez walidacji lotowej. Status: niewalidowane.

## 4. Porównanie z roll (pkt 1)

| | napęd/sztywność | tłumienie |
|---|---|---|
| **roll** | CLL ~**4× za niski** (strip-theory + lot) | Clp ~OK |
| **pitch/yaw** | **Cm_α ≈ OK (×1.0)** (mag coning) | Cmq niewyznaczalne (analityczny) |

Istotny wniosek: **roll wymaga dużej korekty napędu, pitch/yaw nie** — sztywność
statyczna pitch jest dobrze zamodelowana.

## 5. Skrypty

| skrypt | rola |
|---|---|
| `diag_pitch_yaw.py` | f_pitch z DATCOM Cm_α, coning z magnetometru |B_⊥|, Cm_scale, werdykt |

(Metody poboczne sprawdzone i odrzucone: de-spin gyro, de-leaking montażu,
tablica 4 akcelerometrów MPOM — na tej samej stacji osiowej, daje roll/
odśrodkowe, nie pitch/yaw.)

## 6. Do zrobienia

- Pkt 4 — tabelka: roll CLL×4 / pitch Cm_α×1, statusy i niepewności.
- (Opcjonalnie) prawdziwe DATCOM CMQ ($DAMP) i CLLP ($RLLO) — zastąpić wzory
  analityczne tłumienia, jeśli potrzebna wyższa pewność Cmq/Clp.
