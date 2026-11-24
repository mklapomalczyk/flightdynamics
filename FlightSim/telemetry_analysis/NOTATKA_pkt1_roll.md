# Notatka — Punkt 1: walidacja dynamiki roll (spin) z lotów

Walidacja modelu 6DOF pod kątem AMS/RMS. Model nie służy do przewidywania
apogeum/zasięgu — dostarcza **obwiedni lotu** (Mach, ciśnienie dynamiczne,
orientacja, **prędkość obrotowa**), w której liczone i porównywane są AMS/RMS.
Punkt 1 dotyczy prędkości obrotowej (roll) jako funkcji kąta zaklinowania
stateczników.

## 1. Cel

- Wyznaczyć z lotów prędkość obrotową ustaloną ω_spin i jej zależność od kąta
  zaklinowania (cant).
- Wyznaczyć skuteczność zaklinowania i tłumienie roll (stosunek Clδ/Clp).
- Porównać z modelem (DATCOM) i — jeśli się różni — zaproponować poprawkę.

## 2. Dane i ekstrakcja spinu

- **Źródło spinu: kanał AX / `acc_RXO`** („Prędkość obrotowa AX [st./s]"),
  przeliczony do °/s i **nienasycający się** (żyroskop roll jest przycięty na
  ±2000 °/s i przy cant ≥ 1.2 saturuje 75–89 % lotu). Bierzemy go wprost jako
  °/s, odejmując baseline przedstartowy (spin = 0 przed startem). **Żyroskop NIE
  jest używany do skalowania** (był pierwotnie — błąd; przenosił swoje nasycenie
  na AX). Magnetometr (częstotliwość) — niezależna kontrola.
- Loty analizowane (ostra): cant **0 / 0.6 / 1.2 / 1.6°** (4 poziomy, powtórzenia
  na 0.6 i 1.6). Lot 23 bez MGLO — ograniczony. Loty cant=0 (13/17/18) pokazują
  **niesymetrię wykonania** (spin bez zaklinowania) — osobny człon „a".

## 3. Wynik empiryczny

Bezwymiarowa prędkość obrotowa w równowadze okołowypaleniowej:

> **p·d/2V = 0.0103·cant[deg] − 0.0008,  R² = 0.89**

- zależność od cant czysta i monotoniczna; nachylenie = −Clδ/Clp,
- wyraz wolny „a" ≈ 0 (niesymetria mała dla większości lotów).

Niezmiennik się potwierdza: loty 1.6° o różnych V i spinie bezwzględnym
(15/16: ~10000 °/s @330 m/s; 22/24/25: ~6000 °/s @180 m/s) lądują na tym samym
p·d/2V (~0.016) — bo lecą z różną prędkością, a cant jest ten sam.

## 4. Porównanie z modelem DATCOM

**Pułapka LREF (ważne):** pierwotne tabele DATCOM dla cant miały
`LREF = 1.285 m` (długość kadłuba) zamiast średnicy 0.070 m — przestarzałe
(`aero.py` takie odrzuca). Biegi zregenerowano z `LREF = średnica`
(`datcom_runs/rocket_70mm_cant06/12/16`, cant=0 = bieg bazowy). Stosunek CLL/Clp
jest niezmienniczy względem LREF, więc wynik jest teraz rygorystyczny:

> model p·d/2V = 0.0026/deg  vs  pomiar 0.0103/deg  →
> **stosunek CLL/Clp niedoszacowany ×4.0**

## 5. Podział błędu: napęd (CLL) czy tłumienie (Clp)?

Stosunek jest identyfikowalny, ale sam podział — nie z lotu (spin jest w
quasi-równowadze). Rozstrzygnięcie dwoma niezależnymi drogami:

- **Czas narastania:** przy poprawnym (silnym) Clp z DATCOM model pika w ~2 s
  (przy wypaleniu) — zgodnie z pomiarem. Timing dobry → brakuje napędu, nie
  nadmiaru tłumienia.
- **Strip-theory (geometria):** równowaga p·d/2V wychodzi **czysto
  geometrycznie** (efektywność płetwy clα się skraca):
  > strip-theory 0.00922/deg  ≈  pomiar 0.01035/deg  (zgodność 12 %),
  > model 0.00259/deg — **3.6× poniżej**.
  Geometria tłumienia analitycznego Clp jest dobra (ramię² 0.829 ≈ strip 0.837).

**Wniosek:** pomiar jest fizycznie poprawny (zgodny ze strip-theory z pierwszych
zasad). DATCOM **zaniża moment napędowy roll od zaklinowania (CLL) ~4×**;
tłumienie Clp ma dobre. Najprawdopodobniejsza przyczyna: cant podawany do DATCOM
jako **wychylenie steru (DELTA)** ma zredukowaną skuteczność klapy zamiast
pełnej incydencji zabudowanej płetwy.

> **Poprawka: CLL_scale ≈ 3.6 (per lot 2.3–4.6), Clp_scale ≈ 1.**
> Zakres: ostra, cant 0–1.6°, Mach ~0.4–1.0.

## 6. Weryfikacja czasowa i dokładność

`compare_roll_model.py` całkuje równanie rolki
`Ixx·dp/dt = q·S·d·[CLL·CLL_scale − Clp·Clp_scale·(p·d/2V)]`, napędzane
rzeczywistą obwiednią lotu, i nakłada na zmierzony przebieg. Po korekcie CLL×4
model trzyma się testu na całym przebiegu.

**Prędkość w obwiedni — kluczowa.** Błąd prędkości wchodzi ~1:1 w model
(p ∝ V). Rozwiązanie: fuzja komplementarna `gps_fusion.fused_speed` — całka
osiowego acc_x (dobra na ciągu, gdzie GPS-doppler przycina) korygowana do
**BEZPOŚREDNIEJ** prędkości GPS (vel_north/east/down, lepszej niż gradient
pozycji) w fazie balistycznej, z bramkami (przyspieszenie + opóźnienie po
wypaleniu + odrzucenie absurdalnie niskich wskazań GPS). Zastąpiła protezę
`max(acc_x, GPS)` i dryfującą rekonstrukcję 3D.

**Dokładność modelu roll (w stabilnych punktach, po korekcie i fuzji):**

| punkt | |model/test − 1| |
|---|---|
| apogeum | **~8 %** |
| maks roll | **~11 %** |
| mediana (wznoszenie, cant>0) | ~12 % |

## 7. Znane ograniczenia

- **Transient narastania** (punkt „1/2 rampy"): model narasta ~1.5× szybciej.
  To efekt stałej czasowej (realny spin-up ma ~0.2 s zwłoki, model siada na
  równowadze ∝ V niemal natychmiast). Sprawdzono: opóźnienie zejścia z
  prowadnicy (3 m ≈ 0.2 s) **nie pomaga** (model „zapomina" start); jedyny czysty
  lewar — Ixx — wymaga niefizycznego ×5. **Zostawione jako ograniczenie** — nie
  jest wejściem do AMS/RMS i jest bardzo czułe na timing.
- **Opadanie:** po wejściu spadochronu model płetwowy nie obowiązuje.
- **cant = 0:** spin pochodzi z niesymetrii wykonania („a"), model DATCOM go nie
  odtwarza — osobny człon, do dołożenia niezależnie.
- **Rozrzut CLL_scale (±~0.8):** jest słaby trend z Machem (r≈−0.71), ale z dużym
  szumem (±0.6 przy tym samym Machu). Strip-theory mówi, że p·d/2V jest
  geometryczne → ~niezależne od Macha, więc „trend" to najpewniej artefakt
  (Mach przy wypaleniu koreluje z V). **Nie dopasowujemy CLL_scale(Mach)** —
  lepiej stała ×4 z niepewnością niż fałszywa precyzja krzywej.

## 8. Skrypty

| skrypt | rola |
|---|---|
| `analyze_spin_cant.py` | ekstrakcja spinu (AX/RXO), p·d/2V vs cant, porównanie z DATCOM, propozycja CLL_scale |
| `roll_strip_check.py` | sprawdzian strip-theory (geometria) vs pomiar vs model |
| `compare_roll_model.py` | całkowanie równania rolki, model vs test w czasie |
| `roll_accuracy_table.py` | tabela dokładności (5 punktów/lot), p·d/2V (wyklucza V) |
| `gps_fusion.py` → `fused_speed` | fuzja prędkości acc_x + bezpośredni GPS |

Tabele DATCOM: `datcom_runs/rocket_70mm_cant06/12/16` (LREF = średnica).
Konfiguracja geometrii/mas/Ixx: `configurations/rocket_70mm_baseline.yaml`.

## 9. Do zrobienia

- Pkt 2 — żyroskopy pitch/yaw (z poprawnym spinem jako wejściem).
- Pkt 4 — tabelka zbiorcza: CLL_scale ≈ 4 ± niepewność, ω_spin, kolumny
  Measured/status + absolute/relative.
- Człon niesymetrii „a" dla cant ≈ 0.
- (Opcjonalnie) prawdziwy DATCOM CLLP ($RLLO) zamiast analitycznego Clp — dla
  pełnego domknięcia podziału CLL/Clp.
