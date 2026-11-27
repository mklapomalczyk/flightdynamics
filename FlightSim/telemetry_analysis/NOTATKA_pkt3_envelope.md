# Notatka — Punkt 3: obwiednia lotu (q, Mach, CD, zasięg, apogeum)

Walidacja modelu 6DOF pod AMS/RMS. Punkt 3 = ogólne oszacowanie **obwiedni
lotu** modelu: ciśnienia dynamicznego q, liczby Macha, współczynnika oporu CD
oraz domknięcia trajektorii (zasięg / maksymalna wysokość). Przypomnienie: model
ma dostarczać obwiedni, w której liczone są AMS/RMS — **nie** przewidywać
dokładnie apogeum/zasięgu. Kluczowa jest więc poprawność V/q/Mach/CD w fazie
**wysokoenergetycznej** (okołowypaleniowej, max q), nie samo apogeum.

## 1. Obwiednia V / q / Mach — DOBRA

Z telemetrii (fuzja prędkości `fused_speed` + baro/ISA), loty „yes":

| lot | nos | V_max [m/s] | Mach_max | q_max [kPa] | apogeum GPS [m] |
|---|---|---|---|---|---|
| 13 | tępa | 286 | 0.84 | 49 | 1161 |
| 14 | ostra | 400 | 1.18 | 95 | 2142 |
| 15 | ostra | 449 | 1.32 | 122 | 1535 |
| 16 | ostra | 423 | 1.25 | 108 | 1473 |
| 17 | tępa | 327 | 0.96 | 65 | 1095 |
| 18 | ostra | 409 | 1.21 | 101 | 1134 |
| 19 | ostra | 420 | 1.24 | 106 | 1537 |
| 20 | ostra | 393 | 1.16 | 93 | 1229 |

Rakieta pracuje w zakresie **Mach 0.8–1.3, q do ~120 kPa** — reżim transoniczny,
dokładnie tam, gdzie AMS/RMS są najistotniejsze (najwyższe ciśnienie dynamiczne).

**V_max modelu vs test: |błąd| med 2.2 %** (closure 2D, niezależny od CD przy
dużym Machu — V_max zdeterminowane ciągiem i masą). To potwierdza, że ciąg, masa
i prędkość w fazie wysokoenergetycznej są dobrze odtworzone → **q = ½ρV² i Mach
są wiarygodne przez cały odcinek wysokiego q.**

## 2. Współczynnik oporu CD — ZGODNY z DATCOM (dwie niezależne metody)

CD wyznaczony z lotu dwiema rozłącznymi metodami, w rozłącznych zakresach Macha:

- **(A) Wznoszenie, coast akcelerometrem** (nowe, pkt 3): `CD = −acc_x·G0·m_coast
  / (q·S)`, okno (t > wypalenie+0.5 s) ∧ (t < apogeum−0.5 s) ∧ (V > 40). acc_x to
  specific force (bez grawitacji), więc `acc_x·G0 = −D/m` wprost. **Jedyne
  źródło CD przy DUŻYM Machu (0.4–1.0)** — rakieta jest w locie swobodnym tylko
  tuż po wypaleniu. Wynik (mediany koszyków): **CD ≈ 0.38 @ M0.35 rosnąc do
  ~0.45–0.55 @ M0.55–0.95** — fizycznie poprawny **wzrost transoniczny**.
- **(B) Zniżanie, GPS** (istniejące `cd_datcom_vs_field_ostra/tepa.csv`): CD przy
  MAŁYM Machu (0.2–0.5), **0.36–0.50**, prawie wszystkie koszyki „inside band"
  względem DATCOM.
- **(C) DATCOM CA(Mach) + CA_base** (opór denny, pełny w coascie): **0.37–0.39**
  przy M≤0.4, rosnąc transonicznie; CA_base ≈ 0.05–0.06.

> **Wszystkie trzy się zgadzają (~10 %). DATCOM CD jest DOBRY.**
> m_coast = m_rocket − (m_full − m_empty) (masa po wypaleniu, ~3.4 kg ostra /
> ~4.8 kg dla 22/24/25).

## 3. Domknięcie apogeum — błąd był w krzywej CD, NIE w aerodynamice

Istniejący `trajectory_closure_summary.csv` pokazywał model **zawyżający apogeum
o 60–170 %** przy V_max zgodnym <5 %. To nie jest błąd aerodynamiki modelu —
to artefakt krzywej CD użytej do całkowania.

**Przyczyna.** `validate_trajectory.simulate_ascent` całkuje 2D od zapłonu do
apogeum, a jedynym wejściem oporu jest `cd_at(Ma, nose, curves)` z
`fit_cd_mach_curves.csv` — dopasowania z **fazy zniżania**, które (1) mają dane
tylko do Mach ~0.5 i (2) na ogonie zapadają się do ~0.15 (ostra), po czym
`cd_at` **trzyma tę płaską wartość dla każdego Mach > 0.5** (`np.interp
right=Cd[-1]`). Ponieważ wznoszenie osiąga Mach ~1.0–1.3, model całkuje niemal
cały ciąg + coast przy **CD ≈ 0.15 zamiast ≈ 0.40** → ~3× za mały opór →
apogeum zawyżone. V_max (przy wypaleniu) prawie nietknięte.

**Dowód.** Re-całkowanie tego samego closure trzema źródłami CD (mediany po lotach):

| źródło CD do closure | błąd apogeum (med) |
|---|---|
| krzywa ZNIŻANIA (fit, obecna) | **+89 %** (zawyża) |
| DATCOM CA + CA_base | **+22 %** |
| stały CD = pomiar wznoszenia (~0.40) | **+34 %** |

Podmiana zapadniętego ogona krzywej zniżania na DATCOM (lub zmierzony opór
wznoszenia) **redukuje błąd apogeum z +89 % do ~+22–34 %**. To domyka pętlę:
**błąd apogeum tkwił w ekstrapolacji krzywej CD poza dane (Mach > 0.5), nie w
modelu aerodynamicznym. DATCOM CD jest potwierdzony pomiarem wznoszenia.**

Resztowe +22 % (DATCOM) jest oczekiwane i akceptowalne dla AMS/RMS:
- closure to całkowanie 2D punktu materialnego (ciąg wzdłuż elewacji startu, bez
  oporu indukowanego od kąta natarcia, bez wiatru, bez zakrętu grawitacyjnego);
- DATCOM CA to opór przy alpha=0; realny lot ma dodatkowy opór od alpha;
- apogeum/zasięg to „sanity check" (ten sam rząd wielkości), nie wejście AMS/RMS.

## 4. Wnioski dla AMS/RMS

- **Obwiednia V/q/Mach: wiarygodna** w fazie wysokoenergetycznej (V_max <5 %, CD
  zgodny z DATCOM ~10 %). To jest to, czego potrzebują AMS/RMS.
- **CD modelu (DATCOM): zwalidowany** — nie wymaga korekty (w przeciwieństwie do
  roll CLL ×4 z pkt 1). Zalecenie: do obwiedni używać **DATCOM CA** (lub scalonej
  krzywej wznoszenie+zniżanie), **nie** krzywej dopasowanej tylko ze zniżania.
- **Apogeum/zasięg: sanity check** — model domyka się w tym samym rzędzie
  wielkości po użyciu poprawnego oporu dużego Macha; dokładne apogeum nie jest
  wymagane.

## 5. Skrypty

| skrypt | rola |
|---|---|
| `analyze_envelope_pkt3.py` | obwiednia V/q/Mach/apogeum per lot; CD wznoszenia (coast, akcelerometr); re-całkowanie closure 3 źródłami CD (fit / DATCOM / stały) |
| `diag_drag.py` | CD(Mach) z fazy ZNIŻANIA (GPS), istniejące `cd_datcom_vs_field_*` |
| `validate_trajectory.py` | closure 2D (reużyty integrator `simulate_ascent`) |

Wyniki: `results/envelope_pkt3_summary.csv`, `results/cd_ascent_coast.csv`,
`results/cd_datcom_vs_field_{ostra,tepa}.csv`, `results/base_drag_phase_factor_*.csv`.

## 6. Do zrobienia

- Pkt 4 — tabela zbiorcza Measured/status + absolute/relative: roll CLL ×4,
  pitch Cm_α ×1, **CD ×1 (DATCOM OK)**, V_max <5 %, q/Mach obwiednia, apogeum
  sanity check.
- (Opcjonalnie) scalić zmierzone CD(wznoszenie) + CD(zniżanie) w jedną krzywą
  obwiedni i zastąpić `fit_cd_mach_curves.csv` do closure.
