"""
datcom_io/config_reader.py
==========================
Czyta plik YAML z konfiguracją rakiety i waliduje dane.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    raise ImportError("Zainstaluj PyYAML: pip install pyyaml")


# ============================================================================
# Dataclasses — reprezentacja konfiguracji
# ============================================================================

@dataclass
class FlightConditions:
    mach:     List[float]
    alpha:    List[float]   # [deg]
    altitude: float = 0.0  # [m]


@dataclass
class Nose:
    type:   str     # ogive | cone | karman | haack
    length: float   # [m]


@dataclass
class Body:
    length:   float
    diameter: float
    nose:     Nose
    boattail:   Optional["Boattail"] = None
    sections_x: Optional[List[float]] = None  # [m] od nosa
    sections_r: Optional[List[float]] = None  # promienie [m]


@dataclass
class Boattail:
    diameter: float          # średnica ogona [m]
    length:   float          # długość zwężenia [m]
    type:     str = "cone"  # cone | ogive


@dataclass
class FinSet:
    name:       str
    count:      int
    position:   float   # x natarcia nasady od nosa [m]
    span:       float
    root_chord: float
    tip_chord:  float
    sweep_le:   float   # [deg]
    thickness:  float = 0.003
    profile:    str   = "hex"
    cant_angle: float = 0.0   # kąt zaklinowania [deg]


@dataclass
class ControlSurface:
    """
    Powierzchnia sterowa (np. canardy).

    Geometrycznie zachowuje sie jak FinSet — generator DATCOM zamienia ja na
    dodatkowy $FINSET (patrz missile_datcom_generator._effective_fin_sets),
    bo DATCOM nie ma osobnego pojecia "powierzchni sterowej": kazdy zestaw
    paneli to $FINSETn, a wychylenie zadaje sie przez DELTAn w $DEFLCT.
    """
    name:       str
    count:      int
    position:   float
    span:       float
    root_chord: float
    tip_chord:  float
    sweep_le:   float
    deflection: float = 0.0   # wychylenie referencyjne [deg]
    thickness:  float = 0.002
    profile:    str   = "hex"
    cant_angle: float = 0.0   # canardy zwykle bez zaklinowania
    # Zarezerwowane na przyszlosc — mapowanie kanalow sterowania na panele.
    # v1 uzywa standardowych wzorcow krzyzowych z control/derivatives.py.
    channel_map: Optional[dict] = None


@dataclass
class DualSpinSection:
    """Parametry jednej sekcji rakiety dual-spin."""
    mass: float         # [kg]
    Ixx:  float         # moment bezwładności roll [kg·m²]
    xcg:  float         # [m od nosa]
    Clp:  float = 0.0   # tłumienie roll [1/rad] — 0 gdy brak powierzchni

@dataclass
class DualSpinConfig:
    """Konfiguracja dual-spin — opcjonalna."""
    enabled:          bool
    bearing_x:        float          # pozycja łożyska [m od nosa]
    bearing_friction: float          # tarcie [N·m·s]
    forward:          DualSpinSection
    aft:              DualSpinSection

@dataclass
class Mass:
    xcg_ref: float   # [m] od nosa — referencyjne xcg dla DATCOM


@dataclass
class MassState:
    """Parametry masowe w jednym stanie (full lub empty)."""
    mass: float   # [kg]
    xcg:  float   # [m] od nosa
    Iyy:  float   # [kg·m²]
    Ixx:  float   # [kg·m²]


@dataclass
class MassModelConfig:
    """Parametry masowe dla modelu dynamiki."""
    full:  MassState
    empty: MassState


@dataclass
class PropulsionConfig:
    """Konfiguracja układu napędowego z profilem ciągu."""
    thrust_profile:   list   # lista par [t, F]
    nozzle_diameter:  float = 0.0   # [m]

    @property
    def t_ignition(self) -> float:
        return float(self.thrust_profile[0][0])

    @property
    def t_burnout(self) -> float:
        return float(self.thrust_profile[-1][0])

    @property
    def t_burn(self) -> float:
        return self.t_burnout - self.t_ignition


@dataclass
class RocketConfig:
    name:             str
    flight_conditions: FlightConditions
    body:             Body
    fins:             List[FinSet]
    mass:             Mass
    control_surfaces: List[ControlSurface] = field(default_factory=list)
    mass_model:       Optional["MassModelConfig"]  = None
    propulsion:       Optional["PropulsionConfig"] = None
    dual_spin:        Optional["DualSpinConfig"]   = None


# ============================================================================
# Parser
# ============================================================================

def load_config(yaml_path: Path) -> RocketConfig:
    """
    Wczytuje plik YAML i zwraca RocketConfig.

    Parameters
    ----------
    yaml_path : Path
        Ścieżka do pliku .yaml z konfiguracją.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Plik konfiguracji nie istnieje: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # ---- Flight conditions ------------------------------------------- #
    fc_data = data["flight_conditions"]
    fc = FlightConditions(
        mach     = [float(m) for m in fc_data["mach"]],
        alpha    = [float(a) for a in fc_data["alpha"]],
        altitude = float(fc_data.get("altitude", 0.0)),
    )

    # ---- Body --------------------------------------------------------- #
    b_data = data["body"]
    nose   = Nose(
        type   = b_data["nose"]["type"],
        length = float(b_data["nose"]["length"]),
    )
    sections = b_data.get("sections", None)
    bt_data  = b_data.get("boattail", None)
    boattail = Boattail(
        diameter = float(bt_data["diameter"]),
        length   = float(bt_data["length"]),
        type     = bt_data.get("type", "cone"),
    ) if bt_data else None

    body = Body(
        length     = float(b_data["length"]),
        diameter   = float(b_data["diameter"]),
        nose       = nose,
        boattail   = boattail,
        sections_x = [float(x) for x in sections["x"]] if sections else None,
        sections_r = [float(r) for r in sections["r"]] if sections else None,
    )

    # ---- Fins --------------------------------------------------------- #
    fins = []
    for f in data.get("fins", []):
        fins.append(FinSet(
            name       = f.get("name", "fins"),
            count      = int(f["count"]),
            position   = float(f["position"]),
            span       = float(f["span"]),
            root_chord = float(f["root_chord"]),
            tip_chord  = float(f["tip_chord"]),
            sweep_le   = float(f["sweep_le"]),
            thickness  = float(f.get("thickness", 0.003)),
            cant_angle = float(f.get("cant_angle", 0.0)),
            profile    = f.get("profile", "hex"),
        ))

    # ---- Control surfaces -------------------------------------------- #
    control = []
    for cs in data.get("control_surfaces", []):
        control.append(ControlSurface(
            name       = cs.get("name", "control"),
            count      = int(cs["count"]),
            position   = float(cs["position"]),
            span       = float(cs["span"]),
            root_chord = float(cs["root_chord"]),
            tip_chord  = float(cs["tip_chord"]),
            sweep_le   = float(cs["sweep_le"]),
            deflection = float(cs.get("deflection", 0.0)),
            thickness  = float(cs.get("thickness", 0.002)),
            profile    = cs.get("profile", "hex"),
            cant_angle = float(cs.get("cant_angle", 0.0)),
            channel_map= cs.get("channel_map", None),
        ))

    # ---- Mass --------------------------------------------------------- #
    mass = Mass(xcg_ref=float(data["mass"]["xcg_ref"]))

    # ---- Mass model -------------------------------------------------- #
    mm_data    = data.get("mass_model", None)
    mass_model = None
    if mm_data:
        def parse_state(s):
            return MassState(
                mass = float(s["mass"]),
                xcg  = float(s["xcg"]),
                Iyy  = float(s["Iyy"]),
                Ixx  = float(s["Ixx"]),
            )
        mass_model = MassModelConfig(
            full  = parse_state(mm_data["full"]),
            empty = parse_state(mm_data["empty"]),
        )

    # ---- Propulsion -------------------------------------------------- #
    prop_data  = data.get("propulsion", None)
    propulsion = None
    if prop_data:
        propulsion = PropulsionConfig(
            thrust_profile  = [[float(t), float(f)] for t, f in prop_data["thrust_profile"]],
            nozzle_diameter = float(prop_data.get("nozzle_diameter", 0.0)),
        )

    # ---- Dual-spin (opcjonalne) ----------------------------------------- #
    ds_data   = data.get("dual_spin", None)
    dual_spin = None
    if ds_data is not None:
        dual_spin = DualSpinConfig(
            enabled          = bool(ds_data.get("enabled", False)),
            bearing_x        = float(ds_data.get("bearing_x", 0.0)),
            bearing_friction = float(ds_data.get("bearing_friction", 0.0)),
            forward = DualSpinSection(
                mass = float(ds_data["forward"]["mass"]),
                Ixx  = float(ds_data["forward"]["Ixx"]),
                xcg  = float(ds_data["forward"]["xcg"]),
                Clp  = float(ds_data["forward"].get("Clp", 0.0)),
            ),
            aft = DualSpinSection(
                mass = float(ds_data["aft"]["mass"]),
                Ixx  = float(ds_data["aft"]["Ixx"]),
                xcg  = float(ds_data["aft"]["xcg"]),
                Clp  = float(ds_data["aft"].get("Clp", 0.0)),
            ),
        )

    return RocketConfig(
        name              = data.get("name", yaml_path.stem),
        flight_conditions = fc,
        body              = body,
        fins              = fins,
        mass              = mass,
        control_surfaces  = control,
        mass_model        = mass_model,
        dual_spin         = dual_spin,
        propulsion        = propulsion,
    )
