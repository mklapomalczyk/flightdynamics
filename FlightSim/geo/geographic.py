"""
geo/geographic.py
=================
Moduł geograficzny — przelicza trajektorię z lokalnego układu launch frame
na współrzędne geograficzne WGS84 i rysuje na mapie.

Założenia:
  - Płaska Ziemia (wystarczające dla zasięgów < 50km)
  - Brak rotacji Ziemi
  - Launch frame: X w kierunku azymutu, Y w prawo (prostopadle), Z w dół

Użycie:
    from geo.geographic import GeoModule, MissionConfig

    mission = MissionConfig.from_yaml("missions/mission_01.yaml")
    geo     = GeoModule(mission)
    geo.plot(result, output_path="trajectory_map.png")
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Konfiguracja misji
# ============================================================================

@dataclass
class MissionConfig:
    """Parametry misji — pozycja wyrzutni i kąty strzału."""
    name:      str
    lat:       float   # szerokość geograficzna wyrzutni [deg]
    lon:       float   # długość geograficzna wyrzutni [deg]
    alt:       float   # wysokość wyrzutni n.p.m. [m]
    azimuth:   float   # azymut strzału [deg], 0=N, 90=E
    elevation: float   # kąt elewacji [deg]

    @classmethod
    def from_yaml(cls, yaml_path) -> "MissionConfig":
        try:
            import yaml
        except ImportError:
            raise ImportError("Zainstaluj PyYAML: pip install pyyaml")

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        lnch = data["launcher"]
        return cls(
            name      = data.get("name", Path(yaml_path).stem),
            lat       = float(lnch["latitude"]),
            lon       = float(lnch["longitude"]),
            alt       = float(lnch.get("altitude", 0.0)),
            azimuth   = float(lnch["azimuth"]),
            elevation = float(lnch["elevation"]),
        )

    @classmethod
    def from_values(
        cls,
        lat: float, lon: float,
        azimuth: float, elevation: float,
        alt: float = 0.0,
        name: str = "mission",
    ) -> "MissionConfig":
        """Tworzy konfigurację misji bezpośrednio z wartości."""
        return cls(
            name=name, lat=lat, lon=lon,
            alt=alt, azimuth=azimuth, elevation=elevation,
        )


# ============================================================================
# Przeliczenie układów współrzędnych
# ============================================================================

# Promień Ziemi [m]
R_EARTH = 6_371_000.0


def launch_to_geo(
    x: np.ndarray,
    y: np.ndarray,
    lat0: float,
    lon0: float,
    azimuth_deg: float = 0.0,
) -> tuple:
    """
    Przelicza pozycje z Launch Frame na WGS84 (lat, lon).

    Launch Frame: X wzdłuż azymutu, Y w prawo prostopadle, Z w dół.
    Solver oblicza trajektorię w LF — tu przeliczamy LF → NED → WGS84.

    LF → NED (obrót o azymut wokół Z):
      dN =  x_LF * cos(az) - y_LF * sin(az)
      dE =  x_LF * sin(az) + y_LF * cos(az)

    Parameters
    ----------
    x, y : np.ndarray  Pozycje w Launch Frame [m]
    lat0, lon0 : float  Pozycja wyrzutni [deg]
    azimuth_deg : float  Azymut [deg], 0=N, 90=E
    """
    az       = np.deg2rad(azimuth_deg)
    lat0_rad = np.deg2rad(lat0)

    # LF → NED
    dN =  x * np.cos(az) - y * np.sin(az)
    dE =  x * np.sin(az) + y * np.cos(az)

    dlat = np.degrees(dN / R_EARTH)
    dlon = np.degrees(dE / (R_EARTH * np.cos(lat0_rad)))

    return lat0 + dlat, lon0 + dlon


# ============================================================================
# Główna klasa modułu
# ============================================================================

class GeoModule:
    """
    Moduł geograficzny — przelicza i wizualizuje trajektorię.

    Parameters
    ----------
    mission : MissionConfig
        Konfiguracja misji z pozycją wyrzutni i azymutem.
    """

    def __init__(self, mission: MissionConfig):
        self.mission = mission

    def trajectory_to_geo(self, result) -> dict:
        """
        Przelicza trajektorię symulacji na współrzędne WGS84.

        Parameters
        ----------
        result : SimResult6DOF  Wynik symulacji 6DOF.

        Returns
        -------
        dict z kluczami: lat, lon, alt, t
        """
        m = self.mission
        lat, lon = launch_to_geo(
            x            = result.x,
            y            = result.y,
            lat0         = m.lat,
            lon0         = m.lon,
            azimuth_deg  = m.azimuth,   # LF→NED obrót o azymut
        )
        # Wysokość: alt_wyrzutni - z (Z_launch w dół → wysokość = -z)
        alt = m.alt + (-result.z)

        return {
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "t":   result.t,
        }

    def plot(
        self,
        result,
        output_path: str = "trajectory_map.png",
        use_basemap:  bool = True,
        title: Optional[str] = None,
    ) -> Path:
        """
        Rysuje trajektorię na mapie i zapisuje do PNG.

        Parameters
        ----------
        result : SimResult6DOF
        output_path : str  Ścieżka do pliku PNG.
        use_basemap : bool  Czy pobierać tło z OpenStreetMap (wymaga internetu).
        title : str, optional  Tytuł wykresu.

        Returns
        -------
        Path  Ścieżka do zapisanego pliku.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        geo  = self.trajectory_to_geo(result)
        lat  = geo["lat"]
        lon  = geo["lon"]
        m    = self.mission

        # Punkt startu i lądowania
        lat_start, lon_start = lat[0],  lon[0]
        lat_end,   lon_end   = lat[-1], lon[-1]

        # Zasięg poziomy [m]
        dx  = result.x[-1] - result.x[0]
        dy  = result.y[-1] - result.y[0]
        range_m = np.sqrt(dx**2 + dy**2)

        fig, ax = plt.subplots(figsize=(10, 8))

        if use_basemap:
            try:
                import contextily as ctx
                from pyproj import Transformer

                # Margines wokół trajektorii
                margin = max(range_m * 0.15, 500.0)

                lat_min = min(lat.min(), lat_start, lat_end) - np.degrees(margin/R_EARTH)
                lat_max = max(lat.max(), lat_start, lat_end) + np.degrees(margin/R_EARTH)
                lon_min = min(lon.min(), lon_start, lon_end) - np.degrees(margin/(R_EARTH*np.cos(np.deg2rad(m.lat))))
                lon_max = max(lon.max(), lon_start, lon_end) + np.degrees(margin/(R_EARTH*np.cos(np.deg2rad(m.lat))))

                # Transformacja WGS84 → Web Mercator (EPSG:3857)
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

                lon_m, lat_m     = transformer.transform(lon, lat)
                lon_s, lat_s     = transformer.transform(lon_start, lat_start)
                lon_e, lat_e     = transformer.transform(lon_end, lat_end)

                ax.plot(lon_m, lat_m, 'r-', lw=2.5, zorder=5, label="Trajektoria")
                ax.plot(lon_s, lat_s, 'go', ms=10, zorder=6, label="Start")
                ax.plot(lon_e, lat_e, 'rx', ms=12, mew=2.5, zorder=6, label="Upadek")

                ax.set_xlim(
                    *transformer.transform([lon_min, lon_max], [m.lat, m.lat])[0]
                )
                ax.set_ylim(
                    *transformer.transform([m.lon, m.lon], [lat_min, lat_max])[1]
                )

                ctx.add_basemap(
                    ax,
                    source   = ctx.providers.OpenStreetMap.Mapnik,
                    zoom     = "auto",
                    crs      = "EPSG:3857",
                    alpha    = 0.8,
                )

                # Tick labels w stopniach geograficznych
                inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

                # X — długość geograficzna
                x_ticks = ax.get_xticks()
                x_ticks = x_ticks[(x_ticks >= ax.get_xlim()[0]) & (x_ticks <= ax.get_xlim()[1])]
                x_labels = [f"{inv.transform(xt, (ax.get_ylim()[0]+ax.get_ylim()[1])/2)[0]:.4f}°E"
                            for xt in x_ticks]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, fontsize=8)

                # Y — szerokość geograficzna
                y_ticks = ax.get_yticks()
                y_ticks = y_ticks[(y_ticks >= ax.get_ylim()[0]) & (y_ticks <= ax.get_ylim()[1])]
                y_labels = [f"{inv.transform((ax.get_xlim()[0]+ax.get_xlim()[1])/2, yt)[1]:.4f}°N"
                            for yt in y_ticks]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels, fontsize=8)

                ax.set_xlabel("Długość geograficzna [°E]")
                ax.set_ylabel("Szerokość geograficzna [°N]")

                # Skala
                self._add_scale_bar(ax, range_m, mode="mercator",
                                    transformer=transformer, lat=m.lat, lon=m.lon)

            except ImportError:
                print("[GeoModule] contextily lub pyproj niedostępne — rysuję bez tła mapy.")
                use_basemap = False

        if not use_basemap:
            # Wersja bez tła — czyste współrzędne geograficzne
            ax.plot(lon, lat, 'r-', lw=2.5, zorder=5, label="Trajektoria")
            ax.plot(lon_start, lat_start, 'go', ms=10, zorder=6, label="Start")
            ax.plot(lon_end, lat_end, 'rx', ms=12, mew=2.5, zorder=6, label="Upadek")

            ax.set_xlabel("Długość geograficzna [°E]")
            ax.set_ylabel("Szerokość geograficzna [°N]")
            ax.grid(alpha=0.3)

            # Skala
            self._add_scale_bar(ax, range_m, mode="geo", lat=m.lat)

        # ---- Tytuł i legenda -------------------------------------------- #
        if title is None:
            title = (
                f"{m.name}  |  "
                f"Az={m.azimuth:.1f}°  El={m.elevation:.1f}°  |  "
                f"Zasięg: {range_m/1000:.2f} km"
            )
        ax.set_title(title, fontsize=11, pad=12)
        ax.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[GeoModule] Zapisano: {output_path}")
        return output_path

    # ------------------------------------------------------------------ #

    def ned_to_lf(self, result) -> dict:
        """
        Zwraca trajektorię w Launch Frame.

        Solver oblicza bezpośrednio w LF (X wzdłuż azymutu, Y w prawo,
        Z w dół) — ta funkcja jest tożsamością, zachowana dla
        kompatybilności API.

        Returns
        -------
        dict z kluczami: x_lf, y_lf, z_lf, t, altitude
        """
        alt = self.mission.alt + (-result.z)   # wysokość n.p.m.
        return {
            "x_lf":    result.x,
            "y_lf":    result.y,
            "z_lf":    result.z,
            "alt":     alt,
            "t":       result.t,
        }

    def euler_lf(self, result) -> dict:
        """
        Zwraca kąty Eulera w Launch Frame.

        Solver oblicza bezpośrednio w LF — kwaternion opisuje orientację
        body względem LF. Kąty liczone wprost z kwaterniona.

        Returns
        -------
        dict z kluczami: psi_deg, theta_deg, phi_deg, t
          psi=0 gdy rakieta skierowana wzdłuż X_LF (kierunek strzału)
          theta=elewacja na starcie
        """
        from core.quaternion import quat_to_euler_zyx as q2e

        n = len(result.t)
        psi_arr   = np.zeros(n)
        theta_arr = np.zeros(n)
        phi_arr   = np.zeros(n)

        for i in range(n):
            q = np.array([result.q0[i], result.q1[i],
                          result.q2[i], result.q3[i]])
            euler = q2e(q)
            psi_arr[i]   = np.degrees(euler[0])
            theta_arr[i] = np.degrees(euler[1])
            phi_arr[i]   = np.degrees(euler[2])

        return {
            "psi_deg":   psi_arr,
            "theta_deg": theta_arr,
            "phi_deg":   phi_arr,
            "t":         result.t,
        }

    def plot_lf(
        self,
        result,
        output_path: str = "trajectory_lf.png",
        title: Optional[str] = None,
    ) -> "Path":
        """
        Rysuje rzut trajektorii w Launch Frame i zapisuje do PNG.

        Dwa subploty:
          - Płaszczyzna strzału: x_LF (zasięg wzdłuż azymutu) vs wysokość
          - Rzut poziomy: x_LF vs y_LF (odchylenie boczne)

        Parameters
        ----------
        result : SimResult6DOF
        output_path : str
        title : str, optional
        """
        import matplotlib.pyplot as plt

        lf  = self.ned_to_lf(result)
        m   = self.mission

        x   = lf["x_lf"]
        y   = lf["y_lf"]
        alt = lf["alt"]

        range_m = float(x[-1])   # zasięg wzdłuż azymutu
        lat_dev = float(y[-1])   # odchylenie boczne w punkcie upadku

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # ---- Płaszczyzna strzału ----------------------------------------
        ax1 = axes[0]
        ax1.plot(x / 1000, alt, 'b-', lw=2)
        ax1.plot(x[0]  / 1000, alt[0],  'go', ms=9, label="Start")
        ax1.plot(x[-1] / 1000, alt[-1], 'rx', ms=11, mew=2.5, label="Upadek")
        ax1.set_xlabel("Zasięg wzdłuż azymutu [km]")
        ax1.set_ylabel("Wysokość [m]")
        ax1.set_title("Płaszczyzna strzału (Launch Frame)")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(bottom=0)

        # Adnotacja zasięgu
        ax1.annotate(
            f"Zasięg: {range_m/1000:.2f} km",
            xy=(x[-1]/1000, alt[-1]),
            xytext=(x[-1]/1000 - 0.3, alt[-1] + alt.max()*0.08),
            fontsize=8, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1),
        )

        # ---- Rzut poziomy -----------------------------------------------
        ax2 = axes[1]
        ax2.plot(x / 1000, y, 'b-', lw=2)
        ax2.plot(x[0]  / 1000, y[0],  'go', ms=9, label="Start")
        ax2.plot(x[-1] / 1000, y[-1], 'rx', ms=11, mew=2.5, label="Upadek")
        ax2.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
        ax2.set_xlabel("Zasięg wzdłuż azymutu [km]")
        ax2.set_ylabel("Odchylenie boczne Y_LF [m]")
        ax2.set_title("Rzut poziomy (Launch Frame)")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # Adnotacja odchylenia
        ax2.annotate(
            f"Δy = {lat_dev:.1f} m",
            xy=(x[-1]/1000, y[-1]),
            xytext=(x[-1]/1000 - 0.3, y[-1] + (y.max()-y.min())*0.1 + 5),
            fontsize=8, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1),
        )

        # ---- Tytuł -------------------------------------------------------
        if title is None:
            title = (
                f"{m.name}  |  Az={m.azimuth:.1f}°  El={m.elevation:.1f}°  |  "
                f"Zasięg: {range_m/1000:.2f} km  Δy: {lat_dev:.1f} m"
            )
        fig.suptitle(title, fontsize=11)
        plt.tight_layout()

        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[GeoModule] Zapisano LF: {output_path}")
        return output_path

    def plot_interactive(
        self,
        result,
        output_path: str = "trajectory_map.html",
        title: Optional[str] = None,
    ) -> Path:
        """
        Rysuje interaktywną mapę w HTML z możliwością przełączania warstw.

        Warstwy:
          - OpenStreetMap (domyślna)
          - Esri Satellite (zdjęcia satelitarne)
          - Esri World Topo

        Parameters
        ----------
        result : SimResult6DOF
        output_path : str  Ścieżka do pliku HTML.
        title : str, optional

        Returns
        -------
        Path  Ścieżka do zapisanego pliku HTML.
        """
        try:
            import folium
        except ImportError:
            raise ImportError("Zainstaluj folium: pip install folium")

        import numpy as np

        geo = self.trajectory_to_geo(result)
        lat = geo["lat"]
        lon = geo["lon"]
        m   = self.mission

        # Punkt startu i lądowania
        lat_start, lon_start = float(lat[0]),  float(lon[0])
        lat_end,   lon_end   = float(lat[-1]), float(lon[-1])

        # Zasięg poziomy
        dx      = result.x[-1] - result.x[0]
        dy      = result.y[-1] - result.y[0]
        range_m = float(np.sqrt(dx**2 + dy**2))

        # Środek mapy
        lat_center = float(np.mean([lat_start, lat_end]))
        lon_center = float(np.mean([lon_start, lon_end]))

        # Dobierz zoom
        if range_m < 500:
            zoom = 15
        elif range_m < 2000:
            zoom = 14
        elif range_m < 8000:
            zoom = 13
        else:
            zoom = 12

        # Mapa bazowa
        fmap = folium.Map(
            location    = [lat_center, lon_center],
            zoom_start  = zoom,
            tiles       = None,   # dodamy warstwy ręcznie
        )

        # Warstwy
        folium.TileLayer(
            tiles     = "OpenStreetMap",
            name      = "OpenStreetMap",
            control   = True,
        ).add_to(fmap)

        folium.TileLayer(
            tiles     = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr      = "Esri",
            name      = "Satelita (Esri)",
            overlay   = False,
            control   = True,
        ).add_to(fmap)

        folium.TileLayer(
            tiles     = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr      = "Esri",
            name      = "Topografia (Esri)",
            overlay   = False,
            control   = True,
        ).add_to(fmap)

        # Trajektoria
        coords = list(zip(lat.tolist(), lon.tolist()))
        folium.PolyLine(
            locations  = coords,
            color      = "red",
            weight     = 3,
            opacity    = 0.9,
            tooltip    = f"Trajektoria | Zasięg: {range_m/1000:.2f} km",
        ).add_to(fmap)

        # Punkt startowy
        folium.Marker(
            location  = [lat_start, lon_start],
            tooltip   = f"Start | Az={m.azimuth:.1f}° El={m.elevation:.1f}°",
            icon      = folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(fmap)

        # Punkt upadku
        folium.Marker(
            location  = [lat_end, lon_end],
            tooltip   = f"Upadek | Zasięg: {range_m/1000:.2f} km",
            icon      = folium.Icon(color="red", icon="times", prefix="fa"),
        ).add_to(fmap)

        # Linia zasięgu (start → upadek)
        folium.PolyLine(
            locations  = [[lat_start, lon_start], [lat_end, lon_end]],
            color      = "gray",
            weight     = 1.5,
            opacity    = 0.6,
            dash_array = "6 4",
            tooltip    = f"Zasięg poziomy: {range_m/1000:.2f} km",
        ).add_to(fmap)

        # Kontrolka warstw
        folium.LayerControl(collapsed=False).add_to(fmap)

        # Tytuł
        if title is None:
            title = (f"{m.name} | Az={m.azimuth:.1f}° El={m.elevation:.1f}° "
                     f"| Zasięg: {range_m/1000:.2f} km")

        title_html = f"""
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    z-index: 1000; background: rgba(255,255,255,0.9);
                    padding: 8px 16px; border-radius: 6px;
                    font-family: sans-serif; font-size: 13px; font-weight: bold;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
            {title}
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(title_html))

        output_path = Path(output_path)
        fmap.save(str(output_path))
        print(f"[GeoModule] Zapisano mapę interaktywną: {output_path}")
        return output_path

    def _add_scale_bar(self, ax, range_m, mode, **kwargs):
        """Dodaje pasek skali do wykresu."""
        # Dobierz ładną wartość skali
        scale_m = 10 ** np.floor(np.log10(range_m / 3))
        if range_m / scale_m > 6:
            scale_m *= 2
        scale_km = scale_m / 1000.0

        if mode == "mercator":
            transformer = kwargs["transformer"]
            lat         = kwargs["lat"]
            lon_ref     = kwargs["lon"]
            x0, y0 = transformer.transform(lon_ref, lat)
            # 1 stopień lon w Mercator
            x1, _  = transformer.transform(
                lon_ref + np.degrees(scale_m / (R_EARTH * np.cos(np.deg2rad(lat)))),
                lat
            )
            scale_px = abs(x1 - x0)

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_bar = xlim[0] + 0.05 * (xlim[1] - xlim[0])
            y_bar = ylim[0] + 0.04 * (ylim[1] - ylim[0])

            ax.plot([x_bar, x_bar + scale_px], [y_bar, y_bar],
                    'k-', lw=3, zorder=10)
            ax.text(x_bar + scale_px/2, y_bar + 0.008*(ylim[1]-ylim[0]),
                    f"{scale_km:.1f} km" if scale_km >= 1 else f"{scale_m:.0f} m",
                    ha='center', va='bottom', fontsize=8, zorder=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            scale_deg = np.degrees(scale_m / (R_EARTH * np.cos(np.deg2rad(kwargs["lat"]))))

            x_bar = xlim[0] + 0.05 * (xlim[1] - xlim[0])
            y_bar = ylim[0] + 0.04 * (ylim[1] - ylim[0])

            ax.plot([x_bar, x_bar + scale_deg], [y_bar, y_bar],
                    'k-', lw=3, zorder=10)
            ax.text(x_bar + scale_deg/2, y_bar + 0.008*(ylim[1]-ylim[0]),
                    f"{scale_km:.1f} km" if scale_km >= 1 else f"{scale_m:.0f} m",
                    ha='center', va='bottom', fontsize=8, zorder=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
