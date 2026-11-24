"""France BAAC Ingestion Adapter.
Parses, cleans, and normalizes French Ministère de l'Intérieur / ONISR BAAC road safety data into the canonical schema.
Licence: Licence Ouverte / Open Licence v2.0 (Etalab).
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas.canonical_accident import (
    CountryCode,
    LicenseType,
    CanonicalRoadClass,
    RoadSurfaceCondition,
    LightCondition,
    WeatherConditionCategory,
    HumanInjurySeverity
)

class FranceBAACAdapter:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (PROJECT_ROOT / "data" / "external" / "france_baac")

    def ingest(self, limit: Optional[int] = None) -> pl.DataFrame:
        """Ingests France BAAC characteristics, locations, users, and vehicles into a canonical Polars DataFrame."""
        caract_path = self.data_dir / "caract-2023.csv"
        lieux_path = self.data_dir / "lieux-2023.csv"
        usagers_path = self.data_dir / "usagers-2023.csv"
        vehicules_path = self.data_dir / "vehicules-2023.csv"

        if not caract_path.exists():
            raise FileNotFoundError(f"France BAAC characteristics file not found at {caract_path}")

        # 1. Aggregate casualties per accident
        if usagers_path.exists():
            usagers_df = pl.read_csv(usagers_path, separator=";", ignore_errors=True)
            usagers_agg = usagers_df.group_by("Num_Acc").agg([
                (pl.col("grav") == 2).sum().alias("fatal_count"),
                (pl.col("grav") == 3).sum().alias("serious_count"),
                (pl.col("grav") == 4).sum().alias("slight_count"),
                pl.len().alias("total_usagers")
            ])
        else:
            usagers_agg = None

        # 2. Count vehicles per accident
        if vehicules_path.exists():
            veh_df = pl.read_csv(vehicules_path, separator=";", ignore_errors=True)
            veh_agg = veh_df.group_by("Num_Acc").agg([
                pl.len().alias("num_vehicles")
            ])
        else:
            veh_agg = None

        # 3. Read Lieux (roadway infrastructure)
        if lieux_path.exists():
            lieux_df = pl.read_csv(lieux_path, separator=";", ignore_errors=True)
            # De-duplicate by Num_Acc if multiple segments
            lieux_df = lieux_df.unique(subset=["Num_Acc"])
        else:
            lieux_df = None

        # 4. Read Caractéristiques
        caract_df = pl.read_csv(caract_path, separator=";", ignore_errors=True)
        if limit:
            caract_df = caract_df.head(limit)

        # Joins
        if lieux_df is not None:
            caract_df = caract_df.join(lieux_df, on="Num_Acc", how="left")
        if usagers_agg is not None:
            caract_df = caract_df.join(usagers_agg, on="Num_Acc", how="left")
        if veh_agg is not None:
            caract_df = caract_df.join(veh_agg, on="Num_Acc", how="left")

        records = []
        for row in caract_df.iter_rows(named=True):
            # Parse Lat/Lon with decimal comma replacement
            raw_lat = row.get("lat")
            raw_lon = row.get("long")
            if not raw_lat or not raw_lon:
                continue

            try:
                lat = float(str(raw_lat).replace(",", ".").strip())
                lon = float(str(raw_lon).replace(",", ".").strip())
            except Exception:
                continue

            # Validate WGS84 coordinates (France mainland + DOM/TOM range)
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            if lat == 0.0 and lon == 0.0:
                continue

            # Parse Date & Time
            year = int(row.get("an", 2023))
            month = int(row.get("mois", 1))
            day = int(row.get("jour", 1))
            hrmn = str(row.get("hrmn", "12:00")).strip()
            
            hour, minute = 12, 0
            if ":" in hrmn:
                parts = hrmn.split(":")
                try:
                    hour = int(parts[0])
                    minute = int(parts[1]) if len(parts) > 1 else 0
                except Exception:
                    hour, minute = 12, 0
            elif len(hrmn) in (3, 4):
                try:
                    hour = int(hrmn[:-2])
                    minute = int(hrmn[-2:])
                except Exception:
                    hour, minute = 12, 0

            try:
                dt = datetime(year, month, day, hour, minute)
            except Exception:
                dt = datetime(2023, 1, 1, 12, 0)

            # Map Road Class (catr)
            catr = row.get("catr", 4)
            if catr == 1:
                road_cls = CanonicalRoadClass.MOTORWAY.value
            elif catr == 2:
                road_cls = CanonicalRoadClass.TRUNK_HIGHWAY.value
            elif catr == 3:
                road_cls = CanonicalRoadClass.PRIMARY_ARTERIAL.value
            elif catr == 4:
                road_cls = CanonicalRoadClass.SECONDARY_COLLECTOR.value
            elif catr == 5:
                road_cls = CanonicalRoadClass.LOCAL_RESIDENTIAL.value
            elif catr == 6:
                road_cls = CanonicalRoadClass.SERVICE_ACCESS.value
            else:
                road_cls = CanonicalRoadClass.LOCAL_RURAL.value

            # Speed Limit (vma)
            raw_vma = row.get("vma")
            try:
                vma = float(str(raw_vma).replace(",", ".").strip()) if raw_vma is not None else 50.0
                if vma <= 0 or vma > 200:
                    vma = 50.0
            except Exception:
                vma = 50.0

            # Junction (int) & Infrastructure
            intersection_type = row.get("int", 1)
            is_junc = bool(intersection_type and intersection_type > 1)
            is_round = (intersection_type == 6) # Giratoire
            is_rail = (intersection_type == 8) # Passage a niveau

            # Environmental: Light (lum)
            lum = row.get("lum", 1)
            if lum == 1:
                light = LightCondition.DAYLIGHT.value
            elif lum == 2:
                light = LightCondition.TWILIGHT_DAWN_DUSK.value
            elif lum == 3:
                light = LightCondition.DARKNESS_NO_STREETLIGHTS.value
            elif lum == 4:
                light = LightCondition.DARKNESS_STREETLIGHTS_OFF.value
            elif lum == 5:
                light = LightCondition.DARKNESS_STREETLIGHTS_ON.value
            else:
                light = LightCondition.UNKNOWN.value

            # Weather (atm)
            atm = row.get("atm", 1)
            if atm in (1, 7):
                weather = WeatherConditionCategory.CLEAR.value
            elif atm == 2:
                weather = WeatherConditionCategory.LIGHT_RAIN.value
            elif atm == 3:
                weather = WeatherConditionCategory.HEAVY_RAIN.value
            elif atm == 4:
                weather = WeatherConditionCategory.SNOW_HAIL.value
            elif atm == 5:
                weather = WeatherConditionCategory.FOG_SMOKE.value
            elif atm == 6:
                weather = WeatherConditionCategory.HIGH_WINDS_STORM.value
            elif atm == 8:
                weather = WeatherConditionCategory.CLOUDY_OVERCAST.value
            else:
                weather = WeatherConditionCategory.OTHER.value

            # Road surface (surf)
            surf = row.get("surf", 1)
            if surf == 1:
                surface = RoadSurfaceCondition.DRY.value
            elif surf in (2, 3):
                surface = RoadSurfaceCondition.WET_DAMP.value
            elif surf == 4:
                surface = RoadSurfaceCondition.FLOOD.value
            elif surf == 5:
                surface = RoadSurfaceCondition.SNOW_SLUSH.value
            elif surf == 6:
                surface = RoadSurfaceCondition.MUD_OIL.value
            elif surf == 7:
                surface = RoadSurfaceCondition.FROST_ICE.value
            else:
                surface = RoadSurfaceCondition.UNKNOWN.value

            # Casualties & Human Injury Severity
            fatals = int(row.get("fatal_count", 0) or 0)
            serious = int(row.get("serious_count", 0) or 0)
            slights = int(row.get("slight_count", 0) or 0)
            total_cas = fatals + serious + slights

            if fatals > 0:
                injury_sev = HumanInjurySeverity.FATAL.value
            elif serious > 0:
                injury_sev = HumanInjurySeverity.SERIOUS.value
            elif slights > 0:
                injury_sev = HumanInjurySeverity.SLIGHT.value
            else:
                injury_sev = HumanInjurySeverity.UNINJURED.value

            records.append({
                "global_accident_id": f"FR-BAAC-{row.get('Num_Acc')}",
                "source_record_id": str(row.get("Num_Acc")),
                "source_dataset": "FRANCE_ONISR_BAAC_2023",
                "country_code": CountryCode.FR.value,
                "licence_type": LicenseType.LICENCE_OUVERTE_V2.value,
                "attribution_text": "Ministère de l'Intérieur / ONISR - Données des accidents corporels de la circulation routière - Licence Ouverte v2.0",
                "latitude": lat,
                "longitude": lon,
                "coordinate_precision": "HIGH_GPS",
                "administrative_region": str(row.get("dep", "FR")),
                "locality_or_city": str(row.get("com", "")),
                "timestamp_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "local_timestamp": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "timezone_iana": "Europe/Paris",
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
                "minute": dt.minute,
                "day_of_week": dt.isoweekday(),
                "is_weekend": dt.isoweekday() in (6, 7),
                "is_night": light in (LightCondition.DARKNESS_STREETLIGHTS_ON.value, LightCondition.DARKNESS_NO_STREETLIGHTS.value),
                "road_class": road_cls,
                "speed_limit_kmh": vma,
                "is_junction": is_junc,
                "is_traffic_signal": False, # Detailed in infra if recorded
                "is_crossing": False,
                "is_roundabout": is_round,
                "is_stop_sign": False,
                "is_give_way": False,
                "is_railway": is_rail,
                "is_traffic_calming": False,
                "number_of_lanes": None,
                "divided_roadway": (str(row.get("circ")) == "3"),
                "weather_condition": weather,
                "light_condition": light,
                "road_surface_condition": surface,
                "temperature_c": None,
                "humidity_pct": None,
                "pressure_hpa": None,
                "visibility_km": None,
                "wind_speed_kmh": None,
                "precipitation_mm": None,
                "num_vehicles_involved": int(row.get("num_vehicles", 1) or 1),
                "num_casualties_total": total_cas,
                "casualty_fatal_count": fatals,
                "casualty_serious_count": serious,
                "casualty_slight_count": slights,
                "human_injury_severity": injury_sev,
                "traffic_impact_severity": None,
                "annual_average_daily_flow": None,
                "distance_to_traffic_counter_km": None
            })

        return pl.DataFrame(records)
