"""UK DfT STATS19 Ingestion Adapter.
Parses, cleans, and normalizes UK Department for Transport road safety data into the canonical schema.
Supports spatial join with GB AADF traffic counts for exposure estimation.
Licence: Open Government Licence v3.0 (OGL).
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import numpy as np
import polars as pl
from scipy.spatial import KDTree

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

class UKStats19Adapter:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (PROJECT_ROOT / "data" / "external" / "uk_stats19")
        self.traffic_dir = PROJECT_ROOT / "data" / "external" / "gb_traffic"
        self._traffic_tree = None
        self._traffic_df = None

    def load_traffic_exposure_index(self):
        """Loads GB AADF traffic counts and builds a 2D spatial index for exposure joins."""
        traffic_csv = self.traffic_dir / "dft_traffic_counts_aadf.csv"
        if not traffic_csv.exists():
            return
        
        # Load count points for recent year (e.g. >= 2020) to keep latest exposure
        df = pl.read_csv(
            traffic_csv,
            columns=["count_point_id", "year", "road_name", "latitude", "longitude", "all_motor_vehicles"],
            ignore_errors=True
        ).filter(pl.col("year") >= 2020).filter(pl.col("latitude").is_not_null())

        # Keep latest record per count point
        df = df.sort("year", descending=True).unique(subset=["count_point_id"])
        self._traffic_df = df

        coords = df.select(["latitude", "longitude"]).to_numpy()
        # Convert lat/lon degrees to approximate metric coordinates (1 deg lat ~ 111 km, 1 deg lon ~ 65 km in UK)
        coords_metric = np.column_stack([coords[:, 0] * 111.0, coords[:, 1] * 65.0])
        self._traffic_tree = KDTree(coords_metric)

    def find_nearest_traffic_exposure(self, lat: float, lon: float, max_dist_km: float = 5.0) -> tuple:
        """Finds nearest traffic counting point within max_dist_km."""
        if self._traffic_tree is None:
            return None, None
        
        target = np.array([lat * 111.0, lon * 65.0])
        dist, idx = self._traffic_tree.query(target)
        if dist <= max_dist_km:
            row = self._traffic_df.row(idx)
            # row: (count_point_id, year, road_name, latitude, longitude, all_motor_vehicles)
            return int(row[5]), round(float(dist), 2)
        return None, None

    def ingest(self, limit: Optional[int] = None) -> pl.DataFrame:
        """Ingests UK STATS19 collision and casualty data into a canonical Polars DataFrame."""
        collisions_path = self.data_dir / "dft-road-casualty-statistics-collision-2022.csv"
        casualties_path = self.data_dir / "dft-road-casualty-statistics-casualty-2022.csv"

        if not collisions_path.exists():
            raise FileNotFoundError(f"Collision file not found at {collisions_path}")

        # 1. Aggregate casualties per collision
        if casualties_path.exists():
            cas_df = pl.read_csv(casualties_path, columns=["collision_index", "casualty_severity"], ignore_errors=True)
            cas_agg = cas_df.group_by("collision_index").agg([
                (pl.col("casualty_severity") == 1).sum().alias("fatal_count"),
                (pl.col("casualty_severity") == 2).sum().alias("serious_count"),
                (pl.col("casualty_severity") == 3).sum().alias("slight_count"),
            ])
        else:
            cas_agg = None

        # 2. Read Collisions
        df = pl.read_csv(collisions_path, ignore_errors=True)
        if limit:
            df = df.head(limit)

        if cas_agg is not None:
            df = df.join(cas_agg, on="collision_index", how="left")

        # Load traffic index for exposure join
        self.load_traffic_exposure_index()

        records = []
        for row in df.iter_rows(named=True):
            lat = row.get("latitude")
            lon = row.get("longitude")
            if lat is None or lon is None or np.isnan(lat) or np.isnan(lon):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            # Parse Date & Time
            date_str = row.get("date", "")
            time_str = row.get("time", "12:00") or "12:00"
            try:
                # Format: DD/MM/YYYY
                dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            except Exception:
                dt = datetime(2022, 1, 1, 12, 0)

            # Map Road Class
            # 1: Motorway, 2: A(M), 3: A, 4: B, 5: C, 6: Unclassified
            frc = row.get("first_road_class", 6)
            if frc == 1:
                road_cls = CanonicalRoadClass.MOTORWAY.value
            elif frc == 2:
                road_cls = CanonicalRoadClass.TRUNK_HIGHWAY.value
            elif frc == 3:
                road_cls = CanonicalRoadClass.PRIMARY_ARTERIAL.value
            elif frc == 4:
                road_cls = CanonicalRoadClass.SECONDARY_COLLECTOR.value
            elif frc == 5:
                road_cls = CanonicalRoadClass.TERTIARY_ROAD.value
            else:
                road_cls = CanonicalRoadClass.LOCAL_RESIDENTIAL.value

            # Speed Limit mph -> km/h
            speed_mph = row.get("speed_limit", 30)
            speed_kmh = round(float(speed_mph) * 1.60934, 1) if speed_mph and speed_mph > 0 else 50.0

            # Junction & Control
            j_detail = row.get("junction_detail", 0)
            j_ctrl = row.get("junction_control", 0)
            is_junc = bool(j_detail and j_detail > 0)
            is_sig = (j_ctrl == 4)
            is_round = (j_detail == 1)
            is_stop = (j_ctrl == 2)
            is_give = (j_ctrl == 3)
            is_cross = bool(row.get("pedestrian_crossing", 0) > 0)

            # Environmental
            light_raw = row.get("light_conditions", 1)
            if light_raw == 1:
                light = LightCondition.DAYLIGHT.value
            elif light_raw == 4:
                light = LightCondition.DARKNESS_STREETLIGHTS_ON.value
            elif light_raw == 5:
                light = LightCondition.DARKNESS_STREETLIGHTS_OFF.value
            elif light_raw in (6, 7):
                light = LightCondition.DARKNESS_NO_STREETLIGHTS.value
            else:
                light = LightCondition.UNKNOWN.value

            weather_raw = row.get("weather_conditions", 1)
            if weather_raw == 1:
                weather = WeatherConditionCategory.CLEAR.value
            elif weather_raw in (2, 5):
                weather = WeatherConditionCategory.LIGHT_RAIN.value if weather_raw == 2 else WeatherConditionCategory.HEAVY_RAIN.value
            elif weather_raw in (3, 6):
                weather = WeatherConditionCategory.SNOW_HAIL.value
            elif weather_raw == 4:
                weather = WeatherConditionCategory.HIGH_WINDS_STORM.value
            elif weather_raw == 7:
                weather = WeatherConditionCategory.FOG_SMOKE.value
            elif weather_raw == 8:
                weather = WeatherConditionCategory.CLOUDY_OVERCAST.value
            else:
                weather = WeatherConditionCategory.OTHER.value

            surface_raw = row.get("road_surface_conditions", 1)
            if surface_raw == 1:
                surface = RoadSurfaceCondition.DRY.value
            elif surface_raw == 2:
                surface = RoadSurfaceCondition.WET_DAMP.value
            elif surface_raw == 3:
                surface = RoadSurfaceCondition.SNOW_SLUSH.value
            elif surface_raw == 4:
                surface = RoadSurfaceCondition.FROST_ICE.value
            elif surface_raw == 5:
                surface = RoadSurfaceCondition.FLOOD.value
            else:
                surface = RoadSurfaceCondition.UNKNOWN.value

            # Target: Human Injury Severity (1: Fatal, 2: Serious, 3: Slight)
            col_sev = row.get("collision_severity", 3)
            if col_sev == 1:
                injury_sev = HumanInjurySeverity.FATAL.value
            elif col_sev == 2:
                injury_sev = HumanInjurySeverity.SERIOUS.value
            else:
                injury_sev = HumanInjurySeverity.SLIGHT.value

            # Traffic Exposure Spatial Join
            aadf_flow, dist_km = self.find_nearest_traffic_exposure(lat, lon)

            records.append({
                "global_accident_id": f"GB-STATS19-{row.get('collision_index')}",
                "source_record_id": str(row.get("collision_index")),
                "source_dataset": "UK_DFT_STATS19_2022",
                "country_code": CountryCode.GB.value,
                "licence_type": LicenseType.OGL_V3.value,
                "attribution_text": "Contains Department for Transport data licensed under the Open Government Licence v3.0",
                "latitude": float(lat),
                "longitude": float(lon),
                "coordinate_precision": "HIGH_GPS",
                "administrative_region": str(row.get("local_authority_ons_district", "UK")),
                "locality_or_city": "Great Britain",
                "timestamp_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "local_timestamp": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "timezone_iana": "Europe/London",
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
                "minute": dt.minute,
                "day_of_week": dt.isoweekday(),
                "is_weekend": dt.isoweekday() in (6, 7),
                "is_night": light in (LightCondition.DARKNESS_STREETLIGHTS_ON.value, LightCondition.DARKNESS_NO_STREETLIGHTS.value),
                "road_class": road_cls,
                "speed_limit_kmh": speed_kmh,
                "is_junction": is_junc,
                "is_traffic_signal": is_sig,
                "is_crossing": is_cross,
                "is_roundabout": is_round,
                "is_stop_sign": is_stop,
                "is_give_way": is_give,
                "is_railway": False,
                "is_traffic_calming": False,
                "number_of_lanes": None,
                "divided_roadway": (row.get("road_type") == 3), # Dual carriageway
                "weather_condition": weather,
                "light_condition": light,
                "road_surface_condition": surface,
                "temperature_c": None, # Filled by ERA5 where available
                "humidity_pct": None,
                "pressure_hpa": None,
                "visibility_km": None,
                "wind_speed_kmh": None,
                "precipitation_mm": None,
                "num_vehicles_involved": int(row.get("number_of_vehicles", 1)),
                "num_casualties_total": int(row.get("number_of_casualties", 1)),
                "casualty_fatal_count": int(row.get("fatal_count", 0) or 0),
                "casualty_serious_count": int(row.get("serious_count", 0) or 0),
                "casualty_slight_count": int(row.get("slight_count", 0) or 0),
                "human_injury_severity": injury_sev,
                "traffic_impact_severity": None, # Ground truth not monitored in STATS19
                "annual_average_daily_flow": aadf_flow,
                "distance_to_traffic_counter_km": dist_km
            })

        return pl.DataFrame(records)
