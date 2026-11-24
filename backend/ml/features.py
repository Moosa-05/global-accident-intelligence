import math
from pathlib import Path
import polars as pl
import numpy as np
from typing import Dict, Any

def extract_features_from_csv(filepath: Path, is_accident: bool) -> Dict[str, Any]:
    """
    Extracts pre-incident features strictly from the first 30 rows (15 minutes at 30s intervals).
    Robustly handles physical bounds, sensor corruption/outliers, and missing neighbor stations.
    Excludes post-incident observations (rows 30+) and target leakage (type, incident_time, data_time).
    """
    df = pl.read_csv(filepath, n_rows=30)
    
    first_row = df[0]
    road = str(first_row['road'][0]) if 'road' in df.columns and first_row['road'][0] is not None else "UNKNOWN"
    try:
        mile = float(first_row['mile'][0]) if 'mile' in df.columns and first_row['mile'][0] is not None else 0.0
    except (ValueError, TypeError):
        mile = 0.0
        
    date_str = str(first_row['date'][0]) if 'date' in df.columns else "2021-01-01"
    try:
        hour = int(first_row['incident_hour'][0]) if 'incident_hour' in df.columns and first_row['incident_hour'][0] is not None else 12
    except (ValueError, TypeError):
        hour = 12
        
    weather = str(first_row['weather'][0]).strip() if 'weather' in df.columns and first_row['weather'][0] is not None else "Clear"
    if weather == "--" or not weather:
        weather = "Clear"
        
    light = str(first_row['light'][0]).strip() if 'light' in df.columns and first_row['light'][0] is not None else "Daylight"
    if light == "--" or not light:
        light = "Daylight"

    # Helper to clean series by physical bounds
    def clean_series(col_name: str, min_val: float, max_val: float) -> np.ndarray:
        if col_name not in df.columns:
            return np.array([], dtype=float)
        vals = df[col_name].cast(pl.Float64, strict=False).to_numpy()
        valid_mask = ~np.isnan(vals) & (vals >= min_val) & (vals <= max_val)
        return vals[valid_mask]

    # Target sensor i (physically plausible limits)
    s_i = clean_series('speed (i)', 0.0, 110.0)
    v_i = clean_series('volume (i)', 0.0, 150.0)
    o_i = clean_series('occupancy (i)', 0.0, 100.0)

    # Upstream sensors (i-1 to i-5)
    s_up_list = [clean_series(f'speed (i-{k})', 0.0, 110.0) for k in range(1, 6)]
    v_up_list = [clean_series(f'volume (i-{k})', 0.0, 150.0) for k in range(1, 6)]
    o_up_list = [clean_series(f'occupancy (i-{k})', 0.0, 100.0) for k in range(1, 6)]

    # Downstream sensors (i+1 to i+5)
    s_dn_list = [clean_series(f'speed (i+{k})', 0.0, 110.0) for k in range(1, 6)]
    v_dn_list = [clean_series(f'volume (i+{k})', 0.0, 150.0) for k in range(1, 6)]
    o_dn_list = [clean_series(f'occupancy (i+{k})', 0.0, 100.0) for k in range(1, 6)]

    # Corridor pooling for robust spatial aggregation
    def pool_sensors(sensor_list):
        pooled = [x for x in sensor_list if len(x) > 0]
        if len(pooled) > 0:
            concat = np.concatenate(pooled)
            return float(np.mean(concat)), float(np.std(concat) if len(concat) > 1 else 0.0)
        return np.nan, np.nan

    s_up_mean, _ = pool_sensors(s_up_list)
    v_up_mean, _ = pool_sensors(v_up_list)
    o_up_mean, _ = pool_sensors(o_up_list)

    s_dn_mean, _ = pool_sensors(s_dn_list)
    v_dn_mean, _ = pool_sensors(v_dn_list)
    o_dn_mean, _ = pool_sensors(o_dn_list)

    # Impute target sensor if missing/offline from nearest corridor non-null readings
    s_i_mean = float(np.mean(s_i)) if len(s_i) > 0 else (s_up_mean if not np.isnan(s_up_mean) else (s_dn_mean if not np.isnan(s_dn_mean) else 60.0))
    s_i_std = float(np.std(s_i)) if len(s_i) > 1 else 0.0
    s_i_min = float(np.min(s_i)) if len(s_i) > 0 else s_i_mean
    s_i_max = float(np.max(s_i)) if len(s_i) > 0 else s_i_mean

    v_i_mean = float(np.mean(v_i)) if len(v_i) > 0 else (v_up_mean if not np.isnan(v_up_mean) else 15.0)
    v_i_std = float(np.std(v_i)) if len(v_i) > 1 else 0.0

    o_i_mean = float(np.mean(o_i)) if len(o_i) > 0 else (o_up_mean if not np.isnan(o_up_mean) else 6.0)
    o_i_max = float(np.max(o_i)) if len(o_i) > 0 else o_i_mean
    o_i_std = float(np.std(o_i)) if len(o_i) > 1 else 0.0

    # Trend in target sensor speed over 15 minutes (last 5 min vs first 5 min)
    if len(s_i) >= 10:
        first_part = s_i[:len(s_i)//3]
        last_part = s_i[-len(s_i)//3:]
        speed_drop = float(np.mean(last_part) - np.mean(first_part))
    else:
        speed_drop = 0.0

    # Fallback for upstream/downstream if whole corridor side is missing
    speed_upstream = s_up_mean if not np.isnan(s_up_mean) else s_i_mean
    volume_upstream = v_up_mean if not np.isnan(v_up_mean) else v_i_mean
    occupancy_upstream = o_up_mean if not np.isnan(o_up_mean) else o_i_mean

    speed_downstream = s_dn_mean if not np.isnan(s_dn_mean) else s_i_mean
    volume_downstream = v_dn_mean if not np.isnan(v_dn_mean) else v_i_mean
    occupancy_downstream = o_dn_mean if not np.isnan(o_dn_mean) else o_i_mean

    speed_gradient = float(speed_downstream - speed_upstream)
    occupancy_gradient = float(occupancy_upstream - occupancy_downstream)
    congestion_index = float(o_i_mean / (s_i_mean + 1.0))

    hour_sin = float(math.sin(2 * math.pi * hour / 24.0))
    hour_cos = float(math.cos(2 * math.pi * hour / 24.0))

    return {
        'event_id': filepath.stem,
        'date': date_str,
        'road': road,
        'mile': mile,
        'hour': hour,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'weather': weather,
        'light': light,
        'speed_mean': s_i_mean,
        'speed_std': s_i_std,
        'speed_min': s_i_min,
        'speed_max': s_i_max,
        'speed_drop': speed_drop,
        'volume_mean': v_i_mean,
        'volume_std': v_i_std,
        'occupancy_mean': o_i_mean,
        'occupancy_max': o_i_max,
        'occupancy_std': o_i_std,
        'speed_upstream': speed_upstream,
        'volume_upstream': volume_upstream,
        'occupancy_upstream': occupancy_upstream,
        'speed_downstream': speed_downstream,
        'volume_downstream': volume_downstream,
        'occupancy_downstream': occupancy_downstream,
        'speed_gradient': speed_gradient,
        'occupancy_gradient': occupancy_gradient,
        'congestion_index': congestion_index,
        'label': 1 if is_accident else 0
    }
