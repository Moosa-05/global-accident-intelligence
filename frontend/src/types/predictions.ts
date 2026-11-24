export interface GeographicMetadata {
  coordinates: {
    latitude: number;
    longitude: number;
  };
  timezone: string;
  is_out_of_distribution: boolean;
  validation_status: 'EMPIRICALLY_VALIDATED' | 'UNVALIDATED_TERRITORY' | string;
  validation_warning: string | null;
  training_geography: string;
  licensing_terms: string;
}

// 1. Human Injury Severity
export interface InjurySeverityRequest {
  latitude: number;
  longitude: number;
  road_class: string;
  speed_limit_kmh?: number;
  divided_roadway?: boolean;
  is_junction?: boolean;
  is_traffic_signal?: boolean;
  is_crossing?: boolean;
  is_roundabout?: boolean;
  units?: 'metric' | 'imperial';
  timezone?: string;
  temperature?: number;
  humidity?: number;
  pressure?: number;
  wind_speed?: number;
  precipitation?: number;
  weather_condition?: string;
  light_condition?: string;
  road_surface_condition?: string;
  is_night?: boolean;
  // UI simulation helpers
  vehicle_type?: string;
  number_of_vehicles?: number;
  number_of_casualties?: number;
  urban_or_rural?: string;
  light_conditions?: string;
  weather_conditions?: string;
  surface_conditions?: string;
}

export interface InjurySeverityResponse {
  prediction_id: string;
  predicted_severity: 'SLIGHT' | 'SERIOUS' | 'FATAL' | string;
  severity_level_code?: number;
  probabilities: {
    SLIGHT: number;
    SERIOUS: number;
    FATAL: number;
    [key: string]: number;
  };
  fatality_risk_alert: boolean;
  latency_ms: number;
  timestamp?: string;
  model_version: string;
  target_definition?: string;
  licensing_notice?: string;
  geographic_metadata?: GeographicMetadata;
}

// 2. Exposure Risk
export interface ExposureRiskRequest {
  road_name?: string;
  road_class: string;
  link_length_km: number;
  speed_limit_kmh: number;
  latitude: number;
  longitude: number;
  annual_average_daily_flow: number;
  // UI alias
  segment_length_km?: number;
}

export interface ExposureRiskResponse {
  prediction_id: string;
  annual_exposure_million_vkt: number;
  predicted_annual_crashes: number;
  predicted_crash_rate_per_million_vkt: number;
  risk_tier: 'LOW_RISK' | 'TYPICAL_RISK' | 'ELEVATED_RISK' | 'HIGH_RISK' | string;
  relative_risk_to_national_baseline?: number;
  statistical_model?: string;
  model_version: string;
  exposure_interpretation_notice?: string;
  latency_ms: number;
  timestamp: string;
}

// 3. Corridor Sensor Risk (Research Benchmark)
export interface CorridorRiskRequest {
  road: string;
  mile: number;
  hour?: number;
  weather?: string;
  light?: string;
  speed_mean: number;
  speed_std?: number;
  speed_min?: number;
  speed_max?: number;
  speed_drop?: number;
  volume_mean: number;
  volume_std?: number;
  occupancy_mean: number;
  occupancy_max?: number;
  occupancy_std?: number;
  speed_upstream?: number;
  speed_downstream?: number;
  // UI aliases
  avg_speed?: number;
  volume?: number;
  surface?: string;
  day_of_week?: string;
}

export interface CorridorRiskResponse {
  prediction_id: string;
  road: string;
  mile: number;
  risk_score: number;
  risk_level: string;
  accident_alert: boolean;
  threshold_used?: number;
  model_version: string;
  latency_ms: number;
  timestamp: string;
}

// 4. Traffic Disruption Severity (CC BY-NC-SA 4.0 Research)
export interface TrafficSeverityRequest {
  latitude: number;
  longitude: number;
  timezone?: string;
  units?: 'metric' | 'imperial';
  temperature?: number;
  humidity?: number;
  pressure?: number;
  visibility?: number;
  wind_speed?: number;
  precipitation?: number;
  weather_condition?: string;
  is_junction?: boolean;
  is_traffic_signal?: boolean;
  is_crossing?: boolean;
  is_station?: boolean;
  is_stop_sign?: boolean;
  is_amenity?: boolean;
  is_give_way?: boolean;
  // Form aliases
  Start_Lat?: number;
  Start_Lng?: number;
  'Temperature(F)'?: number;
  'Humidity(%)'?: number;
  'Pressure(in)'?: number;
  'Visibility(mi)'?: number;
  'Wind_Speed(mph)'?: number;
  'Precipitation(in)'?: number;
  Weather_Condition?: string;
  Crossing?: boolean;
  Junction?: boolean;
  Station?: boolean;
  Stop?: boolean;
  Traffic_Signal?: boolean;
  Sunrise_Sunset?: string;
  month?: number;
  hour?: number;
  day_of_week?: number;
}

export interface TrafficSeverityResponse {
  prediction_id: string;
  predicted_severity: number;
  severity_level: string;
  severity_probabilities: {
    class_1: number;
    class_2: number;
    class_3: number;
    class_4: number;
    [key: string]: number;
  };
  probabilities?: Record<string, number>;
  latency_ms: number;
  timestamp: string;
  model_version: string;
  geographic_metadata?: GeographicMetadata;
}
