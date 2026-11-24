from typing import Dict, Any, Optional
from datetime import datetime
from backend.app.adapters.base import BaseDataProviderAdapter
from backend.app.core.units import UnitConverter, UnitSystem

class TelematicsAdapter(BaseDataProviderAdapter):
    """
    Normalizes connected-vehicle / GPS / smartphone telematics streams
    (latitude, longitude, speed, timestamp, heading) into standard coordinates and speeds.
    """

    def adapt(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        lat = float(raw_payload["latitude"])
        lng = float(raw_payload["longitude"])
        speed = float(raw_payload.get("speed", 0.0))
        unit_system = UnitSystem(raw_payload.get("units", "metric").lower())
        
        speed_kmh = UnitConverter.mph_to_kmh(speed) if unit_system == UnitSystem.IMPERIAL else speed
        
        ts_raw = raw_payload.get("timestamp")
        if isinstance(ts_raw, str):
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        elif isinstance(ts_raw, (int, float)):
            ts = datetime.utcfromtimestamp(ts_raw)
        else:
            ts = datetime.utcnow()

        return {
            "latitude": lat,
            "longitude": lng,
            "speed_kmh": speed_kmh,
            "heading": float(raw_payload.get("heading", 0.0)),
            "timestamp": ts.isoformat(),
            "hour": ts.hour,
            "day_of_week": ts.weekday()
        }
