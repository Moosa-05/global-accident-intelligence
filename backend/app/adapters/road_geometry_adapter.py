from typing import Dict, Any
from backend.app.adapters.base import BaseDataProviderAdapter

class RoadGeometryAdapter(BaseDataProviderAdapter):
    """
    Translates OpenStreetMap (OSM) / HERE / TomTom road feature tags
    into the canonical binary road infrastructure flags.
    """

    def adapt(self, raw_payload: Dict[str, Any]) -> Dict[str, bool]:
        # Handle tags dictionary if OSM style, or flat dictionary
        tags = raw_payload.get("tags", raw_payload)
        
        highway = str(tags.get("highway", "")).lower()
        junction = str(tags.get("junction", "")).lower()
        crossing = str(tags.get("crossing", "")).lower()
        railway = str(tags.get("railway", "")).lower()
        traffic_calming = str(tags.get("traffic_calming", "")).lower()

        return {
            "is_junction": bool(junction or highway in ["motorway_junction", "junction"] or tags.get("is_junction")),
            "is_crossing": bool(crossing or highway in ["crossing", "pedestrian"] or tags.get("is_crossing")),
            "is_traffic_signal": bool(tags.get("traffic_signals") or tags.get("is_traffic_signal") or "traffic_signals" in highway),
            "is_stop_sign": bool(tags.get("stop") or tags.get("is_stop_sign") or highway == "stop"),
            "is_station": bool(tags.get("public_transport") in ["station", "stop_position"] or tags.get("is_station")),
            "is_railway": bool(railway or tags.get("is_railway") or highway == "level_crossing"),
            "is_roundabout": bool(junction == "roundabout" or tags.get("is_roundabout")),
            "is_give_way": bool(tags.get("give_way") or tags.get("is_give_way")),
            "is_traffic_calming": bool(traffic_calming or tags.get("is_traffic_calming") or tags.get("bump")),
            "is_no_exit": bool(tags.get("noexit") == "yes" or tags.get("is_no_exit")),
            "is_amenity": bool(tags.get("amenity") or tags.get("is_amenity"))
        }
