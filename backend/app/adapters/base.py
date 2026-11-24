from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseDataProviderAdapter(ABC):
    """
    Abstract interface for external data provider adapters.
    Ensures that live telematics, weather feeds, and map geometries from any country
    can be adapted into the SaaS platform's canonical feature schemas.
    """

    @abstractmethod
    def adapt(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Translates third-party provider payload into canonical schema."""
        pass
