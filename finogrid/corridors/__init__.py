"""Corridor adapter registry — maps ISO country code to adapter instance."""
from .brazil.adapter import BrazilAdapter
from .argentina.adapter import ArgentinaAdapter
from .vietnam.adapter import VietnamAdapter
from .india.adapter import IndiaAdapter
from .uae.adapter import UAEAdapter
from .indonesia.adapter import IndonesiaAdapter
from .philippines.adapter import PhilippinesAdapter
from .nigeria.adapter import NigeriaAdapter
from .base import CorridorAdapter

CORRIDOR_REGISTRY: dict[str, CorridorAdapter] = {
    "BR": BrazilAdapter(),
    "AR": ArgentinaAdapter(),
    "VN": VietnamAdapter(),
    "IN": IndiaAdapter(),
    "AE": UAEAdapter(),
    "ID": IndonesiaAdapter(),
    "PH": PhilippinesAdapter(),
    "NG": NigeriaAdapter(),
}


def get_adapter(corridor_code: str) -> CorridorAdapter:
    adapter = CORRIDOR_REGISTRY.get(corridor_code.upper())
    if not adapter:
        raise ValueError(f"No adapter registered for corridor: {corridor_code}")
    return adapter


__all__ = ["CORRIDOR_REGISTRY", "get_adapter", "CorridorAdapter"]
