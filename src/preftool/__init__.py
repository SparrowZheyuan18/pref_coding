"""preftool - developer preference capture / extraction / injection toolkit."""

from preftool.models import (
    EVENT_SCHEMA_VERSION,
    PREF_SCHEMA_VERSION,
    Event,
    EvidenceRef,
    ExtractionResult,
    ExtractorConfig,
    InjectionRecord,
    LLMCall,
    Preference,
)

__version__ = "0.1.0"

__all__ = [
    "EVENT_SCHEMA_VERSION",
    "PREF_SCHEMA_VERSION",
    "Event",
    "EvidenceRef",
    "ExtractionResult",
    "ExtractorConfig",
    "InjectionRecord",
    "LLMCall",
    "Preference",
    "__version__",
]
