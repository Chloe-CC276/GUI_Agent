"""Dual-mode GUI Agent eval: Colab inference server ↔ Windows executor client."""

from .protocol import (
    PROTOCOL_VERSION,
    PlanRequest,
    PlanResponse,
    VerifyRequest,
    VerifyResponse,
)

__all__ = [
    "PROTOCOL_VERSION",
    "PlanRequest",
    "PlanResponse",
    "VerifyRequest",
    "VerifyResponse",
]
