from __future__ import annotations

from enum import Enum


class IdentityKey(str, Enum):
    """How person folders are named when rebuilding the face index."""

    identity_card = "IdentityCard"
    job_number = "JobNumber"
