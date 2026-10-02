"""
app/services/personalization.py - Weight profiles for the safety cost formula.

cost = distance * clamp(1 + alpha*CrimeRisk + beta*DarknessRisk + gamma*IsolationRisk - delta*ActivityScore, min=0.5)

Profiles:
  adult_day   - balanced weights for daytime travel
  woman_night - higher beta (darkness) and gamma (isolation); used for night routes
  child       - highest weights on lighting and activity; heavily penalises isolation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import settings


ProfileName = Literal["adult_day", "woman_night", "child"]


@dataclass(frozen=True)
class ProfileWeights:
    """Safety cost formula weights."""
    alpha: float  # CrimeRisk weight
    beta: float   # DarknessRisk weight
    gamma: float  # IsolationRisk weight
    delta: float  # ActivityScore weight (subtracted – more activity = lower cost)
    label: str

    def __post_init__(self) -> None:
        for name, val in [("alpha", self.alpha), ("beta", self.beta),
                          ("gamma", self.gamma), ("delta", self.delta)]:
            if not 0.0 <= val <= 2.0:
                raise ValueError(f"{name}={val} is out of [0, 2] range")

    def cost_multiplier(
        self,
        crime_risk: float,
        darkness_risk: float,
        isolation_risk: float,
        activity_score: float,
    ) -> float:
        """
        Compute the safety cost multiplier for a segment.
        Clamped to [0.5, inf) so no road ever becomes free.

        Returns a float >= 0.5.
        """
        raw = (
            1.0
            + self.alpha * crime_risk
            + self.beta * darkness_risk
            + self.gamma * isolation_risk
            - self.delta * activity_score
        )
        return max(0.5, raw)

    def segment_risk_score(
        self,
        crime_risk: float,
        darkness_risk: float,
        isolation_risk: float,
        activity_score: float,
    ) -> float:
        """
        Composite risk score in [0, 1] for display / colour coding.
        Lower is safer.
        """
        weights_sum = self.alpha + self.beta + self.gamma
        if weights_sum == 0:
            return 0.0
        raw = (
            self.alpha * crime_risk
            + self.beta * darkness_risk
            + self.gamma * isolation_risk
            - self.delta * activity_score
        ) / weights_sum
        return max(0.0, min(1.0, raw))


# ── Profile registry ──────────────────────────────────────────────────────────

# Using config defaults as the "adult_day" baseline
_PROFILES: dict[str, ProfileWeights] = {
    "adult_day": ProfileWeights(
        alpha=settings.ALPHA_CRIME,          # 0.4 default
        beta=settings.BETA_DARKNESS,          # 0.3 default
        gamma=settings.GAMMA_ISOLATION,       # 0.2 default
        delta=settings.DELTA_ACTIVITY,        # 0.1 default
        label="Adult (daytime)",
    ),
    "woman_night": ProfileWeights(
        # Strengthened weights on darkness and isolation – key risks for women at night
        alpha=0.80,
        beta=1.20,   # darkness matters most at night
        gamma=1.00,  # isolation heavily penalised
        delta=0.25,  # reward busy, active streets
        label="Woman (night)",
    ),
    "child": ProfileWeights(
        # Children need very safe, well-lit, busy routes
        alpha=0.70,
        beta=1.40,   # highest emphasis on lighting
        gamma=1.20,  # highest emphasis on avoiding isolated roads
        delta=0.30,  # reward busy streets
        label="Child",
    ),
}


def get_profile(name: ProfileName | str) -> ProfileWeights:
    """Return the ProfileWeights for a given profile name."""
    if name not in _PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Valid: {list(_PROFILES)}")
    return _PROFILES[name]


def list_profiles() -> dict[str, ProfileWeights]:
    """Return all available profiles."""
    return dict(_PROFILES)
