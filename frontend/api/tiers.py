"""Three-tier moderator routing aligned with dissertation §4.6."""

TIER_URGENT = 1
TIER_STANDARD = 2
TIER_MONITOR = 3
TIER_NONE = 0

TIER_LABELS = {
    TIER_URGENT: "Tier 1 — Urgent review",
    TIER_STANDARD: "Tier 2 — Standard review",
    TIER_MONITOR: "Tier 3 — Monitor",
    TIER_NONE: "No review queue",
}


def assign_tier(
    predicted_cyberbullying: bool,
    probability: float,
    peerness: float | None,
    combined_score: float | None,
    is_repeat_offender: int | None,
) -> tuple[int, str]:
    """Route flagged cases using rules from the intervention workflow (§4.6)."""
    if not predicted_cyberbullying:
        if probability >= 0.35:
            return TIER_MONITOR, "Borderline score — log for monitoring"
        return TIER_NONE, "Below classification threshold"

    if (
        combined_score is not None
        and is_repeat_offender is not None
        and combined_score >= 0.45
        and is_repeat_offender == 1
    ):
        return TIER_URGENT, "High combined_score with repeat-offender pattern"

    if peerness is not None and peerness >= 0.5:
        return TIER_STANDARD, "Cyberbullying flagged with elevated peerness"

    if combined_score is not None and combined_score >= 0.2:
        return TIER_STANDARD, "Cyberbullying flagged with moderate combined_score"

    return TIER_MONITOR, "Cyberbullying flagged — borderline relational context"
