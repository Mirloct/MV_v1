"""Block operational publication after a critical latent-health verdict.

Data sources / inputs: VAE latent-health diagnostic verdict.
Created: 2026-11-25
Last modified: 2026-11-25
Changelog: 2026-11-25 — separate a computed diagnostic from release acceptance.
"""


def require_acceptable_health(verdict: dict) -> None:
    """Fail before official exports when measured latent collapse is critical."""
    if verdict.get("collapsed") and verdict.get("severity") == "critical":
        raise RuntimeError(
            "Phase 7 model acceptance failed: critical VAE latent-collapse check. "
            "Official OOT publication is blocked. Inspect latent diagnostics and "
            "retrain with validated beta/latent capacity before publishing."
        )
