from datetime import datetime, timezone
import math

ranked = [{"id": 1, "rrf": 1.0, "added_at": "2023-01-01T12:00:00Z"}]

recency_boost = 0.5
now = datetime.now(timezone.utc)

for r in ranked:
    added_at = r.get("added_at")
    if added_at:
        try:
            if isinstance(added_at, str) and added_at.endswith(("Z", "z")):
                added_at = added_at[:-1] + "+00:00"
            created_dt = datetime.fromisoformat(added_at)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            days_ago = max(0.0, (now - created_dt).total_seconds() / 86400)
            decay = math.exp(-0.05 * days_ago)
            r["rrf"] = float(r["rrf"]) * (1.0 + recency_boost * decay)
        except (TypeError, ValueError, AttributeError):
            pass

print(ranked)
