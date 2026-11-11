"""
Profile Database — persistent storage for user profiles.

Each user has one persistent profile that evolves over time.
Profile fields:
  user_id          : unique identifier
  name             : display name
  expertise        : novice | intermediate | expert
  tone             : casual | formal | technical
  domain           : primary domain (e.g. AI, fintech, medicine)
  interests        : JSON list of topics the user has discussed
  topics_discussed : JSON list of all topics ever asked about
  preferred_format : bullets | prose | plain
  interaction_count: total number of sessions
  last_updated     : ISO timestamp of last profile update

Stored through SQLAlchemy: SQLite by default, any DATABASE_URL in production.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, insert, select, update

import config
from db.engine import database_url, get_engine, profiles

# ── Default profile structure ─────────────────────────────────────────────────

DEFAULT_PROFILE = {
    "name":             "",
    "expertise":        "intermediate",
    "tone":             "casual",
    "domain":           "",
    "interests":        [],
    "topics_discussed": [],
    "preferred_format": "prose",
    "interaction_count": 0,
    "last_updated":     "",
}


PERSONAS_PATH = Path(__file__).resolve().parent.parent / "data" / "personas.json"


class ProfileDB:
    """Profile CRUD over SQLAlchemy (SQLite file or DATABASE_URL)."""

    def __init__(self, db_path: str | None = None):
        # Resolved at call time so tests that reload config get their tmp DB
        url = db_path or config.DATABASE_URL or config.DB_PATH
        self.engine = get_engine(database_url(url))

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, user_id: str) -> dict:
        """
        Retrieve a user profile by user_id.
        Returns a default profile if the user doesn't exist yet.
        """
        with self.engine.connect() as conn:
            row = conn.execute(
                select(profiles.c.profile_json).where(profiles.c.user_id == user_id)
            ).first()

        profile = json.loads(row[0]) if row else dict(DEFAULT_PROFILE)
        profile["user_id"] = user_id
        return profile

    def exists(self, user_id: str) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(
                select(profiles.c.user_id).where(profiles.c.user_id == user_id)
            ).first() is not None

    def save(self, user_id: str, profile: dict) -> None:
        """
        Insert or update a user profile.
        Automatically sets last_updated timestamp.
        """
        profile = dict(profile)
        profile.pop("user_id", None)          # don't store user_id inside JSON
        profile["last_updated"] = datetime.now(UTC).isoformat()
        payload = json.dumps(profile)

        with self.engine.begin() as conn:
            updated = conn.execute(
                update(profiles).where(profiles.c.user_id == user_id)
                .values(profile_json=payload)
            ).rowcount
            if not updated:
                conn.execute(insert(profiles).values(user_id=user_id, profile_json=payload))

    def merge_updates(self, user_id: str, updates: dict) -> dict:
        """
        Load existing profile, apply updates intelligently, save and return.

        Merge rules:
          interests / topics_discussed → union (accumulate, no duplicates)
          interaction_count            → increment by 1
          expertise / tone / domain    → overwrite only if new value is non-empty
          preferred_format             → overwrite only if new value is non-empty
        """
        profile = self.get(user_id)

        # Accumulate list fields
        for list_field in ("interests", "topics_discussed"):
            existing = set(profile.get(list_field) or [])
            incoming = set(updates.get(list_field) or [])
            profile[list_field] = sorted(existing | incoming)

        # Overwrite scalar fields only if new value provided
        for scalar in ("expertise", "tone", "domain", "preferred_format", "name"):
            if updates.get(scalar):
                profile[scalar] = updates[scalar]

        # Always increment interaction count
        profile["interaction_count"] = profile.get("interaction_count", 0) + 1

        self.save(user_id, profile)
        return profile

    def delete(self, user_id: str) -> None:
        """Delete a user profile (for testing / reset)."""
        with self.engine.begin() as conn:
            conn.execute(delete(profiles).where(profiles.c.user_id == user_id))

    def list_users(self) -> list[str]:
        """Return all stored user IDs."""
        with self.engine.connect() as conn:
            return [r[0] for r in conn.execute(select(profiles.c.user_id))]

    # ── Personas ──────────────────────────────────────────────────────────────

    @staticmethod
    def personas(path: Path = PERSONAS_PATH) -> list[dict]:
        """The demo personas shipped in data/personas.json."""
        return json.loads(path.read_text(encoding="utf-8"))

    def seed_personas(self, path: Path = PERSONAS_PATH) -> int:
        """Create any persona profile that doesn't exist yet. Returns how many."""
        created = 0
        for persona in self.personas(path):
            if persona.get("profile") is None or self.exists(persona["user_id"]):
                continue
            self.save(persona["user_id"], {**DEFAULT_PROFILE, **persona["profile"]})
            created += 1
        return created
