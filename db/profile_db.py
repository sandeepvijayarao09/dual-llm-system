"""
Profile Database — SQLite-backed storage for user profiles.

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

No external dependencies — uses Python's built-in sqlite3.
"""

import json
import sqlite3
from datetime import datetime, timezone
from config import DB_PATH


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


class ProfileDB:
    """Thin SQLite wrapper for user profile CRUD operations."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, user_id: str) -> dict:
        """
        Retrieve a user profile by user_id.
        Returns a default profile if the user doesn't exist yet.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM profiles WHERE user_id = ?",
                (user_id,)
            ).fetchone()

        if row:
            profile = json.loads(row[0])
            profile["user_id"] = user_id
            return profile

        # First-time user — return default profile
        profile = dict(DEFAULT_PROFILE)
        profile["user_id"] = user_id
        return profile

    def save(self, user_id: str, profile: dict) -> None:
        """
        Insert or update a user profile.
        Automatically sets last_updated timestamp.
        """
        profile = dict(profile)
        profile.pop("user_id", None)          # don't store user_id inside JSON
        profile["last_updated"] = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO profiles (user_id, profile_json)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json
                """,
                (user_id, json.dumps(profile))
            )

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
        with self._connect() as conn:
            conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))

    def list_users(self) -> list[str]:
        """Return all stored user IDs."""
        with self._connect() as conn:
            rows = conn.execute("SELECT user_id FROM profiles").fetchall()
        return [r[0] for r in rows]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the profiles table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id      TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
