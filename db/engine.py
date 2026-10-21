"""
SQLAlchemy engine and schema shared by the profile store and routing log.

`DATABASE_URL` (any SQLAlchemy URL; Postgres in production) puts both stores
in one database. Without it each store keeps its own SQLite file, as before.
Tables are created on first use; columns added since the first release are
added to an existing database automatically (`_upgrade`).
"""

from __future__ import annotations

from functools import cache

from sqlalchemy import (
    Column,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    inspect,
    text,
)

metadata = MetaData()

profiles = Table(
    "profiles", metadata,
    Column("user_id", String(255), primary_key=True),
    Column("profile_json", Text, nullable=False),
)

routing_events = Table(
    "routing_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", String(40), nullable=False),
    Column("user_id", String(255)),
    Column("query", Text, nullable=False),
    Column("ml_decision", String(16)),        # "small" | "big" | NULL
    Column("ml_confidence", Float),
    Column("llm_decision", String(16)),
    Column("llm_confidence", Float),
    Column("final_routing", String(32), nullable=False),
    Column("model_used", String(128)),
    Column("routed_by", String(16)),          # "ml" | "llm" | "pre-flight" | "slash"
    Column("latency_ms", Float),              # end-to-end, as seen by the user
    Column("input_tokens", Integer),          # summed over every LLM call for the query
    Column("output_tokens", Integer),
    Column("cached_input_tokens", Integer),
    Column("cost_usd", Float),                # estimate; NULL if a model is unpriced
)

llm_calls = Table(                            # one row per API request
    "llm_calls", metadata,
    Column("event_id", Integer, ForeignKey("routing_events.id"), index=True),
    Column("stage", String(32)),              # "classifier" | "answerer" | "reasoner" | ...
    Column("model", String(128)),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("cached_input_tokens", Integer),
    Column("latency_ms", Float),
    Column("cost_usd", Float),
    Column("error", String(64)),
)

# routing_events columns added with usage accounting; a log file from before
# then gets them on first use.
_ADDED_COLUMNS = ("routed_by", "latency_ms", "input_tokens", "output_tokens",
                  "cached_input_tokens", "cost_usd")


def database_url(path_or_url: str) -> str:
    return path_or_url if "://" in path_or_url else f"sqlite:///{path_or_url}"


@cache
def get_engine(url: str) -> Engine:
    engine = create_engine(url, future=True)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):    # concurrent readers + writer
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
    metadata.create_all(engine)
    _upgrade(engine)
    return engine


def _upgrade(engine: Engine) -> None:
    existing = {c["name"] for c in inspect(engine).get_columns("routing_events")}
    with engine.begin() as conn:
        for name in _ADDED_COLUMNS:
            if name not in existing:
                col = routing_events.c[name]
                conn.execute(text(
                    f"ALTER TABLE routing_events ADD COLUMN {name} "
                    f"{col.type.compile(engine.dialect)}"
                ))
