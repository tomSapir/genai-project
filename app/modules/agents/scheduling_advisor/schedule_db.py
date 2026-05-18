"""
schedule_db.py — SQLite implementation of the Schedule table from db_Tech.sql.

Mirrors the SQL Server schema (db_Tech.sql) locally using SQLite + SQLAlchemy.

Schema:
    Schedule(ScheduleID PK, date DATE, time TIME, position VARCHAR(20), available BIT)

Seed logic (matches the SQL script):
    - Year 2024, excluding Saturdays and Mondays
    - Hourly slots 09:00–17:00 (9 slots per day)
    - Positions: 'Python Dev', 'Sql Dev', 'Analyst', 'ML'
    - Availability: pseudo-normal distribution (avg of two uniform draws >= 0.5)

Public API:
    get_nearest_slots(reference_date, position, n) -> list[dict]
    get_engine() -> SQLAlchemy Engine
"""

import random
from datetime import date, time, timedelta
from pathlib import Path

from sqlalchemy import Boolean, Column, Date, Integer, String, Time, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

# DB file lives at genai-project/data/tech.db
_DB_PATH = Path(__file__).parent.parent.parent / "data" / "tech.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_ENGINE = create_engine(f"sqlite:///{_DB_PATH}", echo=False)


class _Base(DeclarativeBase):
    pass


class _Schedule(_Base):
    __tablename__ = "Schedule"

    ScheduleID = Column(Integer, primary_key=True, autoincrement=True)
    date       = Column(Date,        nullable=False)
    time       = Column(Time,        nullable=False)
    position   = Column(String(20),  nullable=False)
    available  = Column(Boolean,     nullable=False)


def _seed(session: Session) -> None:
    """Populate the Schedule table with 2024 slots (matches db_Tech.sql logic)."""
    random.seed(42)

    # weekday() → Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    # Excludes Friday and Saturday → skip {4, 5}
    _SKIP      = {4, 5}
    _TIMES     = [time(h) for h in range(9, 18)]          # 09:00–17:00
    _POSITIONS = ["Python Dev", "Sql Dev", "Analyst", "ML"]

    rows, d = [], date(2024, 1, 1)
    while d <= date(2024, 12, 31):
        if d.weekday() not in _SKIP:
            for t in _TIMES:
                for pos in _POSITIONS:
                    # Pseudo-normal: average two uniform samples, threshold 0.5
                    avail = (random.random() + random.random()) / 2 >= 0.5
                    rows.append(_Schedule(date=d, time=t, position=pos, available=avail))
        d += timedelta(days=1)

    session.add_all(rows)
    session.commit()


def _init_db() -> None:
    """Create the table and seed it on first run; no-op if data already exists."""
    _Base.metadata.create_all(_ENGINE)
    with Session(_ENGINE) as session:
        if session.query(_Schedule).count() == 0:
            _seed(session)


def get_engine():
    """Return the SQLAlchemy engine, initialising the DB if needed."""
    _init_db()
    return _ENGINE


def get_nearest_slots(
    reference_date: date,
    position: str = "Python Dev",
    n: int = 3,
) -> list[dict]:
    """
    Return the n nearest available slots on or after reference_date.

    Args:
        reference_date: earliest date to consider.
        position:       one of 'Python Dev', 'Sql Dev', 'Analyst', 'ML'.
        n:              number of slots to return (default 3).

    Returns:
        List of dicts: [{"date": "YYYY-MM-DD", "time": "HH:MM:SS", "position": str}]
    """
    _init_db()
    with Session(_ENGINE) as session:
        rows = (
            session.query(_Schedule)
            .filter(
                _Schedule.date >= reference_date,
                _Schedule.position == position,
                _Schedule.available == True,
            )
            .order_by(_Schedule.date, _Schedule.time)
            .limit(n)
            .all()
        )
        return [
            {"date": str(r.date), "time": str(r.time), "position": r.position}
            for r in rows
        ]
