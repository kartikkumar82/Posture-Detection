"""
session_logger.py — SQLite session logging
Records every monitoring session with posture quality stats,
duration, and main issues detected.
"""

import sqlite3
import time
import os
from config import DB_PATH


def init_db() -> sqlite3.Connection:
    """Create database and tables if they don't exist. Returns connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT    NOT NULL,
            start_time   TEXT    NOT NULL,
            duration_sec INTEGER NOT NULL,
            good_pct     REAL    NOT NULL,
            total_frames INTEGER NOT NULL,
            main_issue   TEXT,
            notes        TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS issue_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp  TEXT    NOT NULL,
            issue_key  TEXT    NOT NULL,
            duration_s REAL    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    con.commit()
    return con


def log_session(con: sqlite3.Connection,
                duration_sec: int,
                good_pct: float,
                total_frames: int,
                main_issue: str = None,
                notes: str = None) -> int:
    """
    Record a completed monitoring session.
    Returns the new session ID.
    """
    now = time.localtime()
    cursor = con.execute(
        """INSERT INTO sessions
           (date, start_time, duration_sec, good_pct, total_frames, main_issue, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            time.strftime("%Y-%m-%d", now),
            time.strftime("%H:%M:%S", now),
            duration_sec,
            round(good_pct, 1),
            total_frames,
            main_issue,
            notes,
        ),
    )
    con.commit()
    return cursor.lastrowid


def log_issue_event(con: sqlite3.Connection,
                    session_id: int,
                    issue_key: str,
                    duration_s: float):
    """Record one alert event within a session."""
    con.execute(
        """INSERT INTO issue_events (session_id, timestamp, issue_key, duration_s)
           VALUES (?, ?, ?, ?)""",
        (session_id, time.strftime("%H:%M:%S"), issue_key, round(duration_s, 1)),
    )
    con.commit()


def get_all_sessions(con: sqlite3.Connection):
    """Return all sessions as a list of dicts, newest first."""
    cursor = con.execute(
        "SELECT * FROM sessions ORDER BY id DESC"
    )
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_weekly_summary(con: sqlite3.Connection) -> dict:
    """Return average good_pct and total sessions for the last 7 days."""
    cursor = con.execute(
        """SELECT COUNT(*) as cnt,
                  AVG(good_pct) as avg_good,
                  SUM(duration_sec) as total_sec
           FROM sessions
           WHERE date >= date('now', '-7 days')"""
    )
    row = cursor.fetchone()
    return {
        "sessions":    row[0] or 0,
        "avg_good":    round(row[1] or 0, 1),
        "total_min":   round((row[2] or 0) / 60, 1),
    }
