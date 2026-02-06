"""
SQLite database module for Grade Collection Form system.
Manages collection sessions and student grade submissions.
"""

import sqlite3
import secrets
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'grade_forms.db')


def _get_conn():
    """Get a database connection with WAL mode and foreign keys enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _row_to_dict(row):
    """Convert sqlite3.Row to dict."""
    if row is None:
        return None
    return dict(row)


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS collection_sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                token         TEXT    UNIQUE NOT NULL,
                title         TEXT    NOT NULL,
                description   TEXT    DEFAULT '',
                term_indices  TEXT    NOT NULL,
                show_prediction INTEGER DEFAULT 0,
                model_filename TEXT   DEFAULT '',
                expires_at    TEXT    DEFAULT NULL,
                is_active     INTEGER DEFAULT 1,
                created_at    TEXT    NOT NULL,
                updated_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS form_submissions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    INTEGER NOT NULL,
                student_id    TEXT    NOT NULL,
                student_name  TEXT    DEFAULT '',
                grades_json   TEXT    NOT NULL,
                gpa           REAL    DEFAULT 0.0,
                total_credits REAL    DEFAULT 0.0,
                submitted_at  TEXT    NOT NULL,
                ip_address    TEXT    DEFAULT '',
                FOREIGN KEY (session_id) REFERENCES collection_sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_submissions_session
                ON form_submissions(session_id);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_submissions_unique
                ON form_submissions(session_id, student_id);
        """)
        conn.commit()
        logger.info("Grade form database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing grade form database: {e}")
        raise
    finally:
        conn.close()


def create_session(title, description, term_indices, show_prediction=False,
                   model_filename='', expires_at=None):
    """Create a new collection session and return it."""
    conn = _get_conn()
    try:
        token = secrets.token_hex(8)
        now = datetime.now().isoformat()
        conn.execute(
            """INSERT INTO collection_sessions
               (token, title, description, term_indices, show_prediction,
                model_filename, expires_at, is_active, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (token, title, description, json.dumps(term_indices),
             1 if show_prediction else 0, model_filename or '',
             expires_at, now, now)
        )
        conn.commit()
        session = _row_to_dict(conn.execute(
            "SELECT * FROM collection_sessions WHERE token = ?", (token,)
        ).fetchone())
        logger.info(f"Created grade form session: {title} (token={token})")
        return session
    except sqlite3.IntegrityError:
        # Token collision - retry once
        token = secrets.token_hex(8)
        now = datetime.now().isoformat()
        conn.execute(
            """INSERT INTO collection_sessions
               (token, title, description, term_indices, show_prediction,
                model_filename, expires_at, is_active, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (token, title, description, json.dumps(term_indices),
             1 if show_prediction else 0, model_filename or '',
             expires_at, now, now)
        )
        conn.commit()
        session = _row_to_dict(conn.execute(
            "SELECT * FROM collection_sessions WHERE token = ?", (token,)
        ).fetchone())
        return session
    finally:
        conn.close()


def get_session(token):
    """Get a session by its public token. Returns None if not found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM collection_sessions WHERE token = ?", (token,)
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_session_by_id(session_id):
    """Get a session by its primary key ID."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM collection_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_sessions():
    """List all sessions with submission counts, ordered by newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT s.*,
                   COALESCE(sub.cnt, 0) AS submission_count,
                   COALESCE(sub.avg_gpa, 0) AS avg_gpa
            FROM collection_sessions s
            LEFT JOIN (
                SELECT session_id,
                       COUNT(*) AS cnt,
                       AVG(gpa) AS avg_gpa
                FROM form_submissions
                GROUP BY session_id
            ) sub ON sub.session_id = s.id
            ORDER BY s.created_at DESC
        """).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def update_session(session_id, **kwargs):
    """Update session fields. Allowed fields: title, description, is_active,
    show_prediction, model_filename, expires_at."""
    allowed = {'title', 'description', 'is_active', 'show_prediction',
               'model_filename', 'expires_at'}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False

    updates['updated_at'] = datetime.now().isoformat()
    set_clause = ', '.join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [session_id]

    conn = _get_conn()
    try:
        conn.execute(
            f"UPDATE collection_sessions SET {set_clause} WHERE id = ?",
            values
        )
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def delete_session(session_id):
    """Delete a session and all its submissions (CASCADE)."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM collection_sessions WHERE id = ?", (session_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def submit_grades(session_id, student_id, student_name, grades_json,
                  gpa, total_credits, ip_address=''):
    """Submit or update grades for a student. Uses INSERT OR REPLACE
    to handle duplicate student_id per session."""
    conn = _get_conn()
    try:
        now = datetime.now().isoformat()
        # Check if submission already exists
        existing = conn.execute(
            "SELECT id FROM form_submissions WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        ).fetchone()

        if existing:
            conn.execute(
                """UPDATE form_submissions
                   SET student_name = ?, grades_json = ?, gpa = ?,
                       total_credits = ?, submitted_at = ?, ip_address = ?
                   WHERE session_id = ? AND student_id = ?""",
                (student_name, json.dumps(grades_json) if isinstance(grades_json, dict) else grades_json,
                 gpa, total_credits, now, ip_address, session_id, student_id)
            )
        else:
            conn.execute(
                """INSERT INTO form_submissions
                   (session_id, student_id, student_name, grades_json, gpa,
                    total_credits, submitted_at, ip_address)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, student_id, student_name,
                 json.dumps(grades_json) if isinstance(grades_json, dict) else grades_json,
                 gpa, total_credits, now, ip_address)
            )
        conn.commit()

        row = conn.execute(
            "SELECT * FROM form_submissions WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_submissions(session_id):
    """Get all submissions for a session."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT * FROM form_submissions
               WHERE session_id = ?
               ORDER BY submitted_at DESC""",
            (session_id,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_submission_count(session_id):
    """Get number of submissions for a session."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM form_submissions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        return row['cnt'] if row else 0
    finally:
        conn.close()


def delete_submission(submission_id):
    """Delete a single submission by its ID."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM form_submissions WHERE id = ?", (submission_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def export_submissions_as_long_format(session_id, courses_data):
    """Convert all submissions in a session to long format rows
    suitable for batch prediction.

    Returns list of dicts: [{STUDENT_ID, COURSE_CODE, GRADE, CREDIT}, ...]
    """
    submissions = get_submissions(session_id)
    if not submissions:
        return []

    # Build course credit lookup
    credit_lookup = {}
    for course in courses_data:
        credit_lookup[course['id']] = course.get('credit', 3)

    rows = []
    for sub in submissions:
        grades = json.loads(sub['grades_json']) if isinstance(sub['grades_json'], str) else sub['grades_json']
        for course_id, grade in grades.items():
            if grade and grade.strip():
                rows.append({
                    'STUDENT_ID': sub['student_id'],
                    'COURSE_CODE': course_id,
                    'GRADE': grade.strip().upper(),
                    'CREDIT': credit_lookup.get(course_id, 3)
                })
    return rows
