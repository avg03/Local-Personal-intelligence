import sqlite3
from pathlib import Path

DB_PATH = Path("database/student_memory.db")

# Ensure database folder exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn