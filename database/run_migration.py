"""
run_migration.py
One-off script — applies fts_migration.sql via Python instead of the
sqlite3 CLI, sidesteps PowerShell's lack of `<` redirect support.
"""

from pathlib import Path

from db import get_connection

MIGRATION_SQL = Path(__file__).parent / "fts_migration.sql"

conn = get_connection()
conn.executescript(MIGRATION_SQL.read_text(encoding="utf-8"))
conn.commit()
conn.close()

print("Migration applied.")
