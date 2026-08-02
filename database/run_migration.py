"""
run_migration.py
One-off script — applies fts_migration.sql via Python instead of the
sqlite3 CLI, sidesteps PowerShell's lack of `<` redirect support.
"""

from db import get_connection

conn = get_connection()
with open("database/fts_migration.sql") as f:
    conn.executescript(f.read())
conn.commit()
conn.close()

print("Migration applied.")