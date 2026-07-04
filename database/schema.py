from db import get_connection

conn = get_connection()
cursor = conn.cursor()

# Resources Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS resources (

    resource_id TEXT PRIMARY KEY,

    name TEXT,

    type TEXT,

    hash TEXT UNIQUE,

    path TEXT,

    summary TEXT,

    created_at TEXT,

    modified_at TEXT
)
""")

# Chunks Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (

    chunk_id TEXT PRIMARY KEY,

    resource_id TEXT,

    chunk_text TEXT,

    page_number INTEGER,

    paragraph_number INTEGER,

    embedding_id TEXT,

    FOREIGN KEY(resource_id)
        REFERENCES resources(resource_id)
)
""")

# Concepts
cursor.execute("""
CREATE TABLE IF NOT EXISTS concepts (

    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,

    resource_id TEXT,

    concept TEXT,

    confidence REAL,

    FOREIGN KEY(resource_id)
        REFERENCES resources(resource_id)
)
""")

# Relationships
cursor.execute("""
CREATE TABLE IF NOT EXISTS relationships (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    source_concept TEXT,

    target_concept TEXT,

    relation_type TEXT,

    confidence REAL
)
""")

# User Interaction History
cursor.execute("""
CREATE TABLE IF NOT EXISTS interactions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    query TEXT,

    timestamp TEXT,

    resource_used TEXT
)
""")

conn.commit()
conn.close()

print("Database initialized successfully.")