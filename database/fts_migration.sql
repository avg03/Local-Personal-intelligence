-- fts_migration.sql
-- Run once. Adds full-text search over chunks.chunk_text, kept in sync
-- automatically via triggers on the existing chunks table.

-- chunk_id / resource_id marked UNINDEXED: stored for retrieval/filtering,
-- but not tokenized for search — only chunk_text should be full-text matched.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    resource_id UNINDEXED,
    chunk_text
);

-- Backfill: copy whatever's already in chunks into the FTS index.
-- Safe to re-run — DELETE first avoids duplicate rows on a second run.
DELETE FROM chunks_fts;
INSERT INTO chunks_fts (chunk_id, resource_id, chunk_text)
SELECT chunk_id, resource_id, chunk_text FROM chunks;

-- Keep FTS in sync going forward. Note: SQLite's "INSERT OR REPLACE"
-- (what db_writer.save_chunks uses) fires a DELETE then an INSERT under
-- the hood when a chunk_id conflicts — so these two triggers together
-- correctly handle both new chunks AND re-ingested/updated ones.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (chunk_id, resource_id, chunk_text)
    VALUES (new.chunk_id, new.resource_id, new.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
END;

-- Full-text search over resources.name + resources.summary for hybrid
-- resource resolution (semantic Chroma + lexical FTS via RRF).
CREATE VIRTUAL TABLE IF NOT EXISTS resources_fts USING fts5(
    resource_id UNINDEXED,
    name,
    summary
);

DELETE FROM resources_fts;
INSERT INTO resources_fts (resource_id, name, summary)
SELECT resource_id, name, COALESCE(summary, '') FROM resources;

CREATE TRIGGER IF NOT EXISTS resources_ai AFTER INSERT ON resources BEGIN
    INSERT INTO resources_fts (resource_id, name, summary)
    VALUES (new.resource_id, new.name, COALESCE(new.summary, ''));
END;

CREATE TRIGGER IF NOT EXISTS resources_au AFTER UPDATE ON resources BEGIN
    DELETE FROM resources_fts WHERE resource_id = old.resource_id;
    INSERT INTO resources_fts (resource_id, name, summary)
    VALUES (new.resource_id, new.name, COALESCE(new.summary, ''));
END;

CREATE TRIGGER IF NOT EXISTS resources_ad AFTER DELETE ON resources BEGIN
    DELETE FROM resources_fts WHERE resource_id = old.resource_id;
END;