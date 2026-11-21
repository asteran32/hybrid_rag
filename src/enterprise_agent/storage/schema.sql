CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    document_id text PRIMARY KEY CHECK (document_id LIKE 'dsid\_%'),
    source_type text NOT NULL,
    source_path text NOT NULL,
    title text,
    content text NOT NULL,
    metadata jsonb NOT NULL,
    chunk_size integer NOT NULL,
    chunk_overlap integer NOT NULL,
    fingerprint text NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents ON DELETE CASCADE,
    position integer NOT NULL,
    start_offset integer NOT NULL,
    end_offset integer NOT NULL CHECK (end_offset > start_offset),
    content text NOT NULL,
    search_vector tsvector NOT NULL,
    UNIQUE(document_id, position) DEFERRABLE INITIALLY DEFERRED
);
