-- Hackstack index schema (Postgres 17 + pgvector 0.8). Applied by PostgresStore.ensure_schema().
-- __DIM__ is replaced with embed.dim; collection_meta records which embedder built the index,
-- and the store refuses to open or write an index built with a different model or dimension.
-- `make reset` drops these tables and recreates them.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS collection_meta (
  id          boolean PRIMARY KEY DEFAULT true CHECK (id),  -- exactly one row
  embed_model text NOT NULL,
  dim         int  NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id         text PRIMARY KEY,
  source         text NOT NULL,
  source_type    text,
  url            text,
  title          text,
  owners         text[] NOT NULL DEFAULT '{}',
  created_at     timestamptz,
  updated_at     timestamptz,
  parent_id      text,
  acl_principals text[] NOT NULL,          -- empty = nobody: fail closed
  content_hash   text,
  extra          jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id       text PRIMARY KEY,         -- f"{doc_id}#{ordinal}": upserts are idempotent
  doc_id         text NOT NULL REFERENCES documents ON DELETE CASCADE,
  ordinal        int  NOT NULL,
  section_path   text[] NOT NULL DEFAULT '{}',
  text           text NOT NULL,
  context        text,                     -- contextual-retrieval blurb, if enrichment ran
  embed_text     text NOT NULL,            -- what is embedded and full-text indexed
  embedding      vector(__DIM__) NOT NULL,
  source         text NOT NULL,
  updated_at     timestamptz,
  acl_principals text[] NOT NULL,          -- copied from the document onto every chunk
  tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED
);

-- HNSW defaults (m=16, ef_construction=64); queries set hnsw.ef_search and iterative scans.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
  USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_acl_gin ON chunks USING gin (acl_principals);
CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_source_updated ON chunks (source, updated_at);

-- Per-connector incremental-sync cursors (owned by the sync manager).
CREATE TABLE IF NOT EXISTS sync_state (
  connector  text PRIMARY KEY,
  cursor     jsonb,
  updated_at timestamptz
);
