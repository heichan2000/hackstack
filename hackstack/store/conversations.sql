-- Conversation memory: the raw provider messages, tool and search_result blocks included, so the
-- history replays exactly and citation numbering stays aligned across turns.
CREATE TABLE IF NOT EXISTS conversations (
    id         text PRIMARY KEY,
    as_user    text NOT NULL,          -- a conversation belongs to one user: its passages are theirs
    app        text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    conversation_id text NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    seq             integer NOT NULL,  -- position in the history, from 0
    role            text NOT NULL,     -- user | assistant
    content_json    jsonb NOT NULL,    -- a string or a list of content blocks, verbatim
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, seq)
);
