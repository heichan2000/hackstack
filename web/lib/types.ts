// The /chat SSE event vocabulary. Contract: docs/sse.md (server: hackstack/api/sse.py).

export type Hit = {
  number: number | null;
  title: string;
  score: number | null;
  doc_id: string;
  chunk_id: string;
  url: string | null;
};

export type Source = Hit & { cited: boolean };

export type Citation = {
  doc_id: string;
  chunk_id: string;
  title: string;
  url: string | null;
  cited_text: string;
};

export type ChatEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_start"; id: string; name: string; query: string | null; input: unknown }
  | {
      type: "tool_end";
      id: string;
      name: string;
      query: string | null;
      is_error: boolean;
      results: number;
      seconds: number;
      summary: string;
      hits: Hit[];
    }
  | { type: "citation"; number: number | null; citation: Citation }
  | { type: "grounding"; low_grounding: boolean; cited: number[]; sources: Source[] }
  | {
      type: "done";
      conversation_id: string;
      answer: string;
      stop_reason: string;
      truncated: boolean;
      refusal: { category?: string; explanation?: string } | null;
      seconds: number;
      mode: string;
      low_grounding?: boolean;
    }
  | { type: "error"; code: string; message: string };

export type User = { id: string; name: string; role: string; groups: string[] };

export type UsersResponse = { title: string; default_user: string | null; users: User[] };

export type Mode = "agent" | "single_shot";
