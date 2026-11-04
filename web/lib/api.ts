// Thin client for the hackstack API (docs/sse.md). The browser calls it directly; the API's
// CORS allows the dev UI origin. Override the base URL with NEXT_PUBLIC_API_URL.
import { readEvents } from "./sse";
import type { ChatEvent, Mode, UsersResponse } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function detail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function getUsers(signal?: AbortSignal): Promise<UsersResponse> {
  const res = await fetch(`${API_URL}/users`, { signal });
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return res.json();
}

export function newConversationId(): string {
  return crypto.randomUUID().replaceAll("-", "");
}

export async function* streamChat(
  body: { conversation_id: string; message: string; as_user: string; mode: Mode },
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw new ApiError(res.status, await detail(res));
  yield* readEvents<ChatEvent>(res.body);
}
