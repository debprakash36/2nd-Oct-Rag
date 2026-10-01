/**
 * Typed shapes mirroring the Phase 4 API responses.
 *
 * Duplicated from the FastAPI Pydantic models on purpose. A generated client would
 * couple the frontend build to the backend's OpenAPI output; for six endpoints the
 * duplication is cheaper, and a mismatch shows up as a test failure rather than a
 * type error at the call site.
 */

export interface Turn {
  turn_id: string;
  turn_index: number;
  role: "user" | "assistant";
  content: string;
  citations: string[];
  query_id: string | null;
  abstained: boolean;
  created_at: string;
}

export interface Conversation {
  conversation_id: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
  preview: string;
}

export interface ConversationDetail extends Conversation {
  turns: Turn[];
}

export interface Source {
  index: number;
  chunk_id: string;
  breadcrumb: string | null;
  filename: string;
  page: number | null;
}

export interface Passage {
  chunk_id: string;
  document_id: string;
  filename: string;
  breadcrumb: string | null;
  page: number | null;
  text: string;
}

/** Mirrors `app/api/admin_documents.py::DocumentOut`. */
export interface DocumentRow {
  doc_id: string;
  filename: string;
  mime_type: string;
  state: DocumentState;
  version: number;
  byte_size: number;
  page_count: number | null;
  chunk_count: number;
  content_hash: string | null;
  duplicate_of: string | null;
  error_reason: string | null;
  acl_tags: string[];
  uploaded_at: string;
  indexed_at: string | null;
}

/**
 * Mirrors `DocumentState`. Listed explicitly rather than typed as `string` so a
 * state the backend adds shows up as a type error here instead of silently
 * rendering with no pill styling.
 */
export type DocumentState =
  | "pending"
  | "extracting"
  | "chunking"
  | "embedding"
  | "indexing"
  | "live"
  | "disabled"
  | "failed"
  | "deleted"
  | "duplicate";

/** Mirrors `UploadResult`. */
export interface UploadResult {
  filename: string;
  doc_id: string | null;
  state: DocumentState | null;
  chunks: number;
  duplicate_of: string | null;
  queued: boolean;
  warning: string | null;
  error: string | null;
}

/** Mirrors `UploadResponse`. */
export interface UploadResponse {
  results: UploadResult[];
  accepted: number;
  rejected: number;
}

export type FeedbackValue = "up" | "down" | "none";