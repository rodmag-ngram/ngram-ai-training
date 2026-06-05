import type { Config } from "@netlify/functions";
import { supabaseRest } from "./_shared/supabase-rest.mts";

type ExamRow = {
  id: string;
  exam_code: string;
  source_file_name?: string | null;
  metadata?: Record<string, any> | null;
};

type AiReviewRow = {
  exam_id: string;
  summary?: Record<string, any> | null;
};

function jsonHeaders(extra: Record<string, string> = {}) {
  return {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    ...extra,
  };
}

function getSupabaseUrl() {
  return Netlify.env.get("SUPABASE_URL") || "";
}

function getPublishableKey() {
  return Netlify.env.get("SUPABASE_PUBLISHABLE_KEY") || "";
}

function getServiceRoleKey() {
  return Netlify.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";
}

function getViewerPayloadBucket() {
  return (Netlify.env.get("SUPABASE_VIEWER_PAYLOAD_BUCKET") || "eeg-viewer-payloads").replace(/^\/+|\/+$/g, "");
}

function normalizeStoragePath(path: string) {
  const cleaned = path.replace(/^\/+/, "");
  if (!cleaned || /^https?:\/\//i.test(cleaned)) return cleaned;
  const bucket = getViewerPayloadBucket();
  if (!bucket || cleaned.startsWith(`${bucket}/`)) return cleaned;
  return `${bucket}/${cleaned}`;
}

function buildPrivateStorageUrl(path: string) {
  const cleaned = normalizeStoragePath(path);
  if (!cleaned) return null;
  if (/^https?:\/\//i.test(cleaned)) return cleaned;
  const supabaseUrl = getSupabaseUrl();
  if (!supabaseUrl) return null;
  return `${supabaseUrl.replace(/\/+$/, "")}/storage/v1/object/authenticated/${cleaned}`;
}

function extractManifestSource(exam: ExamRow, aiReview?: AiReviewRow | null) {
  const examMeta = exam.metadata || {};
  const summary = aiReview?.summary || {};
  const candidates = [
    examMeta.viewer_payload_storage_path,
    summary.viewer_payload_storage_path,
  ].filter(Boolean) as string[];

  for (const candidate of candidates) {
    const url = buildPrivateStorageUrl(candidate);
    if (url) {
      return {
        manifestPath: candidate,
        manifestUrl: url,
      };
    }
  }

  return null;
}

async function fetchJson(url: string, forwardedAuthHeader?: string | null) {
  const headers: Record<string, string> = {
    accept: "application/json",
  };
  const serviceRoleKey = getServiceRoleKey();
  if (serviceRoleKey) {
    headers.apikey = serviceRoleKey;
    headers.Authorization = `Bearer ${serviceRoleKey}`;
  } else if (forwardedAuthHeader) {
    const publishableKey = getPublishableKey();
    if (!publishableKey) {
      throw new Error("SUPABASE_PUBLISHABLE_KEY is not configured.");
    }
    headers.apikey = publishableKey;
    headers.Authorization = forwardedAuthHeader;
  } else {
    throw new Error("SUPABASE_SERVICE_ROLE_KEY is not configured and no user session was forwarded.");
  }

  const response = await fetch(url, { headers });
  if (!response.ok) {
    throw new Error(`Viewer payload fetch failed (${response.status})`);
  }
  return response.json();
}

export default async (request: Request) => {
  const url = new URL(request.url);
  const examId = url.searchParams.get("id");
  const chunkName = url.searchParams.get("chunk");
  const forwardedAuthHeader = request.headers.get("authorization");

  if (!examId || !chunkName) {
    return new Response(JSON.stringify({
      error: "Missing exam id or chunk name.",
      code: "MISSING_CHUNK_PARAMS",
    }), { status: 400, headers: jsonHeaders() });
  }

  const encodedId = encodeURIComponent(examId);
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  const orFilters = [
    `exam_code.eq.${encodedId}`,
    `source_file_name.eq.${encodedId}`,
  ];
  if (uuidPattern.test(examId)) {
    orFilters.push(`id.eq.${encodedId}`);
  }

  const exams = (await supabaseRest(
    `exams?select=id,exam_code,source_file_name,metadata&or=(${orFilters.join(",")})&limit=1`,
    forwardedAuthHeader,
  )) as ExamRow[];
  const exam = exams[0];

  if (!exam) {
    return new Response(JSON.stringify({
      error: "Exam not found.",
      code: "EXAM_NOT_FOUND",
      exam_id: examId,
    }), { status: 404, headers: jsonHeaders() });
  }

  const aiReviews = (await supabaseRest(
    `exam_ai_reviews?select=exam_id,summary&exam_id=eq.${encodeURIComponent(exam.id)}&limit=1`,
    forwardedAuthHeader,
  )) as AiReviewRow[];
  const manifestSource = extractManifestSource(exam, aiReviews[0] || null);

  if (!manifestSource) {
    return new Response(JSON.stringify({
      error: "Chunked viewer manifest is not available for this exam.",
      code: "VIEWER_PAYLOAD_MISSING",
      exam_id: examId,
    }), { status: 409, headers: jsonHeaders() });
  }

  const manifestDir = manifestSource.manifestPath.replace(/\/manifest\.json$/i, "");
  const chunkPath = `${manifestDir}/chunks/${chunkName.replace(/^\/+/, "")}`;
  const chunkUrl = buildPrivateStorageUrl(chunkPath);

  if (!chunkUrl) {
    return new Response(JSON.stringify({
      error: "Could not derive chunk URL from manifest path.",
      code: "INVALID_CHUNK_SOURCE",
      exam_id: examId,
      manifest_path: manifestSource.manifestPath,
    }), { status: 500, headers: jsonHeaders() });
  }

  try {
    const chunkPayload = await fetchJson(chunkUrl, forwardedAuthHeader);
    return new Response(JSON.stringify(chunkPayload), {
      status: 200,
      headers: jsonHeaders({
        "x-viewer-payload-chunk": chunkName,
      }),
    });
  } catch (error) {
    return new Response(JSON.stringify({
      error: "Chunk exists by reference, but could not be fetched.",
      code: "VIEWER_CHUNK_FETCH_FAILED",
      exam_id: examId,
      chunk: chunkName,
      chunk_path: chunkPath,
      details: error instanceof Error ? error.message : String(error),
    }), { status: 502, headers: jsonHeaders() });
  }
};

export const config: Config = {
  path: "/api/exam-chunk",
};
