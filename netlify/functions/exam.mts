import type { Config } from "@netlify/functions";
import { supabaseRest } from "./_shared/supabase-rest.mts";

type ExamRow = {
  id: string;
  exam_code: string;
  source_file_name?: string | null;
  edf_storage_path?: string | null;
  metadata?: Record<string, any> | null;
};

type AiReviewRow = {
  exam_id: string;
  model_name?: string | null;
  model_version?: string | null;
  pipeline_version?: string | null;
  review_status?: string | null;
  summary?: Record<string, any> | null;
  predictions?: unknown;
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

function getServiceRoleKey() {
  return Netlify.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";
}

function buildPublicStorageUrl(path: string) {
  const cleaned = path.replace(/^\/+/, "");
  if (!cleaned) return null;
  if (/^https?:\/\//i.test(cleaned)) return cleaned;
  const supabaseUrl = getSupabaseUrl();
  if (!supabaseUrl) return null;
  return `${supabaseUrl.replace(/\/+$/, "")}/storage/v1/object/public/${cleaned}`;
}

function buildPrivateStorageUrl(path: string) {
  const cleaned = path.replace(/^\/+/, "");
  if (!cleaned) return null;
  if (/^https?:\/\//i.test(cleaned)) return cleaned;
  const supabaseUrl = getSupabaseUrl();
  if (!supabaseUrl) return null;
  return `${supabaseUrl.replace(/\/+$/, "")}/storage/v1/object/authenticated/${cleaned}`;
}

function extractPayloadLocation(exam: ExamRow, aiReview?: AiReviewRow | null) {
  const examMeta = exam.metadata || {};
  const summary = aiReview?.summary || {};

  const privateCandidates = [
    examMeta.viewer_payload_storage_path,
    summary.viewer_payload_storage_path,
  ].filter(Boolean) as string[];

  for (const candidate of privateCandidates) {
    const maybeUrl = buildPrivateStorageUrl(candidate);
    if (maybeUrl) {
      return {
        source: candidate,
        url: maybeUrl,
        access: "private" as const,
      };
    }
  }

  const candidates = [
    examMeta.viewer_payload_url,
    examMeta.viewer_payload_public_url,
    examMeta.viewer_payload_path,
    summary.viewer_payload_url,
    summary.viewer_payload_public_url,
    summary.viewer_payload_path,
  ].filter(Boolean) as string[];

  for (const candidate of candidates) {
    const maybeUrl = buildPublicStorageUrl(candidate);
    if (maybeUrl) {
      return {
        source: candidate,
        url: maybeUrl,
        access: "public" as const,
      };
    }
  }

  return null;
}

async function fetchViewerPayload(url: string, access: "public" | "private") {
  const headers: Record<string, string> = {
    accept: "application/json",
  };

  if (access === "private") {
    const serviceRoleKey = getServiceRoleKey();
    if (!serviceRoleKey) {
      throw new Error("SUPABASE_SERVICE_ROLE_KEY is not configured.");
    }
    headers.apikey = serviceRoleKey;
    headers.Authorization = `Bearer ${serviceRoleKey}`;
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

  if (!examId) {
    return new Response(
      JSON.stringify({
        error: "Missing exam id.",
        code: "MISSING_EXAM_ID",
      }),
      { status: 400, headers: jsonHeaders() },
    );
  }

  const encodedId = encodeURIComponent(examId);
  const exams = (await supabaseRest(
    `exams?select=id,exam_code,source_file_name,edf_storage_path,metadata&or=(exam_code.eq.${encodedId},source_file_name.eq.${encodedId},id.eq.${encodedId})&limit=1`,
  )) as ExamRow[];

  const exam = exams[0];
  if (!exam) {
    return new Response(
      JSON.stringify({
        error: "Exam not found.",
        code: "EXAM_NOT_FOUND",
        exam_id: examId,
      }),
      { status: 404, headers: jsonHeaders() },
    );
  }

  const aiReviews = (await supabaseRest(
    `exam_ai_reviews?select=exam_id,model_name,model_version,pipeline_version,review_status,summary,predictions&exam_id=eq.${encodeURIComponent(exam.id)}&limit=1`,
  )) as AiReviewRow[];
  const aiReview = aiReviews[0] || null;

  const payloadLocation = extractPayloadLocation(exam, aiReview);

  if (payloadLocation) {
    try {
      const payload = await fetchViewerPayload(payloadLocation.url, payloadLocation.access);
      return new Response(JSON.stringify(payload), {
        status: 200,
        headers: jsonHeaders({
          "x-viewer-payload-source": payloadLocation.source,
          "x-viewer-payload-access": payloadLocation.access,
        }),
      });
    } catch (error) {
      return new Response(
        JSON.stringify({
          error: "Viewer payload reference exists, but could not be fetched.",
          code: "VIEWER_PAYLOAD_FETCH_FAILED",
          exam_id: examId,
          payload_source: payloadLocation.source,
          payload_url: payloadLocation.url,
          details: error instanceof Error ? error.message : String(error),
        }),
        { status: 502, headers: jsonHeaders() },
      );
    }
  }

  return new Response(
    JSON.stringify({
      error: "Viewer payload is not available for this exam in production yet.",
      code: "VIEWER_PAYLOAD_MISSING",
      exam_id: examId,
      exam_db_id: exam.id,
      review_status: aiReview?.review_status || "pending",
      current_sources_checked: [
        "exams.metadata.viewer_payload_storage_path",
        "exams.metadata.viewer_payload_url",
        "exams.metadata.viewer_payload_public_url",
        "exams.metadata.viewer_payload_path",
        "exam_ai_reviews.summary.viewer_payload_storage_path",
        "exam_ai_reviews.summary.viewer_payload_url",
        "exam_ai_reviews.summary.viewer_payload_public_url",
        "exam_ai_reviews.summary.viewer_payload_path",
      ],
      next_steps: [
        "Export a production viewer payload for this exam from the local pipeline.",
        "Upload that payload to hosted storage.",
        "Write the payload URL or storage path into exams.metadata or exam_ai_reviews.summary.",
      ],
    }),
    { status: 409, headers: jsonHeaders() },
  );
};

export const config: Config = {
  path: "/api/exam",
};
