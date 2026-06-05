import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import type { Config } from "@netlify/functions";
import { supabaseRest } from "./_shared/supabase-rest.mts";

type LegacySummary = {
  legacy_final_by_rater?: Record<string, string>;
  consensus_dominant_label_final?: string | null;
  consensus_available?: boolean;
  ai_dominant_label?: string | null;
  accuracy_vs_consensus?: number | null;
};

type ExamRow = {
  id: string;
  exam_code: string;
  source_file_name?: string | null;
  patient_code?: string | null;
  duration_seconds?: number | null;
  metadata?: Record<string, any> | null;
};

type AiReviewRow = {
  exam_id: string;
  review_status?: string | null;
  summary?: Record<string, any> | null;
};

type HumanReviewRow = {
  exam_id: string;
  reviewer_id: string;
  review_status?: string | null;
  exported_payload?: Record<string, any> | null;
};

type ProfileRow = {
  id: string;
  full_name?: string | null;
  email?: string | null;
};

type ReviewerSegment = {
  t0: number;
  t1: number;
  label: string;
};

type WindowMetricStats = {
  consensus_window_count: number;
  matched_window_count: number;
  window_agreement: number | null;
  per_label_stats: Record<string, {
    consensus_window_count: number;
    matched_window_count: number;
    window_agreement: number | null;
  }>;
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LEGACY_PATH = resolve(__dirname, "../../viewer/legacy_final_reviews.json");
const VIEWER_PAYLOAD_BUCKET = (Netlify.env.get("SUPABASE_VIEWER_PAYLOAD_BUCKET") || "eeg-viewer-payloads").replace(/^\/+|\/+$/g, "");

async function loadLegacySummaries(): Promise<Record<string, LegacySummary>> {
  try {
    const raw = await readFile(LEGACY_PATH, "utf8");
    return JSON.parse(raw);
  } catch {
    return {};
  }
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

function normalizeStoragePath(path: string) {
  const cleaned = path.replace(/^\/+/, "");
  if (!cleaned || /^https?:\/\//i.test(cleaned)) return cleaned;
  if (!VIEWER_PAYLOAD_BUCKET || cleaned.startsWith(`${VIEWER_PAYLOAD_BUCKET}/`)) return cleaned;
  return `${VIEWER_PAYLOAD_BUCKET}/${cleaned}`;
}

function buildPrivateStorageUrl(path: string) {
  const cleaned = normalizeStoragePath(path);
  if (!cleaned) return null;
  if (/^https?:\/\//i.test(cleaned)) return cleaned;
  const supabaseUrl = getSupabaseUrl();
  if (!supabaseUrl) return null;
  return `${supabaseUrl.replace(/\/+$/, "")}/storage/v1/object/authenticated/${cleaned}`;
}

function buildPublicStorageUrl(path: string) {
  const cleaned = normalizeStoragePath(path);
  if (!cleaned) return null;
  if (/^https?:\/\//i.test(cleaned)) return cleaned;
  const supabaseUrl = getSupabaseUrl();
  if (!supabaseUrl) return null;
  return `${supabaseUrl.replace(/\/+$/, "")}/storage/v1/object/public/${cleaned}`;
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
      return { source: candidate, url: maybeUrl, access: "private" as const };
    }
  }

  const publicCandidates = [
    examMeta.viewer_payload_url,
    examMeta.viewer_payload_public_url,
    examMeta.viewer_payload_path,
    summary.viewer_payload_url,
    summary.viewer_payload_public_url,
    summary.viewer_payload_path,
  ].filter(Boolean) as string[];

  for (const candidate of publicCandidates) {
    const maybeUrl = buildPublicStorageUrl(candidate);
    if (maybeUrl) {
      return { source: candidate, url: maybeUrl, access: "public" as const };
    }
  }

  return null;
}

async function fetchViewerPayload(url: string, access: "public" | "private", forwardedAuthHeader?: string | null) {
  const headers: Record<string, string> = { accept: "application/json" };
  if (access === "private") {
    const serviceRoleKey = getServiceRoleKey();
    if (serviceRoleKey) {
      headers.apikey = serviceRoleKey;
      headers.Authorization = `Bearer ${serviceRoleKey}`;
    } else if (forwardedAuthHeader) {
      const publishableKey = getPublishableKey();
      if (!publishableKey) throw new Error("SUPABASE_PUBLISHABLE_KEY is not configured.");
      headers.apikey = publishableKey;
      headers.Authorization = forwardedAuthHeader;
    } else {
      throw new Error("SUPABASE_SERVICE_ROLE_KEY is not configured and no user session was forwarded.");
    }
  }

  const response = await fetch(url, { headers });
  if (!response.ok) {
    throw new Error(`Viewer payload fetch failed (${response.status})`);
  }
  return response.json();
}

function getReviewerIdentityKey(review: HumanReviewRow, profile?: ProfileRow | null) {
  const normalized = String(profile?.email || "").trim().toLowerCase();
  if (normalized === "elaine@neurogram.com") return "elaine";
  if (normalized === "amandamichelucci@hotmail.com") return "amanda";
  if (normalized === "marinadalio@gmail.com") return "marina";
  return review.reviewer_id;
}

function normalizeLegacyReviewerKey(name: string) {
  return String(name || "").trim().toLowerCase();
}

function getSegmentsWindowLabel(segments: ReviewerSegment[], t0: number, t1: number) {
  let best: { overlap: number; label: string } | null = null;
  segments.forEach(segment => {
    const overlap = Math.min(t1, Number(segment.t1 || 0)) - Math.max(t0, Number(segment.t0 || 0));
    if (overlap <= 0) return;
    if (!best || overlap > best.overlap) {
      best = { overlap, label: segment.label || "normal" };
    }
  });
  return best?.label || "normal";
}

function buildReviewerSources(payload: any, reviews: HumanReviewRow[], profilesById: Map<string, ProfileRow>) {
  const reviewerSources = new Map<string, ReviewerSegment[]>();
  const payloadTracks = payload?.tracks && typeof payload.tracks === "object" ? payload.tracks : {};

  Object.entries(payloadTracks).forEach(([name, segments]) => {
    const normalizedName = String(name).toLowerCase();
    if (normalizedName === "ai" || normalizedName === "consensus") return;
    if (!Array.isArray(segments)) return;
    reviewerSources.set(
      normalizeLegacyReviewerKey(name),
      segments
        .map((segment: any) => ({
          t0: Number(segment?.t0 ?? 0),
          t1: Number(segment?.t1 ?? 0),
          label: String(segment?.label || "normal"),
        }))
        .filter((segment: ReviewerSegment) => Number.isFinite(segment.t0) && Number.isFinite(segment.t1) && segment.t1 > segment.t0),
    );
  });

  reviews
    .filter(review => review.review_status === "completed")
    .forEach(review => {
      const key = getReviewerIdentityKey(review, profilesById.get(review.reviewer_id));
      const annotations = Array.isArray(review.exported_payload?.annotations) ? review.exported_payload?.annotations : [];
      reviewerSources.set(
        key,
        annotations
          .map((annotation: any) => ({
            t0: Number(annotation?.t0 ?? annotation?.t0_seconds ?? 0),
            t1: Number(annotation?.t1 ?? annotation?.t1_seconds ?? 0),
            label: String(annotation?.label || annotation?.primary_label || "normal"),
          }))
          .filter((segment: ReviewerSegment) => Number.isFinite(segment.t0) && Number.isFinite(segment.t1) && segment.t1 > segment.t0),
      );
    });

  return reviewerSources;
}

function computeWindowMetrics(payload: any, reviews: HumanReviewRow[], profilesById: Map<string, ProfileRow>): WindowMetricStats | null {
  const predictions = Array.isArray(payload?.predictions) ? payload.predictions : [];
  if (!predictions.length) return null;

  const reviewerSources = buildReviewerSources(payload, reviews, profilesById);
  const perLabelStats: WindowMetricStats["per_label_stats"] = {};
  let consensusWindowCount = 0;
  let matchedWindowCount = 0;

  predictions.forEach((prediction: any) => {
    const t0 = Number(prediction?.t0 ?? 0);
    const t1 = Number(prediction?.t1 ?? 0);
    const aiLabel = String(prediction?.label || "normal");
    const counts: Record<string, number> = {};

    reviewerSources.forEach(segments => {
      const label = getSegmentsWindowLabel(segments, t0, t1);
      counts[label] = (counts[label] || 0) + 1;
    });

    const [consensusLabel, count] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] || [];
    if (!consensusLabel || Number(count) < 2) return;

    consensusWindowCount += 1;
    if (!perLabelStats[consensusLabel]) {
      perLabelStats[consensusLabel] = {
        consensus_window_count: 0,
        matched_window_count: 0,
        window_agreement: null,
      };
    }
    perLabelStats[consensusLabel].consensus_window_count += 1;

    if (aiLabel === consensusLabel) {
      matchedWindowCount += 1;
      perLabelStats[consensusLabel].matched_window_count += 1;
    }
  });

  Object.values(perLabelStats).forEach(stats => {
    stats.window_agreement = stats.consensus_window_count
      ? Number((stats.matched_window_count / stats.consensus_window_count).toFixed(4))
      : null;
  });

  return {
    consensus_window_count: consensusWindowCount,
    matched_window_count: matchedWindowCount,
    window_agreement: consensusWindowCount ? Number((matchedWindowCount / consensusWindowCount).toFixed(4)) : null,
    per_label_stats: perLabelStats,
  };
}

function buildManifestExam(
  row: ExamRow,
  aiReview: AiReviewRow | undefined,
  legacy: LegacySummary | undefined,
  payload: any | null,
  windowMetrics: WindowMetricStats | null,
) {
  const examId = row.exam_code || row.source_file_name || row.id;
  const aiLabel =
    legacy?.ai_dominant_label ||
    aiReview?.summary?.dominant_label ||
    aiReview?.summary?.ai_dominant_label ||
    null;

  return {
    exam_id: examId,
    patient_id: row.patient_code || null,
    duration_s: Number(row.duration_seconds || payload?.duration || 0),
    dominant_label: aiLabel || "pending",
    ai_dominant_label: aiLabel,
    legacy_final_by_rater: legacy?.legacy_final_by_rater || {},
    consensus_available: Boolean((windowMetrics?.consensus_window_count || 0) > 0),
    consensus_dominant_label_final: null,
    accuracy_vs_consensus: windowMetrics?.window_agreement ?? legacy?.accuracy_vs_consensus ?? null,
    window_metrics: windowMetrics,
  };
}

export default async (request: Request) => {
  const forwardedAuthHeader = request.headers.get("authorization");
  const [legacySummaries, examsRows, aiRows, humanRows, profileRows] = await Promise.all([
    loadLegacySummaries(),
    supabaseRest("exams?select=id,exam_code,source_file_name,patient_code,duration_seconds,metadata&order=uploaded_at.desc", forwardedAuthHeader),
    supabaseRest("exam_ai_reviews?select=exam_id,review_status,summary", forwardedAuthHeader),
    supabaseRest("exam_human_reviews?select=exam_id,reviewer_id,review_status,exported_payload", forwardedAuthHeader),
    supabaseRest("profiles?select=id,full_name,email", forwardedAuthHeader),
  ]);

  const aiByExamId = new Map<string, AiReviewRow>((aiRows as AiReviewRow[]).map(row => [row.exam_id, row]));
  const profilesById = new Map<string, ProfileRow>((profileRows as ProfileRow[]).map(row => [row.id, row]));
  const reviewsByExamId = new Map<string, HumanReviewRow[]>();
  (humanRows as HumanReviewRow[]).forEach(row => {
    const list = reviewsByExamId.get(row.exam_id) || [];
    list.push(row);
    reviewsByExamId.set(row.exam_id, list);
  });

  const exams = await Promise.all((examsRows as ExamRow[]).map(async row => {
    const examId = row.exam_code || row.source_file_name || row.id;
    const aiReview = aiByExamId.get(row.id);
    const payloadLocation = extractPayloadLocation(row, aiReview);
    let payload: any | null = null;

    if (payloadLocation) {
      try {
        payload = await fetchViewerPayload(payloadLocation.url, payloadLocation.access, forwardedAuthHeader);
      } catch {
        payload = null;
      }
    }

    const windowMetrics = payload
      ? computeWindowMetrics(payload, reviewsByExamId.get(row.id) || [], profilesById)
      : null;

    return buildManifestExam(row, aiReview, legacySummaries[examId], payload, windowMetrics);
  }));

  return Response.json({ exams });
};

export const config: Config = {
  path: "/api/exams",
};
