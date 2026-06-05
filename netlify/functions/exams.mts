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

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LEGACY_PATH = resolve(__dirname, "../../viewer/legacy_final_reviews.json");

async function loadLegacySummaries(): Promise<Record<string, LegacySummary>> {
  try {
    const raw = await readFile(LEGACY_PATH, "utf8");
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

function buildManifestExam(row: ExamRow, aiReview: AiReviewRow | undefined, legacy: LegacySummary | undefined) {
  const examId = row.exam_code || row.source_file_name || row.id;
  const aiLabel =
    legacy?.ai_dominant_label ||
    aiReview?.summary?.dominant_label ||
    aiReview?.summary?.ai_dominant_label ||
    null;
  const consensusLabel = legacy?.consensus_dominant_label_final || null;
  const consensusAvailable = Boolean(legacy?.consensus_available);
  const dominantLabel = consensusLabel || aiLabel || "pending";

  return {
    exam_id: examId,
    patient_id: row.patient_code || null,
    duration_s: Number(row.duration_seconds || 0),
    dominant_label: dominantLabel,
    ai_dominant_label: aiLabel,
    legacy_final_by_rater: legacy?.legacy_final_by_rater || {},
    consensus_available: consensusAvailable,
    consensus_dominant_label_final: consensusLabel,
    accuracy_vs_consensus:
      consensusAvailable && legacy?.accuracy_vs_consensus !== undefined && legacy?.accuracy_vs_consensus !== null
        ? Number(legacy.accuracy_vs_consensus)
        : null,
  };
}

export default async (request: Request) => {
  const forwardedAuthHeader = request.headers.get("authorization");
  const [legacySummaries, examsRows, aiRows] = await Promise.all([
    loadLegacySummaries(),
    supabaseRest("exams?select=id,exam_code,source_file_name,patient_code,duration_seconds,metadata&order=uploaded_at.desc", forwardedAuthHeader),
    supabaseRest("exam_ai_reviews?select=exam_id,review_status,summary", forwardedAuthHeader),
  ]);

  const aiByExamId = new Map<string, AiReviewRow>((aiRows as AiReviewRow[]).map(row => [row.exam_id, row]));
  const exams = (examsRows as ExamRow[]).map(row => {
    const examId = row.exam_code || row.source_file_name || row.id;
    return buildManifestExam(row, aiByExamId.get(row.id), legacySummaries[examId]);
  });

  return Response.json({ exams });
};

export const config: Config = {
  path: "/api/exams",
};
