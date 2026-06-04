import type { Config } from "@netlify/functions";

export default async () => {
  const supabaseUrl = Netlify.env.get("SUPABASE_URL") || "";
  const supabasePublishableKey = Netlify.env.get("SUPABASE_PUBLISHABLE_KEY") || "";

  return Response.json({
    supabaseUrl,
    supabasePublishableKey,
  });
};

export const config: Config = {
  path: "/api/runtime-config",
};
