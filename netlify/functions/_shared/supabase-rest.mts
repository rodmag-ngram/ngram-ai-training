const getEnv = (key: string) => Netlify.env.get(key) || "";

export function getSupabaseConfig() {
  const url = getEnv("SUPABASE_URL");
  const serviceRoleKey = getEnv("SUPABASE_SERVICE_ROLE_KEY");

  if (!url || !serviceRoleKey) {
    throw new Error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.");
  }

  return { url, serviceRoleKey };
}

export async function supabaseRest(path: string) {
  const { url, serviceRoleKey } = getSupabaseConfig();
  const res = await fetch(`${url}/rest/v1/${path}`, {
    headers: {
      apikey: serviceRoleKey,
      Authorization: `Bearer ${serviceRoleKey}`,
      Accept: "application/json",
    },
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Supabase REST ${path} failed: ${res.status} ${text}`);
  }

  return res.json();
}
