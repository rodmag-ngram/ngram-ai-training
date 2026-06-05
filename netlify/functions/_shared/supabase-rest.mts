const getEnv = (key: string) => Netlify.env.get(key) || "";

export function getSupabaseConfig() {
  const url = getEnv("SUPABASE_URL");
  const serviceRoleKey = getEnv("SUPABASE_SERVICE_ROLE_KEY");
  const publishableKey = getEnv("SUPABASE_PUBLISHABLE_KEY");

  if (!url) {
    throw new Error("Missing SUPABASE_URL.");
  }

  return { url, serviceRoleKey, publishableKey };
}

export async function supabaseRest(path: string, authHeader?: string | null) {
  const { url, serviceRoleKey, publishableKey } = getSupabaseConfig();
  const headers: Record<string, string> = {
    Accept: "application/json",
  };

  if (serviceRoleKey) {
    headers.apikey = serviceRoleKey;
    headers.Authorization = `Bearer ${serviceRoleKey}`;
  } else if (authHeader && publishableKey) {
    headers.apikey = publishableKey;
    headers.Authorization = authHeader;
  } else {
    throw new Error("Missing SUPABASE_SERVICE_ROLE_KEY, or SUPABASE_PUBLISHABLE_KEY with forwarded user session.");
  }

  const res = await fetch(`${url}/rest/v1/${path}`, {
    headers,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Supabase REST ${path} failed: ${res.status} ${text}`);
  }

  return res.json();
}
