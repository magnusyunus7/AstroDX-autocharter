// Thin wrappers around the backend API. All functions throw Error(message)
// with the server's "detail" text on failure.

async function req(method, url, body) {
  const opts = { method };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j.detail) msg = j.detail; } catch {}
    throw new Error(msg);
  }
  return r;
}

export const api = {
  listProjects: async () => (await req("GET", "/api/projects")).json(),
  getProject: async (id) => (await req("GET", `/api/projects/${id}`)).json(),
  createProject: async (formData) => (await req("POST", "/api/projects", formData)).json(),
  patchProject: async (id, fields) => (await req("PATCH", `/api/projects/${id}`, fields)).json(),
  deleteProject: async (id) => (await req("DELETE", `/api/projects/${id}`)).json(),
  putTiming: async (id, timing) => (await req("PUT", `/api/projects/${id}/timing`, timing)).json(),

  trackUrl: (id) => `/api/projects/${id}/track.mp3`,
  jacketUrl: (id) => `/api/projects/${id}/jacket`,
  peaksMeta: async (id) => (await req("GET", `/api/projects/${id}/peaks.json`)).json(),
  peaks: async (id) => new Int8Array(await (await req("GET", `/api/projects/${id}/peaks`)).arrayBuffer()),
  trackBytes: async (id) => (await req("GET", `/api/projects/${id}/track.mp3`)).arrayBuffer(),
};

export function fmtTime(t) {
  if (!isFinite(t)) t = 0;
  const neg = t < 0; t = Math.abs(t);
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${neg ? "-" : ""}${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
}