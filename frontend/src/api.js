// Small fetch helpers for the customer app and the admin dashboard.

async function handle(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = typeof data.detail === "string" ? data.detail : "Something went wrong, please try again.";
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

export const api = {
  catalog: () => fetch("/api/catalog").then(handle),

  createSession: ({ name, phone, email, selfie }) => {
    const fd = new FormData();
    fd.append("name", name);
    fd.append("phone", phone || "");
    fd.append("email", email || "");
    fd.append("consent", "true");
    fd.append("selfie", selfie, "selfie.jpg");
    return fetch("/api/sessions", { method: "POST", body: fd }).then(handle);
  },

  tryOn: (sessionId, wigId) => {
    const fd = new FormData();
    fd.append("session_id", sessionId);
    fd.append("wig_id", wigId);
    return fetch("/api/tryon", { method: "POST", body: fd }).then(handle);
  },
};

const TOKEN_KEY = "tryon-admin-token";

export const adminAuth = {
  get: () => {
    try { return sessionStorage.getItem(TOKEN_KEY); } catch { return null; }
  },
  set: (t) => {
    try { t ? sessionStorage.setItem(TOKEN_KEY, t) : sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ }
  },
};

export const adminApi = {
  login: (password) =>
    fetch("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    }).then(handle),

  get: (path) =>
    fetch(`/api/admin${path}`, { headers: { Authorization: `Bearer ${adminAuth.get()}` } }).then(handle),

  send: (path, method, body) =>
    fetch(`/api/admin${path}`, {
      method,
      headers: { Authorization: `Bearer ${adminAuth.get()}` },
      body, // FormData (wig photo + fields) or undefined
    }).then(handle),

  del: (path) =>
    fetch(`/api/admin${path}`, { method: "DELETE", headers: { Authorization: `Bearer ${adminAuth.get()}` } }).then(handle),

  downloadCsv: async () => {
    const res = await fetch("/api/admin/export.csv", { headers: { Authorization: `Bearer ${adminAuth.get()}` } });
    if (!res.ok) throw new Error("Export failed");
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = res.headers.get("content-disposition")?.match(/filename="(.+)"/)?.[1] || "customers.csv";
    a.click();
  },
};
