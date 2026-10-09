import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  LogOut, RefreshCw, Download, Search, Users, Sparkles, CalendarDays, Gauge, Phone, ChevronRight, X,
  Trash2, Smartphone, Clock, ShieldCheck, AlertTriangle, Mail, Globe,
} from "lucide-react";
import { adminApi, adminAuth } from "../api.js";
import WigManager from "./WigManager.jsx";
import { Button, IconButton, fmtDateTime, timeAgo } from "../ui/ui.jsx";

export default function AdminApp() {
  const [token, setToken] = useState(adminAuth.get());
  useEffect(() => { document.title = "Be Beautiful Hair · Try-On Admin"; }, []);
  const logout = () => { adminAuth.set(null); setToken(null); };
  return token ? <Dashboard onLogout={logout} /> : <Login onLogin={(t) => { adminAuth.set(t); setToken(t); }} />;
}

/* ---------------- login ---------------- */

function Login({ onLogin }) {
  const [pw, setPw] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const { token } = await adminApi.login(pw);
      onLogin(token);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-full place-items-center bg-gradient-to-b from-blush to-cream px-6">
      <form onSubmit={submit} className="w-full max-w-sm rounded-[28px] bg-white p-7 shadow-lift">
        <img src="/logo.png" alt="Be Beautiful Hair" className="size-16 rounded-full shadow-soft" />
        <h1 className="mt-5 font-display text-3xl font-semibold tracking-tight">Try-On Admin</h1>
        <p className="mt-1 text-sm text-cocoa">See who tried which wig, and when.</p>
        <input
          type="password"
          autoFocus
          value={pw}
          onChange={(e) => setPw(e.target.value)}
          placeholder="Password"
          className="mt-6 h-13 w-full rounded-2xl bg-cream px-4 text-[16px] outline-none ring-1 ring-espresso/10 focus:ring-2 focus:ring-brand"
        />
        {error && <p className="mt-2 text-sm text-red-700">{error}</p>}
        <Button type="submit" variant="dark" loading={busy} disabled={!pw} className="mt-4 w-full h-13">Sign in</Button>
      </form>
    </div>
  );
}

/* ---------------- dashboard ---------------- */

const PAGE = 30;

function Dashboard({ onLogout }) {
  const [stats, setStats] = useState(null);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(true);
  const [openId, setOpenId] = useState(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("activity");
  const [health, setHealth] = useState(null);
  const qRef = useRef(q);
  qRef.current = q;

  const guard = useCallback((e) => {
    if (e.status === 401) onLogout();
    else setError(e.message);
  }, [onLogout]);

  const loadList = useCallback(async (offset = 0) => {
    setLoading(true);
    try {
      const r = await adminApi.get(`/sessions?limit=${PAGE}&offset=${offset}&q=${encodeURIComponent(qRef.current)}`);
      setTotal(r.total);
      setItems((prev) => (offset ? [...prev, ...r.items] : r.items));
    } catch (e) { guard(e); } finally { setLoading(false); }
  }, [guard]);

  const refresh = useCallback(() => {
    setError("");
    adminApi.get("/stats").then(setStats).catch(guard);
    loadList(0);
  }, [guard, loadList]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    const check = () => adminApi.get("/health").then(setHealth).catch(() => {});
    check();
    const t = setInterval(check, 30000);
    return () => clearInterval(t);
  }, []);
  const firstSearch = useRef(true);
  useEffect(() => {
    if (firstSearch.current) { firstSearch.current = false; return; } // refresh() already loaded page 1
    const t = setTimeout(() => loadList(0), 300);
    return () => clearTimeout(t);
  }, [q, loadList]);

  return (
    <div className="min-h-full bg-cream">
      <header className="sticky top-0 z-20 border-b border-espresso/5 bg-cream/85 backdrop-blur-md safe-top">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 pb-3 sm:px-6">
          <img src="/logo.png" alt="" className="size-10 rounded-full" />
          <div className="flex-1">
            <p className="text-xs font-semibold uppercase tracking-wider text-brand">Be Beautiful Hair</p>
            <h1 className="font-display text-xl sm:text-2xl leading-tight tracking-tight">Try-on activity</h1>
          </div>
          <IconButton label="Refresh" onClick={refresh} className="bg-white ring-1 ring-espresso/10">
            <RefreshCw className={`size-[18px] ${loading ? "animate-spin" : ""}`} />
          </IconButton>
          <span className="hidden sm:block">
            <Button variant="soft" className="h-11" onClick={() => adminApi.downloadCsv().catch(guard)}>
              <Download className="size-4" /> Export CSV
            </Button>
          </span>
          <span className="sm:hidden">
            <IconButton label="Export CSV" onClick={() => adminApi.downloadCsv().catch(guard)} className="bg-white ring-1 ring-espresso/10">
              <Download className="size-[18px]" />
            </IconButton>
          </span>
          <IconButton label="Sign out" onClick={onLogout} className="bg-espresso text-cream">
            <LogOut className="size-[18px]" />
          </IconButton>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 pb-16 pt-5 sm:px-6">
        {error && <p className="mb-4 rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-800 ring-1 ring-red-200">{error}</p>}
        <HealthBanner health={health} />

        <div className="mb-5 inline-flex rounded-full bg-white p-1 ring-1 ring-espresso/10">
          {[["activity", "Activity"], ["wigs", "Wigs"]].map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`rounded-full px-5 py-2 text-sm font-semibold transition ${tab === k ? "bg-brand text-white" : "text-cocoa"}`}>
              {label}
            </button>
          ))}
        </div>

        {tab === "wigs" ? <WigManager onError={guard} /> : (
        <>

        {/* Stat cards */}
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat icon={Users} label="Customers" value={stats?.people} sub={stats && `${stats.peopleToday} today`} />
          <Stat icon={Sparkles} label="Try-ons" value={stats?.tryons} sub={stats && `${stats.tryonsToday} today`} accent />
          <Stat icon={Gauge} label="Success rate" value={stats?.successRate != null ? `${stats.successRate}%` : "–"} sub="AI results delivered" />
          <Stat icon={Clock} label="Avg. wait" value={stats?.avgSeconds != null ? `${stats.avgSeconds}s` : "–"} sub="per try-on" />
        </div>

        <div className="mt-3 grid gap-3 lg:grid-cols-5">
          <Panel title="Last 14 days" icon={CalendarDays} className="lg:col-span-3">
            <DailyBars data={stats?.daily || []} />
          </Panel>
          <Panel title="Most tried wigs" icon={Sparkles} className="lg:col-span-2">
            <TopWigs data={stats?.topWigs || []} />
          </Panel>
        </div>

        {/* Customers */}
        <section className="mt-6">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="font-display text-2xl tracking-tight">Customers</h2>
              <p className="text-sm text-cocoa">{total} {total === 1 ? "person" : "people"} took a photo</p>
            </div>
            <label className="relative w-full sm:w-72">
              <Search className="absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-cocoa/60" />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search name or phone"
                className="h-12 w-full rounded-full bg-white pl-10 pr-4 text-[15px] outline-none ring-1 ring-espresso/10 focus:ring-2 focus:ring-brand"
              />
            </label>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {items.map((c) => <CustomerCard key={c.id} c={c} onOpen={() => setOpenId(c.id)} />)}
            {loading && !items.length && Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="relative h-[104px] overflow-hidden rounded-3xl bg-white/70 shimmer" />
            ))}
          </div>
          {!loading && !items.length && (
            <div className="mt-6 rounded-3xl bg-white/70 p-10 text-center text-cocoa ring-1 ring-espresso/5">
              {q ? "No customers match that search." : "No one has taken a photo yet. Share the try-on link to get started."}
            </div>
          )}
          {items.length < total && (
            <div className="mt-5 text-center">
              <Button variant="soft" loading={loading} onClick={() => loadList(items.length)}>Load more</Button>
            </div>
          )}
        </section>
        </>
        )}
      </main>

      <AnimatePresence>
        {openId && (
          <CustomerDetail
            id={openId}
            onClose={() => setOpenId(null)}
            onDeleted={() => { setOpenId(null); refresh(); }}
            onError={guard}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

function Stat({ icon: Icon, label, value, sub, accent }) {
  return (
    <div className={`rounded-3xl p-4 sm:p-5 ring-1 ring-espresso/5 ${accent ? "bg-espresso text-cream" : "bg-white"}`}>
      <div className="flex items-center gap-2">
        <Icon className={`size-4 ${accent ? "text-gold" : "text-brand"}`} />
        <span className={`text-[13px] font-semibold ${accent ? "text-cream/80" : "text-cocoa"}`}>{label}</span>
      </div>
      <p className="mt-2 font-display text-[34px] leading-none tracking-tight">{value ?? "–"}</p>
      <p className={`mt-1.5 text-xs ${accent ? "text-cream/60" : "text-cocoa/70"}`}>{sub || " "}</p>
    </div>
  );
}

function Panel({ title, icon: Icon, className = "", children }) {
  return (
    <div className={`rounded-3xl bg-white p-4 sm:p-5 ring-1 ring-espresso/5 ${className}`}>
      <div className="mb-4 flex items-center gap-2">
        <Icon className="size-4 text-brand" />
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      {children}
    </div>
  );
}

function DailyBars({ data }) {
  const max = Math.max(1, ...data.map((d) => d.count));
  return (
    <div className="flex h-40 items-end gap-1.5">
      {data.map((d, i) => {
        const date = new Date(d.date + "T00:00:00");
        const today = i === data.length - 1;
        return (
          <div key={d.date} className="group flex h-full flex-1 flex-col items-center justify-end gap-1.5">
            <span className="text-[10px] font-semibold text-cocoa opacity-0 group-hover:opacity-100 transition">{d.count}</span>
            <div
              className={`w-full rounded-t-lg rounded-b-sm transition-all ${today ? "bg-brand" : "bg-blush group-hover:bg-brand/60"}`}
              style={{ height: `${Math.max(3, (d.count / max) * 100)}%` }}
              title={`${date.toDateString()}: ${d.count} try-ons`}
            />
            <span className={`text-[10px] ${today ? "font-bold text-brand" : "text-cocoa/70"}`}>
              {date.toLocaleDateString(undefined, { weekday: "narrow" })}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function TopWigs({ data }) {
  if (!data.length) return <p className="py-10 text-center text-sm text-cocoa">No try-ons yet</p>;
  const max = data[0].count;
  return (
    <ul className="space-y-3">
      {data.slice(0, 6).map((w) => (
        <li key={w.wig_id}>
          <div className="flex justify-between text-[13px]">
            <span className="truncate pr-2 font-medium">{w.wig_name}</span>
            <span className="font-semibold text-cocoa">{w.count}</span>
          </div>
          <div className="mt-1 h-2 overflow-hidden rounded-full bg-sand">
            <div className="h-full rounded-full bg-gradient-to-r from-brand to-gold" style={{ width: `${(w.count / max) * 100}%` }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

function CustomerCard({ c, onOpen }) {
  return (
    <button
      onClick={onOpen}
      className="flex w-full items-center gap-3.5 rounded-3xl bg-white p-3.5 text-left ring-1 ring-espresso/5 transition hover:shadow-soft active:scale-[.99]"
    >
      <img src={c.selfie} alt="" loading="lazy" className="size-[72px] shrink-0 rounded-2xl object-cover bg-sand" />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <p className="truncate font-semibold">{c.name}</p>
          <span className="shrink-0 text-[11px] text-cocoa/70">{timeAgo(c.createdAt)}</span>
        </div>
        {c.phone && <p className="text-[13px] text-cocoa">{c.phone}</p>}
        <div className="mt-1.5 flex flex-wrap gap-1">
          {c.wigs.length ? c.wigs.slice(0, 2).map((w) => (
            <span key={w} className="max-w-[140px] truncate rounded-full bg-cream px-2 py-0.5 text-[11px] font-medium text-cocoa">{w}</span>
          )) : <span className="text-[11px] text-cocoa/60">Photo only, no try-on yet</span>}
          {c.wigs.length > 2 && <span className="rounded-full bg-cream px-2 py-0.5 text-[11px] font-medium text-cocoa">+{c.wigs.length - 2}</span>}
        </div>
      </div>
      <div className="flex flex-col items-end gap-1">
        <span className="rounded-full bg-brand/10 px-2 py-0.5 text-xs font-bold text-brand">{c.ok}</span>
        <ChevronRight className="size-4 text-cocoa/40" />
      </div>
    </button>
  );
}

/* ---------------- detail sheet ---------------- */

function CustomerDetail({ id, onClose, onDeleted, onError }) {
  const [d, setD] = useState(null);
  const [showBefore, setShowBefore] = useState({});
  const [zoom, setZoom] = useState(null);
  const [confirmDel, setConfirmDel] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    adminApi.get(`/sessions/${id}`).then(setD).catch(onError);
  }, [id, onError]);

  async function remove() {
    setDeleting(true);
    try { await adminApi.del(`/sessions/${id}`); onDeleted(); } catch (e) { onError(e); setDeleting(false); }
  }

  return (
    <motion.div className="fixed inset-0 z-40 flex items-end justify-center sm:items-center sm:p-6"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <div className="absolute inset-0 bg-espresso/50 backdrop-blur-[2px]" onClick={onClose} />
      <motion.div
        className="relative flex max-h-[92dvh] w-full max-w-3xl flex-col overflow-hidden rounded-t-[32px] bg-cream sm:rounded-[32px] shadow-lift"
        initial={{ y: 60, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 60, opacity: 0 }}
        transition={{ type: "spring", stiffness: 260, damping: 28 }}
      >
        <div className="mx-auto mt-2.5 h-1.5 w-10 rounded-full bg-espresso/15 sm:hidden" />
        <div className="flex items-center justify-between px-5 pt-3 pb-2">
          <h2 className="font-display text-2xl tracking-tight">{d?.name || "Loading…"}</h2>
          <IconButton label="Close" onClick={onClose} className="bg-white ring-1 ring-espresso/10"><X className="size-5" /></IconButton>
        </div>

        {d && (
          <div className="overflow-y-auto px-5 pb-6 safe-bottom">
            <div className="grid gap-4 sm:grid-cols-[180px_1fr]">
              <button onClick={() => setZoom(d.selfie)} className="overflow-hidden rounded-3xl bg-sand">
                <img src={d.selfie} alt="Photo taken" className="aspect-[3/4] w-full object-cover" />
              </button>
              <div className="space-y-2.5 rounded-3xl bg-white p-4 text-sm ring-1 ring-espresso/5">
                <p className="text-xs font-semibold uppercase tracking-wider text-brand">Photo taken by</p>
                <Info icon={Users} label="Name" value={d.name} />
                {d.phone && <Info icon={Phone} label="Phone" value={<a className="font-semibold text-brand underline-offset-2 hover:underline" href={`tel:${d.phone}`}>{d.phone}</a>} />}
                {d.email && <Info icon={Mail} label="Email" value={d.email} />}
                <Info icon={Clock} label="When" value={fmtDateTime(d.createdAt)} />
                <Info icon={Smartphone} label="Device" value={d.device} />
                {d.ip && <Info icon={Globe} label="IP" value={d.ip} />}
                <Info icon={ShieldCheck} label="Consent" value={d.consent ? "Given" : "Not given"} />
              </div>
            </div>

            <h3 className="mt-6 mb-3 font-semibold">Wigs tried ({d.tryons.length})</h3>
            {!d.tryons.length && <p className="rounded-3xl bg-white p-6 text-center text-sm text-cocoa ring-1 ring-espresso/5">This customer took a photo but hasn't tried a wig yet.</p>}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {d.tryons.map((t) => (
                <div key={t.id} className="overflow-hidden rounded-3xl bg-white ring-1 ring-espresso/5">
                  {t.status === "ok" ? (
                    <button
                      onClick={() => setShowBefore((s) => ({ ...s, [t.id]: !s[t.id] }))}
                      onDoubleClick={() => setZoom(showBefore[t.id] ? t.before : t.after)}
                      className="relative block w-full"
                    >
                      <img src={showBefore[t.id] ? t.before : t.after} alt={t.wigName} loading="lazy" className="aspect-[2/3] w-full object-cover bg-sand" />
                      <span className="absolute top-2 left-2 rounded-full bg-black/50 px-2 py-0.5 text-[10px] font-semibold uppercase text-white">
                        {showBefore[t.id] ? "Before" : "After"} · tap
                      </span>
                    </button>
                  ) : (
                    <div className="grid aspect-[2/3] place-items-center bg-red-50 p-4 text-center text-xs text-red-800">
                      <div><AlertTriangle className="mx-auto mb-2 size-5" />{t.error || "Failed"}</div>
                    </div>
                  )}
                  <div className="p-3">
                    <p className="truncate text-[13px] font-semibold">{t.wigName}</p>
                    <p className="text-[11px] text-cocoa">{timeAgo(t.createdAt)}{t.seconds ? ` · ${t.seconds}s` : ""}{t.status === "ok" && !t.faceLocked ? " · AI face" : ""}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-8 border-t border-espresso/10 pt-4">
              {!confirmDel ? (
                <Button variant="ghost" className="text-red-700" onClick={() => setConfirmDel(true)}>
                  <Trash2 className="size-4" /> Delete customer and photos
                </Button>
              ) : (
                <div className="flex flex-wrap items-center gap-3 rounded-2xl bg-red-50 p-3 ring-1 ring-red-200">
                  <p className="flex-1 text-sm text-red-900">This permanently deletes {d.name}'s photos and history.</p>
                  <Button variant="soft" onClick={() => setConfirmDel(false)}>Cancel</Button>
                  <Button className="bg-red-700 active:bg-red-800" loading={deleting} onClick={remove}>Delete</Button>
                </div>
              )}
            </div>
          </div>
        )}
      </motion.div>

      {zoom && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/90 p-4" onClick={() => setZoom(null)}>
          <img src={zoom} alt="" className="max-h-full max-w-full rounded-2xl object-contain" />
        </div>
      )}
    </motion.div>
  );
}

function Info({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center gap-3">
      <Icon className="size-4 shrink-0 text-cocoa/60" />
      <span className="w-16 shrink-0 text-cocoa/70">{label}</span>
      <span className="min-w-0 flex-1 truncate font-medium">{value}</span>
    </div>
  );
}

function HealthBanner({ health }) {
  if (!health) return null;
  const problems = [];
  if (health.face === "failed") problems.push("The face model couldn't load, so photos can't be processed.");
  if (health.hair === "failed")
    problems.push(`The hair/clothes model couldn't load, so try-ons are paused (otherwise clothes could change). ${health.hairError || ""}`);
  if (!health.openaiKey) problems.push("OPENAI_API_KEY is missing in .env.");
  if (!problems.length) return null;
  return (
    <div className="mb-5 flex gap-3 rounded-2xl bg-red-50 p-4 text-sm text-red-900 ring-1 ring-red-200">
      <AlertTriangle className="size-5 shrink-0" />
      <div className="space-y-1">
        <p className="font-semibold">Try-on needs attention</p>
        {problems.map((p) => <p key={p}>{p}</p>)}
        <p className="text-red-800/80">See the server console for the fix. The server retries every 2 minutes.</p>
      </div>
    </div>
  );
}
