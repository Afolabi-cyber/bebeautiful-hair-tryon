import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Plus, Pencil, Trash2, Eye, EyeOff, ArrowUp, ArrowDown, X, ImagePlus, AlertTriangle } from "lucide-react";
import { adminApi } from "../api.js";
import { Button, IconButton } from "../ui/ui.jsx";

/** Admin: add, edit, hide, reorder and delete the wigs customers can try on. */
export default function WigManager({ onError }) {
  const [items, setItems] = useState(null);
  const [editing, setEditing] = useState(null); // null | "new" | wig object
  const [confirmDel, setConfirmDel] = useState(null);
  const [busyId, setBusyId] = useState(null);

  const load = useCallback(() => {
    adminApi.get("/wigs").then((r) => setItems(r.items)).catch(onError);
  }, [onError]);
  useEffect(() => { load(); }, [load]);

  async function act(id, fn) {
    setBusyId(id);
    try { await fn(); load(); } catch (e) { onError(e); } finally { setBusyId(null); }
  }
  const toggle = (w) => act(w.id, () => {
    const fd = new FormData(); fd.append("active", String(!w.active));
    return adminApi.send(`/wigs/${w.id}`, "PATCH", fd);
  });
  const move = (w, direction) => act(w.id, () => adminApi.send(`/wigs/${w.id}/move?direction=${direction}`, "POST"));
  const remove = (w) => act(w.id, async () => { await adminApi.del(`/wigs/${w.id}`); setConfirmDel(null); });

  return (
    <section>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-display text-2xl">Wigs</h2>
          <p className="text-sm text-cocoa">
            {items ? `${items.filter((w) => w.active).length} live, ${items.filter((w) => !w.active).length} hidden` : "Loading…"}
          </p>
        </div>
        <Button onClick={() => setEditing("new")}><Plus className="size-4" /> Add wig</Button>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {items?.map((w, i) => (
          <div key={w.id} className={`overflow-hidden rounded-3xl bg-white ring-1 ring-espresso/5 ${w.active ? "" : "opacity-60"}`}>
            <div className="relative">
              <img src={`/${w.thumb || w.image}`} alt={w.name} className="aspect-[2/3] w-full object-cover bg-sand" />
              {!w.active && <span className="absolute top-2 left-2 rounded-full bg-espresso px-2 py-0.5 text-[11px] font-semibold text-cream">Hidden</span>}
            </div>
            <div className="p-3">
              <p className="truncate text-sm font-semibold">{w.name}</p>
              <p className="text-xs text-cocoa">{w.price || "No price"}</p>
              <div className="mt-2.5 flex items-center justify-between">
                <div className="flex">
                  <Small label="Move up" disabled={i === 0 || busyId} onClick={() => move(w, "up")}><ArrowUp className="size-4" /></Small>
                  <Small label="Move down" disabled={i === items.length - 1 || busyId} onClick={() => move(w, "down")}><ArrowDown className="size-4" /></Small>
                </div>
                <div className="flex">
                  <Small label={w.active ? "Hide from customers" : "Show to customers"} disabled={busyId} onClick={() => toggle(w)}>
                    {w.active ? <Eye className="size-4" /> : <EyeOff className="size-4" />}
                  </Small>
                  <Small label="Edit" onClick={() => setEditing(w)}><Pencil className="size-4" /></Small>
                  <Small label="Delete" onClick={() => setConfirmDel(w)} className="text-red-700"><Trash2 className="size-4" /></Small>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      {items && !items.length && (
        <p className="mt-4 rounded-3xl bg-white/70 p-8 text-center text-sm text-cocoa ring-1 ring-espresso/5">No wigs yet. Add your first one.</p>
      )}

      <AnimatePresence>
        {editing && (
          <WigForm wig={editing === "new" ? null : editing} onClose={() => setEditing(null)} onSaved={() => { setEditing(null); load(); }} />
        )}
        {confirmDel && (
          <Sheet onClose={() => setConfirmDel(null)} title="Delete wig?">
            <p className="text-sm text-cocoa">“{confirmDel.name}” will be removed for customers and its photo deleted. Past try-ons stay in the history. To take it down temporarily, hide it instead.</p>
            <div className="mt-5 flex justify-end gap-2">
              <Button variant="soft" onClick={() => setConfirmDel(null)}>Cancel</Button>
              <Button className="bg-red-700 active:bg-red-800" loading={busyId === confirmDel.id} onClick={() => remove(confirmDel)}>Delete</Button>
            </div>
          </Sheet>
        )}
      </AnimatePresence>
    </section>
  );
}

function Small({ label, className = "", ...props }) {
  return (
    <button title={label} aria-label={label}
      className={`grid size-8 place-items-center rounded-full hover:bg-sand active:scale-90 transition disabled:opacity-30 ${className}`} {...props} />
  );
}

function Sheet({ title, onClose, children }) {
  return (
    <motion.div className="fixed inset-0 z-40 flex items-end justify-center sm:items-center sm:p-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <div className="absolute inset-0 bg-espresso/50" onClick={onClose} />
      <motion.div
        className="relative max-h-[92dvh] w-full max-w-lg overflow-y-auto rounded-t-[28px] bg-cream p-5 shadow-lift sm:rounded-[28px] safe-bottom"
        initial={{ y: 50, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 50, opacity: 0 }}
        transition={{ type: "spring", stiffness: 260, damping: 28 }}
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className="font-display text-xl">{title}</h3>
          <IconButton label="Close" onClick={onClose} className="bg-white ring-1 ring-espresso/10"><X className="size-5" /></IconButton>
        </div>
        {children}
      </motion.div>
    </motion.div>
  );
}

function WigForm({ wig, onClose, onSaved }) {
  const [f, setF] = useState({
    name: wig?.name || "", price: wig?.price || "", description: wig?.description || "",
    buyUrl: wig?.buyUrl || "", bangs: !!wig?.bangs,
  });
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(wig ? `/${wig.image}` : null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");
  const input = useRef(null);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value });

  function pick(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setFile(file);
    setPreview(URL.createObjectURL(file));
  }

  async function save(e) {
    e.preventDefault();
    if (!wig && !file) return setError("Add a photo of the wig");
    if (f.name.trim().length < 2) return setError("Give the wig a name");
    setBusy(true); setError("");
    const fd = new FormData();
    Object.entries(f).forEach(([k, v]) => fd.append(k, String(v)));
    if (file) fd.append("image", file);
    try {
      const r = await adminApi.send(wig ? `/wigs/${wig.id}` : "/wigs", wig ? "PATCH" : "POST", fd);
      if (r.warning) { setWarning(r.warning); setBusy(false); return; } // saved; show the tip, then close
      onSaved();
    } catch (err) {
      setError(err.message); setBusy(false);
    }
  }

  return (
    <Sheet title={wig ? "Edit wig" : "Add a wig"} onClose={warning ? onSaved : onClose}>
      {warning ? (
        <div>
          <div className="flex gap-3 rounded-2xl bg-amber-50 p-4 text-sm text-amber-900 ring-1 ring-amber-200">
            <AlertTriangle className="size-5 shrink-0" />
            <p>Saved. One tip: {warning}</p>
          </div>
          <Button className="mt-4 w-full" onClick={onSaved}>Done</Button>
        </div>
      ) : (
        <form onSubmit={save} className="space-y-4">
          <button type="button" onClick={() => input.current.click()}
            className="relative grid aspect-[3/4] w-40 place-items-center overflow-hidden rounded-2xl bg-white ring-1 ring-dashed ring-espresso/20 mx-auto">
            {preview ? <img src={preview} alt="" className="absolute inset-0 size-full object-cover" /> : (
              <span className="flex flex-col items-center gap-1 text-xs text-cocoa"><ImagePlus className="size-6" />Add photo</span>
            )}
            {preview && <span className="absolute bottom-2 rounded-full bg-black/55 px-2 py-0.5 text-[11px] text-white">Change</span>}
          </button>
          <input ref={input} type="file" accept="image/*" hidden onChange={pick} />
          <p className="text-center text-xs text-cocoa">Front view of the wig alone, on a plain light background (like the current ones).</p>

          <Field label="Name"><input value={f.name} onChange={set("name")} placeholder="e.g. Honey Blonde Boho" className={inputCls} /></Field>
          <Field label="Price (optional)"><input value={f.price} onChange={set("price")} placeholder="e.g. R 450" className={inputCls} /></Field>
          <Field label="Description for the AI" hint="Length, braid or curl type, colours and ombre, parting. This helps the AI copy the wig exactly.">
            <textarea value={f.description} onChange={set("description")} rows={3} className={`${inputCls} h-auto py-3`}
              placeholder="e.g. waist-length boho knotless braids, middle part, dark roots to honey blonde curly ends" />
          </Field>
          <Field label="Shop link (optional)"><input value={f.buyUrl} onChange={set("buyUrl")} placeholder="https://bebeautifulhair.com/products/…" className={inputCls} /></Field>
          <label className="flex items-center gap-3 text-sm">
            <input type="checkbox" checked={f.bangs} onChange={set("bangs")} className="size-5 accent-brand" />
            This wig has a fringe (bangs) over the forehead
          </label>

          {error && <p className="text-sm text-red-700">{error}</p>}
          <div className="flex justify-end gap-2 pt-1">
            <Button type="button" variant="soft" onClick={onClose}>Cancel</Button>
            <Button type="submit" loading={busy}>{wig ? "Save changes" : "Add wig"}</Button>
          </div>
        </form>
      )}
    </Sheet>
  );
}

const inputCls = "h-12 w-full rounded-2xl bg-white px-4 text-[15px] outline-none ring-1 ring-espresso/10 focus:ring-2 focus:ring-brand";

function Field({ label, hint, children }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[13px] font-semibold text-cocoa">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-cocoa/80">{hint}</span>}
    </label>
  );
}
