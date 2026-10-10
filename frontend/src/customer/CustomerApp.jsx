import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { api } from "../api.js";
import Welcome from "./Welcome.jsx";
import Camera from "./Camera.jsx";
import Review from "./Review.jsx";
import Generating from "./Generating.jsx";
import Result from "./Result.jsx";

// Customers arrive from a product page on the website (?wig=...), so the style is already chosen.
// One straight path: details -> photo -> your look.
const STEP = { welcome: 0, camera: 1, review: 1, generating: 2, failed: 2, result: 2 };
const BACK = { camera: "welcome", review: "camera" };

// Only name and phone are remembered on this phone (to prefill the form next time).
// Photos and results are never restored: every visit starts fresh.
const DETAILS = "bbh-tryon-details";
const loadDetails = () => { try { return JSON.parse(localStorage.getItem(DETAILS)); } catch { return null; } };
const saveDetails = (d) => { try { localStorage.setItem(DETAILS, JSON.stringify(d)); } catch { /* blocked */ } };

// A link from a product page on the website: ?wig=<product handle or wig id>.
// It matches the wig's id, or the product address set as its "Shop link" in the admin.
const LINKED = (() => {
  try { return (new URLSearchParams(window.location.search).get("wig") || "").trim().toLowerCase(); } catch { return ""; }
})();
const handleOf = (url) => {
  const m = /\/products\/([^/?#]+)/.exec(url || "");
  return m ? decodeURIComponent(m[1]).toLowerCase() : "";
};

export default function CustomerApp() {
  const [brand, setBrand] = useState({ name: "Be Beautiful Hair" });
  const [products, setProducts] = useState([]);
  const [customer, setCustomer] = useState(loadDetails); // { name, phone }
  const [sessionId, setSessionId] = useState(null);
  const [selfie, setSelfie] = useState(null); // data URL of the confirmed photo
  const [draft, setDraft] = useState(null); // { blob, url } waiting for review
  const [wigId, setWigId] = useState(null);
  const [looks, setLooks] = useState([]); // [{ wigId, before, after, faceLocked }]
  const [active, setActive] = useState(0);
  const [screen, setScreen] = useState("welcome");
  const [error, setError] = useState("");
  const [preset, setPreset] = useState(null); // wig chosen on the product page
  const [loaded, setLoaded] = useState(false); // catalogue loaded
  const screenRef = useRef(screen);
  screenRef.current = screen;

  useEffect(() => {
    api.catalog().then((c) => {
      setBrand(c.brand || {});
      setProducts(c.products || []);
      if (c.brand?.accent) document.documentElement.style.setProperty("--color-brand", c.brand.accent);
      if (c.brand?.name) document.title = `${c.brand.name} · Virtual Try-On`;
      if (LINKED) {
        const hit = (c.products || []).find((p) => p.id === LINKED || handleOf(p.buyUrl) === LINKED);
        if (hit) setPreset(hit.id);
      }
      setLoaded(true);
    }).catch(() => setError("Couldn't load styles. Check your connection."));
  }, []);

  // Phone back button goes one step back instead of leaving the site.
  useEffect(() => {
    if (screen !== "welcome" && screen !== "generating") window.history.pushState({ screen }, "");
  }, [screen]);
  useEffect(() => {
    const onPop = () => {
      const target = BACK[screenRef.current];
      if (screenRef.current === "generating") window.history.pushState({}, ""); // stay while working
      else if (target) setScreen(target);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [sessionId]);

  const product = (id) => products.find((p) => p.id === id);

  function startAgain() {
    setSessionId(null); setSelfie(null); setDraft(null);
    setWigId(null); setLooks([]); setActive(0); setError("");
    setScreen("welcome");
    window.scrollTo(0, 0);
  }

  async function confirmPhoto() {
    setError("");
    const res = await api.createSession({ ...customer, selfie: draft.blob });
    setSelfie(await blobToDataUrl(draft.blob));
    setSessionId(res.sessionId);
    setLooks([]);
    setDraft(null);
    if (preset) {
      // Came from a product page: the wig is already chosen, so start making the look straight away.
      setWigId(preset);
      tryOn(preset, res.sessionId, []);
    } else {
      setScreen("welcome"); // no style chosen: the welcome screen explains how to start from the shop
    }
  }

  async function tryOn(id = wigId, sid = sessionId, have = looks) {
    const p = product(id);
    if (!p) return;
    const existing = have.findIndex((l) => l.wigId === id);
    if (existing >= 0) { setActive(existing); setScreen("result"); return; }
    setError("");
    setScreen("generating");
    try {
      const r = await api.tryOn(sid, p.id);
      readyAlert();
      setLooks((prev) => [{ wigId: p.id, before: r.before, after: r.after, faceLocked: r.faceLocked }, ...prev]);
      setActive(0);
      setScreen("result");
    } catch (e) {
      setError(e.message);
      if (e.status === 404) { setSessionId(null); setScreen("camera"); }
      else setScreen("failed");
    }
  }

  const views = {
    welcome: loaded && !preset ? <NoStyle brand={brand} unknown={!!LINKED} /> : (
      <Welcome
        brand={brand}
        products={products}
        preset={product(preset)}
        initial={customer}
        onContinue={(c) => { setCustomer(c); saveDetails(c); setError(""); setScreen("camera"); }}
      />
    ),
    camera: (
      <Camera
        onBack={() => setScreen("welcome")}
        onCapture={(blob) => { setDraft({ blob, url: URL.createObjectURL(blob) }); setError(""); setScreen("review"); }}
        error={error}
      />
    ),
    review: draft && (
      <Review url={draft.url} name={customer?.name} onRetake={() => setScreen("camera")} onConfirm={confirmPhoto} />
    ),
    generating: <Generating selfie={selfie} wig={product(wigId)} />,
    failed: (
      <Failed
        message={error}
        onRetry={() => tryOn(wigId)}
        onNewPhoto={() => { setError(""); setScreen("camera"); }}
      />
    ),
    result: looks[active] && (
      <Result
        looks={looks}
        active={active}
        setActive={setActive}
        product={product}
        onNewPhoto={() => { setError(""); setScreen("camera"); }}
        onStartAgain={startAgain}
      />
    ),
  };

  return (
    <div className="mx-auto h-full max-w-md bg-cream md:shadow-lift relative overflow-hidden">
      <Progress step={STEP[screen]} dark={screen === "camera"} />
      <AnimatePresence mode="wait">
        <motion.div
          key={screen}
          className="absolute inset-0"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.22, ease: "easeOut" }}
        >
          {views[screen] || views.welcome}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

/** Three thin segments along the top: details, photo, your look. */
function Progress({ step, dark }) {
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-30 flex gap-1.5 px-4 pt-[max(env(safe-area-inset-top),4px)]">
      {[0, 1, 2].map((i) => (
        <div key={i} className={`h-[3px] flex-1 overflow-hidden rounded-full ${dark ? "bg-white/25" : "bg-espresso/10"}`}>
          <div className={`h-full rounded-full bg-brand transition-all duration-500 ${i <= step ? "w-full" : "w-0"}`} />
        </div>
      ))}
    </div>
  );
}

/** Opened without a product link (or with one for a style that isn't set up for try-on). */
function NoStyle({ brand, unknown }) {
  const shop = brand.website || "https://bebeautifulhair.com";
  return (
    <div className="flex h-full flex-col items-center justify-center px-8 text-center safe-top safe-bottom">
      <img src="/logo.png" alt={brand.name || "Be Beautiful Hair"} className="size-16 rounded-full shadow-soft ring-2 ring-white" />
      <h1 className="mt-6 font-display text-[30px] leading-tight">
        {unknown ? "This style isn't ready for try-on yet" : <>Pick a style, then <span className="text-brand">try it on</span></>}
      </h1>
      <p className="mt-3 max-w-xs text-[15px] leading-relaxed text-cocoa">
        {unknown
          ? "Try-on isn't available for this style yet. Have a look at our other styles and tap Try it on where you see it."
          : "Open any style on our website and tap Try it on to see it on your own photo."}
      </p>
      <a href={shop} className="mt-8 inline-flex h-14 items-center gap-2 rounded-full bg-brand px-8 text-base font-semibold text-cream shadow-soft active:bg-brand-dark">
        Browse styles
      </a>
    </div>
  );
}

/** The AI step failed (busy, network): try the same photo again or take a new one. */
function Failed({ message, onRetry, onNewPhoto }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-8 text-center safe-top safe-bottom">
      <h1 className="font-display text-[30px] leading-tight">That didn't work this time</h1>
      <p className="mt-3 max-w-xs text-[15px] leading-relaxed text-cocoa">{message || "Something went wrong. Please try again."}</p>
      <div className="mt-8 flex w-full max-w-xs flex-col gap-2.5">
        <button onClick={onRetry} className="h-14 rounded-full bg-brand text-base font-semibold text-cream shadow-soft active:bg-brand-dark">Try again</button>
        <button onClick={onNewPhoto} className="h-12 rounded-full bg-white text-sm font-semibold ring-1 ring-espresso/10 active:bg-sand">Take a new photo</button>
      </div>
    </div>
  );
}

/** If she switched to another app while waiting: a short buzz and a tab title saying it's ready. */
function readyAlert() {
  try { navigator.vibrate?.(80); } catch { /* not supported */ }
  if (document.hidden) {
    const t = document.title;
    document.title = "Your look is ready!";
    const back = () => { document.title = t; document.removeEventListener("visibilitychange", back); };
    document.addEventListener("visibilitychange", back);
  }
}

function blobToDataUrl(blob) {
  return new Promise((resolve) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.readAsDataURL(blob);
  });
}