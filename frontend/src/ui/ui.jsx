import { useRef, useState, useCallback } from "react";
import { Loader2 } from "lucide-react";

export function Button({ variant = "primary", loading, className = "", children, ...props }) {
  const styles = {
    primary: "bg-brand text-cream shadow-soft active:bg-brand-dark disabled:bg-brand/40",
    dark: "bg-espresso text-cream shadow-soft active:bg-black disabled:bg-espresso/40",
    soft: "bg-white text-espresso ring-1 ring-espresso/10 active:bg-sand",
    ghost: "bg-transparent text-espresso active:bg-espresso/5",
    glass: "bg-white/15 text-white backdrop-blur-md ring-1 ring-white/25 active:bg-white/25",
  }[variant];
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-full px-5 h-12 font-semibold text-[15px]
        transition-all duration-150 active:scale-[0.97] disabled:cursor-not-allowed disabled:active:scale-100 ${styles} ${className}`}
      disabled={loading || props.disabled}
      {...props}
    >
      {loading && <Loader2 className="size-4 animate-spin" />}
      {children}
    </button>
  );
}

export function IconButton({ className = "", children, label, ...props }) {
  return (
    <button
      aria-label={label}
      className={`grid place-items-center size-11 rounded-full transition active:scale-90 ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

/** Drag left/right to compare the original photo with the try-on. */
export function CompareSlider({ before, after, className = "" }) {
  const [pos, setPos] = useState(50);
  const ref = useRef(null);
  const dragging = useRef(false);

  const move = useCallback((clientX) => {
    const r = ref.current.getBoundingClientRect();
    setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)));
  }, []);

  return (
    <div
      ref={ref}
      className={`relative select-none overflow-hidden touch-pan-y ${className}`}
      onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); move(e.clientX); }}
      onPointerMove={(e) => dragging.current && move(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
      onPointerCancel={() => (dragging.current = false)}
    >
      <img src={after} alt="With the wig" className="absolute inset-0 size-full object-cover" draggable={false} />
      <img
        src={before}
        alt="Original photo"
        className="absolute inset-0 size-full object-cover"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        draggable={false}
      />
      <div className="absolute inset-y-0 w-0.5 bg-white/90 shadow-[0_0_12px_rgba(0,0,0,.4)]" style={{ left: `${pos}%` }}>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 size-10 rounded-full bg-white shadow-lift grid place-items-center">
          <svg viewBox="0 0 24 24" className="size-5 text-espresso" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
            <path d="M9 6l-5 6 5 6M15 6l5 6-5 6" />
          </svg>
        </div>
      </div>
      <span className="absolute top-3 left-3 rounded-full bg-black/45 backdrop-blur px-2.5 py-1 text-[11px] font-semibold tracking-wide text-white uppercase">Before</span>
      <span className="absolute top-3 right-3 rounded-full bg-brand/90 backdrop-blur px-2.5 py-1 text-[11px] font-semibold tracking-wide text-white uppercase">After</span>
    </div>
  );
}

export function timeAgo(ts) {
  const s = Math.round(Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 604800) return `${Math.floor(s / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

export function fmtDateTime(ts) {
  return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
