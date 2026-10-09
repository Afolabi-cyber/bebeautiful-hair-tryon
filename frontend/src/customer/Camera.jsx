import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowLeft, SwitchCamera, ImageUp, Timer, TimerOff } from "lucide-react";
import { IconButton } from "../ui/ui.jsx";

const MAX_SIDE = 1800;

export default function Camera({ onBack, onCapture, error }) {
  const video = useRef(null);
  const stream = useRef(null);
  const [facing, setFacing] = useState("user");
  const [ready, setReady] = useState(false);
  const [denied, setDenied] = useState(false);
  const [useTimer, setUseTimer] = useState(true);
  const [count, setCount] = useState(0);
  const [flash, setFlash] = useState(false);
  // Where the camera picture actually sits on screen (it is shown whole, not zoomed in),
  // so the head guide can be drawn over the real frame.
  const box = useRef(null);
  const [frame, setFrame] = useState(null);

  useEffect(() => {
    const measure = () => {
      const v = video.current, b = box.current;
      if (!v || !b || !v.videoWidth) return;
      const W = b.clientWidth, H = b.clientHeight, a = v.videoWidth / v.videoHeight;
      // Fill the width; only zoom (cover) when the picture would otherwise be a thin strip.
      let w = W, h = W / a;
      if (h < H * 0.62) { h = H * 0.62; w = h * a; }
      if (h > H) { h = H; w = h * a; }
      // Never wider than the screen: a wide (landscape) camera is cropped at the sides instead.
      w = Math.min(w, W);
      setFrame({ w, h, left: (W - w) / 2, top: Math.max(0, (H - h) / 2 - H * 0.06) });
    };
    const ro = new ResizeObserver(measure);
    if (box.current) ro.observe(box.current);
    const v = video.current;
    v?.addEventListener("loadedmetadata", measure);
    v?.addEventListener("resize", measure);
    measure();
    return () => { ro.disconnect(); v?.removeEventListener("loadedmetadata", measure); v?.removeEventListener("resize", measure); };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setReady(false);
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: facing, width: { ideal: 1440 }, height: { ideal: 1920 } },
          audio: false,
        });
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream.current = s;
        // Widest view the front camera offers (some phones start zoomed in).
        try {
          const t = s.getVideoTracks()[0];
          const z = t.getCapabilities?.().zoom;
          if (z && z.min < (t.getSettings().zoom ?? z.min)) await t.applyConstraints({ advanced: [{ zoom: z.min }] });
        } catch { /* not supported: fine */ }
        video.current.srcObject = s;
        await video.current.play();
        setDenied(false);
        setReady(true);
      } catch {
        setDenied(true);
      }
    })();
    return () => {
      cancelled = true;
      stream.current?.getTracks().forEach((t) => t.stop());
    };
  }, [facing]);

  async function shoot() {
    if (!ready || count) return;
    if (useTimer) {
      for (let i = 3; i > 0; i--) {
        setCount(i);
        await new Promise((r) => setTimeout(r, 900));
      }
      setCount(0);
    }
    const v = video.current;
    const k = Math.min(1, MAX_SIDE / Math.max(v.videoWidth, v.videoHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(v.videoWidth * k);
    c.height = Math.round(v.videoHeight * k);
    const ctx = c.getContext("2d");
    if (facing === "user") { ctx.translate(c.width, 0); ctx.scale(-1, 1); } // keep it as the customer saw it
    ctx.drawImage(v, 0, 0, c.width, c.height);
    setFlash(true);
    setTimeout(() => setFlash(false), 180);
    c.toBlob((b) => onCapture(b), "image/jpeg", 0.93);
  }

  async function upload(file) {
    if (!file) return;
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const k = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * k);
    c.height = Math.round(bmp.height * k);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    c.toBlob((b) => onCapture(b), "image/jpeg", 0.93);
  }

  return (
    <div className="relative h-full bg-black text-white overflow-hidden">
      <div ref={box} className="absolute inset-0">
        <video
          ref={video}
          playsInline
          muted
          style={frame ? { width: frame.w, height: frame.h, left: frame.left, top: frame.top } : undefined}
          className={`absolute max-w-none object-cover ${frame ? "" : "inset-0 size-full"} ${facing === "user" ? "-scale-x-100" : ""}`}
        />

        {/* Head guide: a big, forgiving outline over the real camera picture; everything else is dimmed */}
        {frame && (
          <div
            className="pointer-events-none absolute rounded-[50%] border-2 border-dashed border-white/85 transition-all"
            style={(() => {
              const w = Math.min(frame.w * 0.7, frame.h * 0.5);
              const h = w * 1.32;
              return {
                width: w, height: h,
                left: frame.left + (frame.w - w) / 2,
                top: frame.top + frame.h * 0.43 - h / 2,
                boxShadow: "0 0 0 200vmax rgba(20,12,8,.5)",
              };
            })()}
          />
        )}
      </div>

      {/* Top bar */}
      <div className="absolute inset-x-0 top-0 safe-top px-4 flex items-center justify-between">
        <IconButton label="Back" onClick={onBack} className="bg-black/30 backdrop-blur-md">
          <ArrowLeft className="size-5" />
        </IconButton>
        <IconButton label={useTimer ? "Timer on" : "Timer off"} onClick={() => setUseTimer(!useTimer)} className="bg-black/30 backdrop-blur-md">
          {useTimer ? <Timer className="size-5" /> : <TimerOff className="size-5" />}
        </IconButton>
      </div>

      {/* Tips */}
      <div className="absolute inset-x-0 bottom-32 safe-bottom px-6 text-center">
        <p className="font-display text-2xl">Your head in the outline</p>
        <p className="mt-1 text-sm text-white/80">It doesn't have to be exact. Hold the phone at arm's length so the top of your head shows, look straight ahead, good light.</p>
        {error && <p className="mx-auto mt-3 max-w-xs rounded-2xl bg-red-500/90 px-4 py-2 text-sm">{error}</p>}
      </div>

      {denied && (
        <div className="absolute inset-0 grid place-items-center bg-espresso px-8 text-center">
          <div>
            <p className="font-display text-2xl">Camera unavailable</p>
            <p className="mt-2 text-white/75 text-sm">Allow camera access in your browser settings, or upload a clear selfie instead.</p>
            <label className="mt-6 inline-flex h-12 cursor-pointer items-center gap-2 rounded-full bg-brand px-6 font-semibold">
              <ImageUp className="size-5" /> Upload a photo
              <input type="file" accept="image/*" hidden onChange={(e) => upload(e.target.files[0])} />
            </label>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="absolute inset-x-0 bottom-0 safe-bottom px-8 pb-4 flex items-center justify-between">
        <label className="grid size-12 cursor-pointer place-items-center rounded-full bg-white/15 backdrop-blur-md ring-1 ring-white/25 active:scale-90 transition">
          <ImageUp className="size-5" />
          <input type="file" accept="image/*" hidden onChange={(e) => upload(e.target.files[0])} />
        </label>
        <button
          aria-label="Take photo"
          onClick={shoot}
          disabled={!ready}
          className="grid size-20 place-items-center rounded-full ring-4 ring-white/90 active:scale-90 transition disabled:opacity-40"
        >
          <span className="size-16 rounded-full bg-white" />
        </button>
        <IconButton label="Switch camera" onClick={() => setFacing(facing === "user" ? "environment" : "user")}
          className="size-12 bg-white/15 backdrop-blur-md ring-1 ring-white/25">
          <SwitchCamera className="size-5" />
        </IconButton>
      </div>

      <AnimatePresence>
        {count > 0 && (
          <motion.div
            key={count}
            className="absolute inset-0 grid place-items-center pointer-events-none"
            initial={{ scale: 1.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <span className="font-display text-[120px] drop-shadow-lg">{count}</span>
          </motion.div>
        )}
      </AnimatePresence>
      {flash && <div className="absolute inset-0 bg-white" />}
    </div>
  );
}