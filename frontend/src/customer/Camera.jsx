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
      <video
        ref={video}
        playsInline
        muted
        className={`absolute inset-0 size-full object-cover ${facing === "user" ? "-scale-x-100" : ""}`}
      />

      {/* Face guide: darken everything outside an oval */}
      <svg className="absolute inset-0 size-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <mask id="oval">
            <rect width="100" height="100" fill="white" />
            <ellipse cx="50" cy="40" rx="27" ry="23" fill="black" />
          </mask>
        </defs>
        <rect width="100" height="100" fill="rgba(20,12,8,.55)" mask="url(#oval)" />
        <ellipse cx="50" cy="40" rx="27" ry="23" fill="none" stroke="white" strokeOpacity=".85"
          strokeWidth=".5" strokeDasharray="1.6 1.2" vectorEffect="non-scaling-stroke" />
      </svg>

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
      <div className="absolute inset-x-0 top-[68%] px-6 text-center">
        <p className="font-display text-2xl">Fit your face in the oval</p>
        <p className="mt-1 text-sm text-white/80">Look straight ahead · good light · hair pulled back if you can</p>
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
