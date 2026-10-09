import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";

const LINES = [
  "Fitting your wig",
  "Laying the braids",
  "Matching your lighting",
  "Blending the hairline",
  "Keeping your face exactly as it is",
  "Final touches",
];

export default function Generating({ selfie, wig }) {
  const [i, setI] = useState(0);
  const [progress, setProgress] = useState(4);

  useEffect(() => {
    const a = setInterval(() => setI((x) => Math.min(x + 1, LINES.length - 1)), 7000);
    // Eases toward 94% over roughly a minute; the real result replaces this screen when ready.
    const b = setInterval(() => setProgress((p) => p + (94 - p) * 0.045), 600);
    return () => { clearInterval(a); clearInterval(b); };
  }, []);

  return (
    <div className="flex h-full flex-col items-center justify-center px-8 safe-top safe-bottom">
      <div className="relative w-full max-w-[290px]">
        <div className="relative aspect-[3/4] overflow-hidden rounded-[32px] bg-sand shadow-lift">
          {selfie && <img src={selfie} alt="" className="size-full object-cover" />}
          <div className="absolute inset-0 bg-gradient-to-b from-brand/10 via-transparent to-espresso/30" />
          <div className="scan-line absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-gold/45 to-transparent mix-blend-screen" />
          <div className="absolute inset-0 shimmer" />
        </div>
        {wig && (
          <motion.img
            src={`/${wig.thumb || wig.image}`}
            alt=""
            className="absolute -bottom-6 -right-5 w-24 aspect-[2/3] rounded-2xl object-cover bg-[#E3DFDB] ring-4 ring-cream shadow-lift"
            initial={{ scale: 0.6, rotate: 12, opacity: 0 }}
            animate={{ scale: 1, rotate: 6, opacity: 1 }}
            transition={{ type: "spring", stiffness: 140, damping: 14 }}
          />
        )}
      </div>

      <div className="mt-12 h-16 text-center">
        <AnimatePresence mode="wait">
          <motion.p
            key={i}
            className="font-display text-2xl tracking-tight"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            {LINES[i]}…
          </motion.p>
        </AnimatePresence>
        <p className="mt-1 text-sm text-cocoa">{wig?.name}</p>
      </div>

      <div className="mt-4 h-1.5 w-full max-w-[290px] overflow-hidden rounded-full bg-sand">
        <div className="h-full rounded-full bg-brand transition-[width] duration-500" style={{ width: `${progress}%` }} />
      </div>
      <p className="mt-3 text-xs text-cocoa/80">Usually 20 to 60 seconds. Please keep this page open.</p>
    </div>
  );
}
