import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";

const LINES = [
  "Fitting your wig",
  "Laying the braids",
  "Shaping the hairline",
  "Matching your lighting",
  "Keeping your face exactly as it is",
  "Blending every strand",
  "Adding the glow",
  "Almost there",
];
const TIPS = [
  "You can try more styles after this one, on the same photo.",
  "Drag the slider on your result to compare before and after.",
  "Save your look or share it with friends straight from the result.",
  "Love it? Tap Shop this look to order this wig.",
];

export default function Generating({ selfie, wig }) {
  const [t, setT] = useState(0); // seconds since start

  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => setT((Date.now() - start) / 1000), 250);
    return () => clearInterval(id);
  }, []);

  const i = Math.min(LINES.length - 1, Math.floor(t / 7.5));
  const tip = Math.floor(t / 6) % TIPS.length;
  // Rises steadily over about a minute and never reaches 100% before the real result arrives.
  const progress = Math.min(97, 4 + 93 * (1 - Math.exp(-t / 24)));

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
        <div className="h-full rounded-full bg-brand transition-[width] duration-300" style={{ width: `${progress}%` }} />
      </div>
      <p className="mt-2 text-xs tabular-nums text-cocoa/80">{Math.floor(t)}s · usually under a minute. Please keep this page open.</p>

      <div className="mt-6 min-h-14 w-full max-w-[300px] rounded-2xl bg-blush/70 px-4 py-3 text-center">
        <AnimatePresence mode="wait">
          <motion.p
            key={tip}
            className="text-[13px] leading-snug text-espresso/85"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <span className="font-semibold text-brand">Tip: </span>{TIPS[tip]}
          </motion.p>
        </AnimatePresence>
      </div>
    </div>
  );
}