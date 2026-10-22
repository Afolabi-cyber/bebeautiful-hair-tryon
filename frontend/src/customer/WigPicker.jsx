import { motion } from "motion/react";
import { Check, Camera, Sparkles, Images } from "lucide-react";
import { Button } from "../ui/ui.jsx";

export default function WigPicker({
  brand, customer, selfie, products, selected, tried, onSelect, onTry, onRetake, onShowLooks, error,
}) {
  const wig = products.find((p) => p.id === selected);
  const first = customer?.name?.split(" ")[0];

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="safe-top px-5 pb-3 bg-cream/90 backdrop-blur-md z-10">
        <div className="flex items-center gap-3">
          <button onClick={onRetake} className="relative shrink-0" aria-label="Retake photo">
            {selfie ? (
              <img src={selfie} alt="" className="size-12 rounded-full object-cover ring-2 ring-brand ring-offset-2 ring-offset-cream" />
            ) : (
              <span className="grid size-12 place-items-center rounded-full bg-sand"><Camera className="size-5" /></span>
            )}
            <span className="absolute -bottom-0.5 -right-0.5 grid size-5 place-items-center rounded-full bg-espresso text-cream">
              <Camera className="size-3" />
            </span>
          </button>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold uppercase tracking-wider text-brand">Choose your style</p>
            <h1 className="font-display text-[26px] leading-tight">{first ? `Pick a style, ${first}` : "Pick a style"}</h1>
          </div>
          <button onClick={onRetake} className="shrink-0 rounded-full bg-white px-3 py-2 text-xs font-semibold ring-1 ring-espresso/10 active:scale-95 transition">
            New photo
          </button>
        </div>
        {onShowLooks && (
          <button onClick={onShowLooks} className="mt-3 flex w-full items-center gap-2 rounded-2xl bg-espresso px-4 py-2.5 text-left text-cream active:scale-[.98] transition">
            <Images className="size-4" />
            <span className="flex-1 text-sm font-semibold">See your looks ({tried.size})</span>
            <span className="text-xs text-cream/70">View</span>
          </button>
        )}
      </header>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto no-scrollbar px-5 pb-36">
        {error && <p className="mb-3 rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-800 ring-1 ring-red-200">{error}</p>}
        <div className="grid grid-cols-2 gap-3.5">
          {products.map((p, i) => {
            const on = p.id === selected;
            return (
              <motion.button
                key={p.id}
                onClick={() => onSelect(p.id)}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                whileTap={{ scale: 0.97 }}
                className={`group relative overflow-hidden rounded-[24px] bg-gradient-to-b from-[#ECE9E6] to-[#DCD7D2] text-left
                  transition-shadow ${on ? "ring-[3px] ring-brand shadow-lift" : "ring-1 ring-espresso/5 shadow-soft"}`}
              >
                <img src={`/${p.thumb || p.image}`} alt={p.name} loading="lazy" className="aspect-[2/3] w-full object-cover" />
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-espresso/85 via-espresso/40 to-transparent p-3 pt-10">
                  <p className="text-[13.5px] font-semibold leading-tight text-cream">{p.name}</p>
                  {p.price && <p className="text-xs text-cream/80 mt-0.5">{p.price}</p>}
                </div>
                {on && (
                  <span className="absolute top-2.5 right-2.5 grid size-7 place-items-center rounded-full bg-brand text-cream shadow-soft">
                    <Check className="size-4" strokeWidth={3} />
                  </span>
                )}
                {tried.has(p.id) && !on && (
                  <span className="absolute top-2.5 left-2.5 rounded-full bg-white/90 px-2 py-0.5 text-[11px] font-semibold text-espresso">
                    Tried
                  </span>
                )}
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Sticky CTA */}
      <div className="absolute inset-x-0 bottom-0 px-5 pt-6 safe-bottom bg-gradient-to-t from-cream via-cream to-cream/0">
        <Button onClick={onTry} disabled={!wig} className="w-full h-14 text-base">
          <Sparkles className="size-5" />
          {wig ? (tried.has(wig.id) ? `View ${wig.name}` : `Try on ${wig.name}`) : "Pick a style to try"}
        </Button>
      </div>
    </div>
  );
}
