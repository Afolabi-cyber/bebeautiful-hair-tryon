import { useState } from "react";
import { motion } from "motion/react";
import { ShieldCheck, ArrowRight } from "lucide-react";
import { Button } from "../ui/ui.jsx";

export default function Welcome({ brand, products, preset, initial, onContinue }) {
  const [name, setName] = useState(initial?.name || "");
  const [phone, setPhone] = useState(initial?.phone || "");
  const [consent, setConsent] = useState(false);
  const [touched, setTouched] = useState(false);

  const nameOk = name.trim().length >= 2;
  const phoneOk = phone.replace(/\D/g, "").length >= 7;
  const ready = nameOk && phoneOk && consent;
  // From a product page, that wig takes the centre spot.
  const others = products.filter((p) => p.id !== preset?.id);
  const shots = preset ? [others[0], preset, others[1]].filter(Boolean) : products.slice(0, 3);

  function submit(e) {
    e.preventDefault();
    setTouched(true);
    if (ready) onContinue({ name: name.trim(), phone: phone.trim() });
  }

  return (
    <div className="h-full overflow-y-auto no-scrollbar">
      {/* Hero */}
      <div className="relative h-[38dvh] min-h-64 overflow-hidden bg-gradient-to-b from-blush via-[#FFF3F9] to-cream">
        <div className="absolute inset-0 flex items-end justify-center gap-3 px-6 pb-2">
          {shots.map((p, i) => (
            <motion.img
              key={p.id}
              src={`/${p.thumb || p.image}`}
              alt=""
              className={`w-[30%] rounded-[28px] object-cover aspect-[2/3] bg-sand shadow-lift ring-4 ring-cream
                ${i === 1 ? "mb-10 z-10" : "mb-2 opacity-95"}`}
              initial={{ opacity: 0, y: 30, rotate: i === 0 ? -6 : i === 2 ? 6 : 0 }}
              animate={{ opacity: 1, y: 0, rotate: i === 0 ? -6 : i === 2 ? 6 : 0 }}
              transition={{ delay: 0.1 + i * 0.12, type: "spring", stiffness: 120, damping: 16 }}
            />
          ))}
        </div>
        <div className="absolute inset-x-0 top-0 safe-top px-6 flex items-center gap-2.5">
          <img src="/logo.png" alt={brand.name || "Be Beautiful Hair"} className="size-11 rounded-full shadow-soft ring-2 ring-white" />
          <div className="leading-tight">
            <p className="text-[16px] font-bold">{brand.name || "Be Beautiful Hair"}</p>
            <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-brand">Virtual Try-On</p>
          </div>
        </div>
      </div>

      <form onSubmit={submit} className="px-6 pt-5 safe-bottom">
        <h1 className="font-display text-[38px] leading-[1.05]">
          Own your <span className="text-brand">look!</span>
        </h1>
        <p className="mt-2.5 text-[15px] leading-relaxed text-cocoa">
          {preset
            ? <>See yourself in the <span className="font-semibold text-espresso">{preset.name}</span> on your own photo before you buy. It takes about a minute.</>
            : "Try our styles on your own photo before you buy. It takes about a minute."}
        </p>

        <div className="mt-6 space-y-3">
          <Field
            label="Your name"
            value={name}
            onChange={setName}
            autoComplete="name"
            placeholder="e.g. Amaka Johnson"
            error={touched && !nameOk && "Please enter your name"}
          />
          <Field
            label="Phone number"
            value={phone}
            onChange={setPhone}
            type="tel"
            inputMode="tel"
            autoComplete="tel"
            placeholder="0803 123 4567"
            error={touched && !phoneOk && "Please enter a valid phone number"}
          />
        </div>

        <label className="mt-4 flex gap-3 items-start rounded-2xl bg-white/70 p-3.5 ring-1 ring-espresso/5 cursor-pointer">
          <input
            type="checkbox"
            checked={consent}
            onChange={(e) => setConsent(e.target.checked)}
            className="mt-0.5 size-5 shrink-0 accent-brand"
          />
          <span className="text-[13px] leading-snug text-cocoa">
            I agree that my photo is processed by AI to show the wig on me, and that {brand.name || "the store"} keeps
            my photo, name and number to follow up on my order.
          </span>
        </label>
        {touched && !consent && <p className="mt-2 text-sm text-red-700">Please tick the box to continue</p>}

        <Button type="submit" className="mt-6 w-full h-14 text-base">
          Take my photo <ArrowRight className="size-5" />
        </Button>

        <p className="mt-4 flex items-center justify-center gap-1.5 text-xs text-cocoa/80">
          <ShieldCheck className="size-3.5" /> Your face stays yours. Only the hair changes.
        </p>
        <p className="mt-5 mb-2 text-center font-display text-[15px] text-sky">Loved By Queens Everywhere</p>
      </form>
    </div>
  );
}

function Field({ label, error, onChange, ...props }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[13px] font-semibold text-cocoa">{label}</span>
      <input
        {...props}
        onChange={(e) => onChange(e.target.value)}
        className={`w-full h-13 rounded-2xl bg-white px-4 text-[16px] outline-none ring-1 transition
          placeholder:text-cocoa/40 focus:ring-2 focus:ring-brand ${error ? "ring-red-400" : "ring-espresso/10"}`}
      />
      {error && <span className="mt-1 block text-sm text-red-700">{error}</span>}
    </label>
  );
}