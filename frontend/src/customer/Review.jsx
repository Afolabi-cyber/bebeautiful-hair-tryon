import { useState } from "react";
import { RotateCcw, Check } from "lucide-react";
import { Button } from "../ui/ui.jsx";

export default function Review({ url, name, onRetake, onConfirm }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function confirm() {
    setBusy(true);
    setError("");
    try {
      await onConfirm();
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full flex-col safe-top">
      <div className="px-6 pt-2">
        <p className="text-sm font-semibold text-brand">Step 2 of 4 · Your photo</p>
        <h1 className="font-display text-3xl tracking-tight">Looking good{name ? `, ${name.split(" ")[0]}` : ""}</h1>
        <p className="text-cocoa text-sm mt-1">Is your face clear and well lit? This is the photo we'll style.</p>
      </div>

      <div className="relative mx-6 mt-4 flex-1 min-h-0 overflow-hidden rounded-[28px] bg-sand shadow-lift">
        <img src={url} alt="Your photo" className="size-full object-cover" />
        {busy && (
          <div className="absolute inset-0 grid place-items-center bg-espresso/40 backdrop-blur-[2px] text-cream">
            <p className="rounded-full bg-black/40 px-4 py-2 text-sm font-semibold">Checking your photo…</p>
          </div>
        )}
      </div>

      {error && <p className="mx-6 mt-3 rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-800 ring-1 ring-red-200">{error}</p>}

      <div className="grid grid-cols-2 gap-3 px-6 pt-4 safe-bottom">
        <Button variant="soft" onClick={onRetake} disabled={busy} className="h-14">
          <RotateCcw className="size-4" /> Retake
        </Button>
        <Button onClick={confirm} loading={busy} className="h-14">
          {!busy && <Check className="size-5" />} Use photo
        </Button>
      </div>
    </div>
  );
}
