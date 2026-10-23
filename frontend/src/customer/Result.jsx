import { Camera, Download, Share2, ShoppingBag, ShieldCheck, RotateCcw, Sparkles } from "lucide-react";
import { Button, CompareSlider } from "../ui/ui.jsx";

export default function Result({ looks, active, setActive, product, onNewPhoto, onStartAgain, onPickStyle, onShop }) {
  const look = looks[active];
  const p = product(look.wigId) || { name: "Your look" };
  const fileName = `my-${look.wigId}.jpg`;

  async function share() {
    const blob = await brandedImage(look.after);
    const file = new File([blob], fileName, { type: "image/jpeg" });
    if (navigator.canShare?.({ files: [file] })) {
      navigator.share({ files: [file], title: p.name, text: `My new look: ${p.name} from Be Beautiful Hair` }).catch(() => {});
    } else {
      download(blob);
    }
  }

  async function download(blob) {
    const b = blob instanceof Blob ? blob : await brandedImage(look.after);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = fileName;
    a.click();
  }

  return (
    <div className="flex h-full flex-col">
      <header className="safe-top flex items-center gap-2 px-4 pb-2">
        <div className="min-w-0 flex-1 px-1">
          <p className="text-xs font-semibold uppercase tracking-wider text-brand">Step 3 of 3 · Your look</p>
          <h1 className="truncate font-display text-xl leading-tight">{p.name}</h1>
        </div>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto no-scrollbar px-4 pb-4">
        <CompareSlider
          key={look.wigId}
          before={look.before}
          after={look.after}
          className="mx-auto aspect-[2/3] h-[min(50dvh,580px)] max-w-full rounded-[28px] bg-sand shadow-lift"
        />

        <div className="mt-3 flex items-center justify-between gap-2">
          <p className="text-xs text-cocoa">Drag the slider to compare</p>
          <span className="inline-flex items-center gap-1 rounded-full bg-white px-2.5 py-1 text-[11px] font-semibold text-cocoa ring-1 ring-espresso/10">
            <ShieldCheck className="size-3.5 text-brand" /> Still you, just glowing
          </span>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2.5">
          <Action icon={Download} label="Save photo" onClick={download} />
          <Action icon={Share2} label="Share" onClick={share} />
        </div>

        {looks.length > 1 && (
          <div className="mt-6">
            <p className="mb-2 text-sm font-semibold">All your looks</p>
            <div className="flex gap-2.5 overflow-x-auto no-scrollbar pb-1">
              {looks.map((l, i) => (
                <button
                  key={l.wigId}
                  onClick={() => setActive(i)}
                  className={`shrink-0 overflow-hidden rounded-2xl transition ${i === active ? "ring-[3px] ring-brand" : "ring-1 ring-espresso/10 opacity-80"}`}
                >
                  <img src={l.after} alt={product(l.wigId)?.name} className="h-28 w-[75px] object-cover" />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="space-y-2.5 px-4 pt-2 safe-bottom">
        {p.buyUrl && (
          <Button onClick={() => (onShop ? onShop(p.buyUrl) : window.open(p.buyUrl, "_blank", "noopener"))} className="w-full h-14 text-base">
            <ShoppingBag className="size-5" /> Shop this look
          </Button>
        )}
        {onPickStyle && (
          <Button variant={p.buyUrl ? "dark" : "primary"} onClick={onPickStyle} className="w-full h-12">
            <Sparkles className="size-4" /> Try another style
          </Button>
        )}
        <div className="grid grid-cols-2 gap-2.5">
          <Button variant="soft" onClick={onNewPhoto} className="h-12">
            <Camera className="size-4" /> New photo
          </Button>
          <Button variant="soft" onClick={onStartAgain} className="h-12">
            <RotateCcw className="size-4" /> Start again
          </Button>
        </div>
      </div>
    </div>
  );
}

function Action({ icon: Icon, label, onClick }) {
  return (
    <button
      onClick={onClick}
      className="flex flex-col items-center gap-1.5 rounded-2xl bg-white py-3 text-[13px] font-semibold ring-1 ring-espresso/10 active:scale-95 transition"
    >
      <Icon className="size-5 text-brand" />
      {label}
    </button>
  );
}

/** The try-on photo with a soft bottom band, the brand logo and website, for saving and sharing. */
async function brandedImage(src) {
  const load = (u) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = u; });
  const img = await load(src);
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const w = c.width, h = c.height, pad = Math.round(w * 0.035);

  const g = ctx.createLinearGradient(0, h * 0.82, 0, h);
  g.addColorStop(0, "rgba(34,34,34,0)");
  g.addColorStop(1, "rgba(34,34,34,0.62)");
  ctx.fillStyle = g;
  ctx.fillRect(0, h * 0.82, w, h * 0.18);

  try {
    const logo = await load("/logo.png");
    const s = Math.round(w * 0.12);
    const x = w - s - pad, y = h - s - pad;
    ctx.save();
    ctx.beginPath();
    ctx.arc(x + s / 2, y + s / 2, s / 2, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(logo, x, y, s, s);
    ctx.restore();
  } catch { /* logo missing: text only */ }

  await document.fonts?.ready;
  ctx.fillStyle = "#fff";
  ctx.font = `400 ${Math.round(w * 0.044)}px "Young Serif", Georgia, serif`;
  ctx.fillText("Own your blend!", pad, h - pad - Math.round(w * 0.045));
  ctx.font = `500 ${Math.round(w * 0.027)}px "Reem Kufi", system-ui, sans-serif`;
  ctx.fillText("bebeautifulhair.com", pad, h - pad);

  return new Promise((res) => c.toBlob(res, "image/jpeg", 0.92));
}
