/*
 * Be Beautiful Hair virtual try-on, inside the product page.
 *
 * Add once to the site (theme footer or product template):
 *   <script src="https://YOUR-TRYON-ADDRESS/embed.js" defer></script>
 * Then any button or link with data-bbh-tryon opens the try-on in a pop-up on the same page:
 *   <button data-bbh-tryon>Try it on</button>                      (uses this product page's handle)
 *   <button data-bbh-tryon="knotless-braids-30">Try it on</button>  (a specific wig: its id or product handle)
 *   <button data-bbh-tryon="all">Try on our styles</button>         (she picks a style in the app)
 */
(function () {
  if (window.__bbhTryon) return;
  window.__bbhTryon = true;
  var me = document.currentScript || document.querySelector('script[src*="embed.js"]');
  var APP = new URL(me.src).origin;
  var wrap, frame, lastFocus, scrollY = 0;

  function handleFromPage() {
    var m = /\/products\/([^/?#]+)/.exec(location.pathname);
    return m ? decodeURIComponent(m[1]) : "";
  }

  function build() {
    wrap = document.createElement("div");
    wrap.setAttribute("role", "dialog");
    wrap.setAttribute("aria-modal", "true");
    wrap.setAttribute("aria-label", "Virtual try-on");
    wrap.style.cssText = "position:fixed;inset:0;z-index:2147483646;background:rgba(20,20,20,.6);display:none;" +
      "align-items:center;justify-content:center;";
    var box = document.createElement("div");
    box.style.cssText = "position:relative;width:100%;height:100%;max-width:440px;max-height:900px;background:#fff;" +
      "display:flex;flex-direction:column;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.35);";
    if (window.matchMedia("(min-width: 600px)").matches) box.style.borderRadius = "20px", box.style.height = "92vh";
    var bar = document.createElement("div");
    bar.style.cssText = "flex:0 0 44px;display:flex;align-items:center;justify-content:space-between;padding:0 8px 0 16px;" +
      "background:#222;color:#fff;font:600 14px/1 system-ui,sans-serif;";
    bar.innerHTML = '<span>Virtual Try-On</span>';
    var x = document.createElement("button");
    x.type = "button";
    x.setAttribute("aria-label", "Close try-on");
    x.innerHTML = "&#x2715;";
    x.style.cssText = "width:36px;height:36px;border:0;border-radius:50%;background:rgba(255,255,255,.12);color:#fff;" +
      "font-size:16px;cursor:pointer;";
    x.onclick = close;
    bar.appendChild(x);
    frame = document.createElement("iframe");
    frame.title = "Virtual try-on";
    frame.setAttribute("allow", "camera; clipboard-write; web-share");
    frame.style.cssText = "flex:1;width:100%;border:0;background:#fff;";
    box.appendChild(bar);
    box.appendChild(frame);
    wrap.appendChild(box);
    wrap.addEventListener("click", function (e) { if (e.target === wrap) close(); });
    document.body.appendChild(wrap);
  }

  function open(wig) {
    if (!wrap) build();
    var q = "?embed=1" + (wig && wig !== "all" ? "&wig=" + encodeURIComponent(wig) : "");
    frame.src = APP + "/" + q; // fresh each time: every visit starts clean
    lastFocus = document.activeElement;
    scrollY = window.scrollY;
    document.documentElement.style.overflow = "hidden";
    wrap.style.display = "flex";
    frame.focus();
  }

  function close() {
    if (!wrap) return;
    wrap.style.display = "none";
    frame.src = "about:blank";
    document.documentElement.style.overflow = "";
    window.scrollTo(0, scrollY);
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  document.addEventListener("click", function (e) {
    var t = e.target.closest && e.target.closest("[data-bbh-tryon]");
    if (!t) return;
    e.preventDefault();
    open(t.getAttribute("data-bbh-tryon") || handleFromPage());
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && wrap && wrap.style.display === "flex") close(); });
  window.addEventListener("message", function (e) {
    if (e.origin !== APP || !e.data || e.data.source !== "bbh-tryon") return;
    if (e.data.type === "close") close();
    if (e.data.type === "open" && /^https?:\/\//.test(e.data.url || "")) { close(); location.href = e.data.url; }
  });
  window.BBHTryOn = { open: open, close: close };
})();
