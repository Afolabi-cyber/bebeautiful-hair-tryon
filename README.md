# Be Beautiful Hair · Virtual Try-On

A mobile-first web app where a customer enters their name and phone, takes a selfie, picks one of your wigs, and sees themselves wearing it, with their face left exactly as it is. An admin dashboard at `/admin` shows who took each photo, which wigs they tried, and when.

- Frontend: React + Tailwind (Vite), in `frontend/`
- Backend: FastAPI + OpenAI image editing + MediaPipe face landmarks, in `server/`
- Storage: SQLite plus photo files in `server/data/`

## Customer flow

Built for the public: each visitor uses the link on their own phone and goes through one straight path, with a progress bar at the top.

1. Details: name, phone, consent tick.
2. Photo: front camera with an oval guide and a 3-second timer, or upload. Then retake or use it.
3. Style: pick a wig from the grid.
4. Result: before/after slider, save, share, shop this look, try another style, or start again.

Every visit starts fresh (no old photos or results come back). Only the name and phone are remembered on that phone so the form is prefilled next time. The phone's back button moves one step back instead of leaving the site.

## The result is the customer's own photo

The customer's clothes, background, body and face are never taken from the AI image:

1. The AI may only edit a fenced area: the wig's outline (measured from its product photo) placed on the customer's head, plus a margin. Everything else is marked "do not edit".
2. The final picture is built from the customer's original photo. Outside the fence nothing from the AI is used. Inside it, the hair/clothes segmentation model keeps only the new hair, so the clothes under and between the braids are the customer's own too.
3. Their old hair is filled in from the background where the wig doesn't cover it.
4. The background is replaced with a clean, light studio backdrop (the "before" photo keeps the original). Set `RESULT_BACKGROUND` to `studio` (default), `white` or `original`.

The segmentation model is required for this. If it can't load, try-ons pause with a friendly message, the admin dashboard shows a red banner, and the server retries every 2 minutes. They never return a result with changed clothes. The console shows the status at startup:

```
MODEL CHECK  face landmarks ............ OK
MODEL CHECK  hair/clothes model ........ OK
```

If the hair/clothes model shows FAILED (for example a network that blocks the download), download it manually from
https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
and save it as `server/models/selfie_multiclass_256x256.tflite`, then restart.

## Managing wigs (admin)

Admin, Wigs tab:
- Add wig: upload a front photo of the wig alone on a plain light background, then add a name, optional price, a description for the AI (length, braid or curl type, colours, parting) and an optional shop link. Tick "fringe" for wigs with bangs.
- Edit: change any field or replace the photo.
- Hide or show: take a wig down without deleting it.
- Arrows: change the order customers see.
- Delete: removes the wig and its photo files. Past try-ons stay in the history.

If a photo's wig can't be made out against its background, the admin gets a tip to use a plainer photo.

## How the face is preserved

1. Edit mask: OpenAI is told the face area must not be edited. Only the hair, hairline and surroundings can change.
2. High input fidelity: with `gpt-image-1.5`, `input_fidelity="high"` is sent, and the selfie goes first so it gets the most detail.
3. Face lock: after the result comes back, the customer's original face is aligned and blended back in, so eyes, nose, mouth, face shape and skin texture are their own. The face then takes on the lighting and white balance of the new photo, so it doesn't look pasted on (for example when the selfie was taken in grey or yellow indoor light). That shift is capped so skin tone stays the customer's.
4. Retouch: skin smoothing that keeps natural texture, even skin tone (blotches and redness), no oily shine, a gentle lift on the face and under the eyes, and slightly crisper eyes, brows and lips. Nothing is moved or reshaped, so identity is untouched: on a real test photo the ArcFace face-recognition model scored the retouched face 0.95 similar to the original at the default strength (same person is above 0.5). Controlled by `FACE_ENHANCE` (0 off, 0.75 default, 1 strongest).
5. Clothes: the AI sometimes repaints the outfit (often dark). After each result, a segmentation model finds the customer's clothes in the original photo and restores those exact pixels, except where the new hair falls over them, including the small gaps between braids. Controlled by `KEEP_CLOTHES` (on by default). The segmentation model (about 16 MB) downloads into `server/models/` on first use.
6. Hair in front of the face: where the AI draws braids falling over the cheeks or temples, those strands are kept instead of being covered by the pasted face.

## How the wig stays exact

Each wig is sent to OpenAI as a reference image, along with a written description from `catalog.json` (braid size, curl pattern, colours and ombre, parting, length). The prompt tells the model to copy that wig and not invent a different style.

## Admin dashboard (`/admin`)

- Sign in with `ADMIN_PASSWORD` from `.env`.
- Stats: customers, try-ons (total and today), success rate, average wait, a 14-day chart, most-tried wigs.
- Customer list with selfie, name, phone, time and wigs tried. Search by name or phone.
- Tap a customer to see their original photo, phone (tap to call), device, time, consent, and every try-on. Tap a result to flip between before and after; double-tap to enlarge.
- Export everything to CSV.
- Delete a customer, which removes their photos from disk.

## Deploy to Render (paid, always on)

The repo includes `Dockerfile` and `render.yaml`. The blueprint creates:
- a web service on the `1c-2g` plan (Standard: 1 CPU, 2 GB RAM). Paid instances don't spin down, so there are no cold starts. The 512 MB plan is too small for the image processing.
- a 5 GB persistent disk at `/var/data` for the database, customer photos, uploaded wigs and admin edits. On first boot the six starting wigs are copied there.
- region Frankfurt (closest to Nigeria), health check `/api/health`, auto-deploy on every push.

Both AI models are downloaded while the image builds, so a deploy fails loudly (and the previous version stays live) instead of starting without them.

Steps:
1. Push this folder to a private GitHub repository (`.env`, `server/data` and `node_modules` are git-ignored).
2. Render dashboard: New, then Blueprint, then connect the repository.
3. Enter `OPENAI_API_KEY` and `ADMIN_PASSWORD` when asked; `SECRET_KEY` is generated. Apply.
4. In the deploy logs, check `MODEL CHECK face landmarks ... OK` and `MODEL CHECK hair/clothes model ... OK`.
5. Open the `onrender.com` link on a phone (https is automatic, so the camera works). Admin is at `/admin`.
6. Optional: Settings, Custom Domains, add for example `tryon.bebeautifulhair.com` and the CNAME record Render shows.

Costs on Render: $25 per month for the instance plus $0.25 per GB per month for the disk, plus OpenAI usage per try-on.

## Run it

Requirements: Python 3.10+ and Node 20+.

```
cd hair-tryon
cp .env.example .env           # add your OpenAI key and an admin password

cd frontend
npm install
npm run build                  # creates frontend/dist, which FastAPI serves

cd ../server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Customer app: http://localhost:8000. Admin: http://localhost:8000/admin.

On the first photo, the server downloads the MediaPipe face model (about 4 MB) into `server/models/`.

For frontend development with hot reload, keep uvicorn running and also run `npm run dev` in `frontend/`. Vite forwards `/api` and `/wigs` to port 8000.

Phones only allow the camera on https. Deploy behind https (Render, Railway, or a VM with Nginx and a certificate), or use `ngrok http 8000` for testing.

## Adding or editing wigs

1. Put the photo in `wigs/` (front view, plain background, like the current ones).
2. Make a thumbnail for the grid, around 480 px wide, in `wigs/thumbs/` (webp or jpg).
3. Add an entry in `catalog.json`:

```json
{
  "id": "bone-straight-30",
  "name": "Bone Straight 30\"",
  "price": "₦85,000",
  "image": "wigs/bone-straight-30.png",
  "thumb": "wigs/thumbs/bone-straight-30.webp",
  "description": "30-inch bone straight wig, jet black, middle part, past the waist",
  "bangs": false,
  "buyUrl": "https://yourstore.com/bone-straight-30"
}
```

- `price` and `buyUrl` are optional. When set, the price shows on the card and a Shop button appears on the result.
- Set `bangs` to `true` for fringe styles.
- Brand name, tagline and accent colour are at the top of `catalog.json`.

## Branding

- Logo: `frontend/public/logo.png`. The current file is 100 x 100 px, which looks soft on high-resolution phones. Replace it with a 512 x 512 px or larger PNG of the same logo (same file name), then run `npm run build`.
- Fonts and colours are taken from bebeautifulhair.com itself: Reem Kufi for text and headings, Young Serif for headlines like "Own your look!", white and #222 with light grey panels, pink #ED008C as the main accent and blue #408BB3 for the "Loved By Queens Everywhere" line. The fonts are bundled into the app (no Google Fonts call), and the colours live in `frontend/src/index.css`.
- Saved and shared photos get a watermark: "Own your look!", bebeautifulhair.com and the logo.
- The Shop button goes to the Braids collection on bebeautifulhair.com. Put each wig's exact product URL in `buyUrl` in `catalog.json` when you have them.

## Settings (.env)

- `OPENAI_IMAGE_MODEL`: `gpt-image-1.5` (default) or `gpt-image-2`
- `OPENAI_IMAGE_QUALITY`: `low`, `medium` or `high`. Use `low` while testing to save cost.
- `RESULT_BACKGROUND`: `studio`, `white` or `original`
- `ALLOW_WITHOUT_HAIR_MODEL`: keep `off` so clothes are always protected
- `FACE_ENHANCE`: 0 to 1, how much retouching the face gets (default 0.75)
- `MAX_TRYONS_PER_PERSON`: cap on AI generations per photo session (default 10)
- `PHOTOS_PER_IP_PER_HOUR`, `TRYONS_PER_IP_PER_HOUR`: public limits per phone or network (defaults 15 and 30). Raise them for an in-store event where many people share one Wi-Fi.
- `ADMIN_PASSWORD`, `SECRET_KEY`: admin sign-in

## Privacy

The app stores each customer's name, phone, selfie and try-on images in `server/data/`. Customers must tick a consent box before taking a photo. Back up that folder, keep it private, and delete customers from the admin page when asked.
