"""Download both models at build time (Docker/Render) so nothing is fetched while customers wait."""
import faces

for url, path in ((faces.MODEL_URL, faces.MODEL_PATH), (faces.SEG_URL, faces.SEG_PATH)):
    faces.ensure_model(url, path)
    print("model ready:", path)
