# Atlas GUI

Web-based model console for the generated Atlas FastAPI inference service. The Atlas CLI is kept independent and unchanged.

## Run locally

Start both the GUI and model service from one terminal:

```powershell
cd web
npm install
npm run dev
```

This starts the FastAPI model service at `http://127.0.0.1:8000` and the GUI at `http://localhost:5173`. Vite forwards `/api/*` to the service automatically. Press `Ctrl+C` once to stop both.

The service model artifact requires `scikit-learn==1.8.0`; this is pinned in `service/requirements.txt`.

To run only the interface, use `npm run gui`; to run only the service, use `npm run service`. For a separately deployed GUI, set `VITE_ATLAS_API_URL` to the service's public base URL (see `.env.example`).

## Connected endpoints

- `GET /health` — service and model availability
- `GET /info` — model metadata
- `POST /predict` — one heart-disease feature record
- `POST /predict_batch` — an array of records
