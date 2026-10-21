# Dual LLM System — web frontend

Next.js (App Router) + TypeScript + Tailwind UI for the dual-llm-system API.
It replaced the earlier Streamlit demo: persona switcher, profile panel,
session cost, chat with per-answer routing details and the exact prompts sent.

All browser requests go to `/api/*`, which Next.js rewrites to `${API_URL}/*`.
The typed client and contract types live in `lib/api.ts`.

## Develop

```bash
npm ci
API_URL=http://localhost:8000 npm run dev   # http://localhost:3000
```

Start the FastAPI backend separately (port 8000 by default).

## Build and check

```bash
npm run build       # standalone output in .next/standalone
npm run start       # or: node .next/standalone/server.js
npm run lint
npm run typecheck
```

## Docker

```bash
docker build --build-arg API_URL=http://api:8000 -t dual-llm-web .
docker run -p 3000:3000 dual-llm-web
```

## Environment

| Variable  | Default                 | Notes |
|-----------|-------------------------|-------|
| `API_URL` | `http://localhost:8000` | Backend base URL for the `/api` rewrite. Rewrites are resolved when `next build` runs, so for production/Docker set it at build time (`--build-arg API_URL=...`). In `npm run dev` it is read on startup. |
| `PORT`    | `3000`                  | Port for `next start` / the standalone server. |
