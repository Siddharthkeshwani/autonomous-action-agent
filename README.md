# OmniOps

OmniOps answers questions about demo orders, UPI transactions, and support policies through a browser or a LiveKit voice session. It can turn query results into charts and asks for approval before submitting a refund, ticket, or email.

## How it works

```text
Browser -> FastAPI gateway -> LangGraph -> MCP tools
Voice   -> LiveKit worker --------^           |
                                             +-> Supabase (read-only SQL)
                                             +-> Qdrant (policy search)
                                             +-> Docker or E2B (Python charts)
                                             +-> Business API -> Supabase (approved writes)
```

Supabase also holds conversation checkpoints. Redis stores recent voice timings. On Render, one Uvicorn worker runs the gateway, the embedded MCP tools, and the key-protected Business API mounted under `/business`. The voice worker runs separately.

## Run locally

Use Python 3.11 or newer, Node.js 24 with npm, and Docker Desktop in Linux-container mode. Run these PowerShell commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
docker build -f Dockerfile.sandbox -t autonomous-action-sandbox:phase4 .
```

Fill in `.env` using the table below. Keep `SANDBOX_BACKEND=docker`, `BUSINESS_API_URL=http://localhost:8001`, and `MCP_SERVER_URL=http://localhost:8002/sse`. For a new database, use this setup command to create the schema, read-only role, and synthetic rows. Use the same read-only password in `SUPABASE_READONLY_URL`.

```powershell
$env:AGENT_READONLY_PASSWORD = Read-Host 'Alphanumeric password for agent_readonly'
.\.venv\Scripts\python.exe -m dotenv run -- .\.venv\Scripts\python.exe -m data.seed_data
Remove-Item Env:AGENT_READONLY_PASSWORD
```

Start one terminal per line. Run the last line only for voice.

```powershell
.\.venv\Scripts\python.exe -m services.business_api.main
.\.venv\Scripts\python.exe -m services.mcp_server.server
.\.venv\Scripts\python.exe -m src.api.main
.\.venv\Scripts\python.exe -m src.voice.worker dev
```

In another terminal, run the frontend and open [localhost:3000](http://localhost:3000).

```powershell
Set-Location frontend
npm ci
npm run dev
```

Check the gateway with `curl.exe --fail http://localhost:8000/health`. A ready response reports PostgreSQL, Redis, and MCP as healthy.

## Backend settings

| Name | Value |
| --- | --- |
| `SUPABASE_DB_URL` | PostgreSQL DSN for schema setup, checkpoints, and Business API writes. For Supabase, use the session pooler DSN with SSL. |
| `SUPABASE_READONLY_URL` | Separate DSN for `agent_readonly`. With the session pooler, use `agent_readonly.<project-ref>` as the username. |
| `UPSTASH_REDIS_URL` | Redis connection URI. Use the `rediss://` URI for Upstash. |
| `BUSINESS_API_KEY` | A private random key shared by the tools and the Business API. |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | Keys for routing and model calls. The model names in `.env.example` are optional overrides. |
| `QDRANT_URL`, `QDRANT_API_KEY` | For policy search. Load the files in `data/policies/` into a `policies` collection with 768-dimensional Gemini embeddings and `source` and `text` fields in each point's payload. |
| `SANDBOX_BACKEND`, `E2B_API_KEY` | Use `docker` locally. The Render image defaults to `e2b`, which needs an E2B key. |
| `FRONTEND_URL` | The browser origin allowed by Render, such as `https://your-project.vercel.app`. |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Required on the gateway and the separate worker if using voice. Put `DEEPGRAM_API_KEY` on the worker. |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Optional tracing. Set tracing to `true` and supply a key and project name to use it. |

Supabase connection options are documented in the [Supabase database guide](https://supabase.com/docs/guides/database/connecting-to-postgres).

## Phase 9 results

| Check | Passed |
| --- | ---: |
| Pytest | 56 / 56 |
| Offline evaluation | 40 / 40 |
| SQL cases | 12 / 12 |
| Retry cases | 8 / 8 |
| Tool cases | 8 / 8 |
| Approval cases | 6 / 6 |
| Security cases | 6 / 6 |

The offline evaluation uses scripted model responses and mocked external backends. To reproduce the checks, point `TEST_DATABASE_URL` at a disposable PostgreSQL database with `data/schema.sql` applied and build the sandbox image as above. The CI workflow provisions this database automatically.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m evaluation.run_eval --offline
```

The evaluation writes `evaluation/results.json`.

## Deploy to Render and Vercel

1. Prepare a Supabase project with the setup command above and create an Upstash Redis database. Push the repository to GitHub.
2. In Render, create one **Web Service** from the repository. Choose the **Docker** runtime and **Free** instance type. Leave the root directory at the repository root, set the Dockerfile path to `./Dockerfile.api`, and set the health check path to `/health`. [Render web service setup](https://render.com/docs/web-services)
3. Set `SUPABASE_DB_URL`, `SUPABASE_READONLY_URL`, `UPSTASH_REDIS_URL`, `BUSINESS_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `SANDBOX_BACKEND=e2b`, and `E2B_API_KEY` on Render. Add policy search, voice, or tracing settings from the table as needed. The image sets up internal Business API and MCP routing by default. Deploy and open `https://your-service.onrender.com/health`.
4. In Vercel, import the GitHub repository under the **Hobby** plan. Set **Root Directory** to `frontend` and **Framework Preset** to **Next.js**. Add `NEXT_PUBLIC_API_URL=https://your-service.onrender.com` for Production and deploy. [Vercel project setup](https://vercel.com/docs/monorepos)
5. Set `FRONTEND_URL` in Render to the Vercel production origin, including `https://` and without a path. Redeploy Render, open the Vercel URL, ask a question about the demo data, and confirm that a refund waits for approval.
6. On GitHub, go to **Settings → Secrets and variables → Actions → Variables**. Add `RENDER_HEALTH_URL` with the full Render `/health` URL. Run **Supabase keepalive** once from the Actions tab. The schedule runs on the default branch three times daily, and each request runs `SELECT 1` against Supabase via the gateway. GitHub can delay schedules and Supabase can still pause a low-activity project. [GitHub schedule rules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule) · [Supabase pausing rules](https://supabase.com/docs/guides/platform/free-project-pausing)

## Limits

- Render Free can take about a minute to start after idling. This setup uses one web container, and provider usage is subject to each account's quota. [Render free tier](https://render.com/docs/free)
- Run the LiveKit worker separately for a voice demo. The Render image includes the token endpoint and leaves out the worker and its speech packages.
- Python runs are capped at 30 seconds and repair attempts at three. Use synthetic data for the public demo. The gateway has no user sign-in.
