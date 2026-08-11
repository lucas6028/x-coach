# Deploying x-coach to Azure

Two Container Apps in one environment, mirroring `docker-compose.yml`: `backend` (FastAPI
plus the pose/rules/retrieval pipeline) behind an **internal** ingress, and `frontend`
(nginx serving the built SPA) with an **external** ingress that proxies `/api` to it.
Postgres/auth stays on Supabase and user uploads stay on Cloudflare R2, so neither has an
Azure resource. Infrastructure lives in `infra/main.bicep`.

Read `docs/docker.md` first — everything about the two images (what is in them, why `data/`
is mounted rather than baked, why the `VITE_*` vars are build args) carries over unchanged.

## Where each part goes

| Part of the project | Azure service | Notes |
| --- | --- | --- |
| `frontend/` (nginx + built SPA) | Container Apps | Keeps the `/api` same-origin proxy, so SSE and 256 MB uploads need no rework |
| `backend/` (FastAPI + MediaPipe + ffmpeg) | Container Apps, Consumption profile | CPU/RAM-heavy, single worker, needs a raised request timeout |
| Both images | Azure Container Registry (Basic) | Pulled with a user-assigned managed identity; the admin account stays off |
| Postgres, auth, RLS | **Supabase, unchanged** | No Azure resource. Moving it means rewriting auth, RLS and `supabase-py` |
| Uploads (video, pose JSON, thumbnail) | **Cloudflare R2, unchanged** | `services/storage.py` speaks the S3 API; Azure Blob has no S3-compatible endpoint, so switching needs a new store implementation |
| `data/` — KG graphml, RAG vector DB, demo library | Azure Files share, mounted read-only at `/app/data` | Same shape as compose's `./data:/app/data:ro` |
| Secrets | Container Apps secrets (or Key Vault references) | Never in the parameters file — see [Secrets](#secrets) |
| Logs and traces | Log Analytics (wired by the template) + Application Insights | |
| Custom domain and TLS | Container Apps custom domain + free managed certificate | Frontend only; the backend is internal |
| Research pipelines (`src/rehab24`, `src/video`, torch/VideoMAE, Gemini KG extraction) | **Not deployed** | If they ever need to run in the cloud, Container Apps *Jobs* or Azure ML — not this app. `requirements-docker.txt` deliberately excludes their dependencies |

**Region.** East Asia is closest to Taiwan; Japan East has the wider service catalogue.
Either works — pick whichever is nearer the Supabase project, because every request pays
that round trip.

## Topology

```
                       ┌─ Front Door (optional: WAF, global cache)
                       │
   [browser / LIFF] ───┴──> frontend app   external ingress :80  (nginx)
                                 │   location /api/ -> proxy_pass ${BACKEND_ORIGIN}
                                 ▼
                            backend app    INTERNAL ingress :8000 (uvicorn)
                                 │
             ┌───────────────────┼────────────────────┐
             ▼                   ▼                    ▼
        Supabase            Cloudflare R2     Azure Files: /app/data (ro)
       (external)          (external)         KG + RAG + demo library
```

The backend's ingress is internal for two reasons: nothing outside the environment needs
the API, and when the `R2_*` variables are unset the backend exposes an **unauthenticated**
`GET /api/local-object/{key}` that must never face the internet.

The one thing that does arrive from outside is the LINE Messaging webhook. It hits the
frontend's public hostname and nginx forwards it like any other `/api` route — no separate
public endpoint, and no reason to make the backend external.

## Why Container Apps for the frontend, and not Static Web Apps or Vercel

The frontend image is not a static site: `nginx.conf.template` is also the reverse proxy,
and it does three things a CDN-first platform will not.

- **Long requests.** A cold analysis runs for minutes; the config allows 900 s.
  Azure Static Web Apps and Vercel both cap proxied requests well below that (Vercel's
  external-rewrite ceiling is 120 s and is not configurable).
- **Large uploads.** `client_max_body_size 256m`. Static Web Apps caps a request at 30 MB.
- **SSE.** `/api/chat` needs `proxy_buffering off` to stream tokens; behaviour through a
  third-party proxy layer is at best undocumented.

Avoiding all three means dropping the `/api` proxy and calling the backend cross-origin —
but every call in `frontend/src/api.ts` is a hard-coded relative path (`fetch("/api/...")`,
and `videoFileUrl` returns `/api/video-file/{id}` straight into a `<video src>`). That is a
`VITE_API_BASE_URL` refactor across ~30 call sites, plus CORS, plus a public backend, in
exchange for CDN delivery of one small SPA. Not worth it at this size.

## Three things to get right

### 1. The ingress request timeout is 240 s by default

Container Apps fronts every app with Envoy, which cancels a request at 240 s. A cold
analysis (MediaPipe + rules + RAG) can exceed that, and `nginx.conf.template` asking for
900 s does not change it — the platform cuts first.

Three ways out, in increasing order of doing it properly:

1. **Premium ingress**, which makes the timeout configurable up to an hour. Not in the
   Bicep template because it changes the environment's billing; enable it out of band:

   ```bash
   az containerapp env update -n xcoach-env -g <rg> \
       --enable-premium-ingress --request-idle-timeout 15
   ```

2. **Prefer the client-side pose path.** The browser runs MediaPipe and posts pose JSON
   (`analyze.py` already supports this); the server then only runs rules and retrieval, and
   finishes in seconds. 240 s is ample.
3. **Make analysis asynchronous.** Upload → Storage Queue or Service Bus → a
   KEDA-triggered **Container Apps Job** → the client polls. `analyze.py:22` already calls
   the in-process semaphore a stop-gap "until the Celery/Redis worker queue lands"; Jobs
   are that queue on Azure without running a broker.

Start with (1) or (2). (3) is the right answer once analysis volume is real, and doing it
first is over-engineering.

### 2. R2 is required, not optional

`/app/data` is mounted read-only, so `LocalObjectStore` has nowhere to write — and replicas
are ephemeral, so anything it did write would vanish on the next revision anyway. All four
`R2_*` variables must be set. A single missing or misspelled one falls the app back to the
local store **silently**.

Verify after every deploy:

```bash
curl -s https://<frontend-fqdn>/api/health | grep storage_configured   # must be true
```

and check the startup log for `Object storage: Cloudflare R2 (bucket=...)` at INFO. The
fallback logs a WARNING instead.

If you would rather keep the local store, add a second Azure Files share mounted
read-write at `/app/data/runtime` — but R2 is cheaper, has no egress fee, and already has
the lifecycle rule for `uploads/anon/` described in `.env.example`.

### 3. Scale with replicas, not workers

`XCOACH_MAX_CONCURRENT_ANALYSES` is a **per-process** semaphore (`backend/app/config.py`).
The template therefore keeps one uvicorn worker per replica and sets the HTTP scale rule's
`concurrentRequests` to the same number, so a replica scales out at the point its semaphore
saturates rather than at Envoy's default of 10 — past which requests queue invisibly while
the replica still looks healthy.

`minReplicas` is 1 on purpose. A cold start loads MediaPipe, the knowledge graph and the
RAG vector DB off the file share; scale-to-zero puts all of that on a real user's first
request.

Consumption tops out at 4 vCPU / 8 GiB per replica, with memory fixed at 2 GiB per vCPU.
Beyond that you need a dedicated workload profile.

## Deploying

### First pass: registry and environment

The container apps reference images that do not exist until the registry does, so the first
deployment skips them.

```bash
RG=xcoach-rg
az group create -n $RG -l eastasia

az deployment group create -g $RG -f infra/main.bicep -p deployApps=false
ACR=$(az deployment group show -g $RG -n main --query properties.outputs.acrName.value -o tsv)
```

### Seed the data share

The KG and RAG stores are produced by the pipelines and are gitignored, so they are
uploaded rather than built in the cloud. From the repo root, after the pipelines have run:

```bash
STORAGE=$(az deployment group show -g $RG -n main \
    --query properties.outputs.storageAccountName.value -o tsv)
KEY=$(az storage account keys list -g $RG -n $STORAGE --query '[0].value' -o tsv)

az storage file upload-batch --account-name $STORAGE --account-key $KEY \
    -d data/kg -s data/kg
az storage file upload-batch --account-name $STORAGE --account-key $KEY \
    -d data/rag/vector_db -s data/rag/vector_db
```

The demo library (`data/Fitness-AQA/Squat/Labeled_Dataset/`) is optional and large; upload
it the same way only if you want the instant-demo videos. Without it the API still serves —
`/api/health` reports those stores missing, exactly as a bare `docker run` does.

> Cold-start alternative: if reading the vector DB over SMB ever shows up in startup
> latency, bake `data/kg/` and `data/rag/vector_db/` into the backend image instead. That
> needs a `.dockerignore` change — Docker cannot re-include a path whose parent directory
> is excluded, so the `data` line has to become `data/*` before `!data/kg` will take
> effect. Do this only if measurement justifies it; the KG and vector DB are small, and the
> embedder is hash-based (`src/knowledge/rag_vector_db.py`), so nothing downloads a model.

### Build and push

Both images build in ACR, so no local Docker daemon is needed. The backend's build context
is the **repo root** (it imports `backend.*` and `src.*` by absolute package path); the
frontend's is `frontend/`.

```bash
TAG=$(git rev-parse --short HEAD)

az acr build -r $ACR -t x-coach-backend:$TAG -f backend/Dockerfile .

az acr build -r $ACR -t x-coach-frontend:$TAG -f frontend/Dockerfile frontend \
    --build-arg VITE_SUPABASE_URL=https://xxxx.supabase.co \
    --build-arg VITE_SUPABASE_ANON_KEY=eyJ... \
    --build-arg VITE_LIFF_ID=1234567890-Abcdefgh
```

Vite inlines `VITE_*` at build time, so those are **build args, not runtime env** — a
change to any of them requires a rebuild, not a restart. The anon key is safe to ship; row
access is governed by Postgres RLS.

### Second pass: the apps

```bash
az deployment group create -g $RG -f infra/main.bicep \
    -p @infra/main.parameters.json \
    -p backendImage=$ACR.azurecr.io/x-coach-backend:$TAG \
    -p frontendImage=$ACR.azurecr.io/x-coach-frontend:$TAG \
    -p supabaseAnonKey=$SUPABASE_ANON_KEY \
    -p supabaseServiceRoleKey=$SUPABASE_SERVICE_ROLE_KEY \
    -p llmApiKey=$LLM_API_KEY \
    -p r2SecretAccessKey=$R2_SECRET_ACCESS_KEY \
    -p lineMessagingChannelSecret=$LINE_MESSAGING_CHANNEL_SECRET \
    -p lineMessagingAccessToken=$LINE_MESSAGING_ACCESS_TOKEN
```

The frontend gets `BACKEND_ORIGIN` pointed at the backend's internal FQDN. nginx substitutes
it into the config at container start (`frontend/nginx.conf.template`), so re-pointing the
proxy is an env change, not a rebuild.

Note the `Host` header in that config is set to `$proxy_host`, not `$host`. The internal
ingress routes by Host, so forwarding the browser's hostname makes the environment fail to
find the backend app — a 404 from the platform that looks like a routing bug in the API.
The browser's hostname is preserved on `X-Forwarded-Host`.

### Subsequent deploys

Only the image tag changes:

```bash
az containerapp update -n xcoach-backend  -g $RG --image $ACR.azurecr.io/x-coach-backend:$TAG
az containerapp update -n xcoach-frontend -g $RG --image $ACR.azurecr.io/x-coach-frontend:$TAG
```

In GitHub Actions, authenticate with an OIDC federated credential (`azure/login@v2` with
`client-id`/`tenant-id`/`subscription-id`, no stored secret) and run the same two commands
after the existing `ci.yml` test and coverage gates pass.

## Secrets

The `@secure()` parameters are deliberately absent from `infra/main.parameters.json` so
that file stays committable. Pass them on the command line as above, or — better for
anything long-lived — put them in Key Vault and reference them:

```json
"llmApiKey": {
  "reference": {
    "keyVault": { "id": "/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.KeyVault/vaults/<vault>" },
    "secretName": "llm-api-key"
  }
}
```

`SUPABASE_SERVICE_ROLE_KEY` is the one to be most careful with: it exists only so the LINE
LIFF bridge can mint a sign-in link, never for data access. It must not reach the frontend
image or any build arg.

## Custom domain and TLS

The generated `*.azurecontainerapps.io` FQDN already serves HTTPS, so TLS needs no work
until you want your own hostname. Container Apps then issues a **free managed certificate**
(DigiCert) and renews it automatically — there is no certbot to run.

Only the frontend needs one; the backend is internal.

1. Add the DNS records. A subdomain needs a `CNAME` to the frontend FQDN plus a
   `TXT` at `asuid.<sub>` holding the verification code. An apex domain needs an `A` record
   to the environment's static IP (`environmentStaticIp` in the template's outputs) plus a
   `TXT` at `asuid`.
2. Bind the domain and let Azure issue the certificate:

   ```bash
   az containerapp hostname add -n xcoach-frontend -g $RG --hostname app.example.com
   az containerapp hostname bind -n xcoach-frontend -g $RG --hostname app.example.com \
       --environment xcoach-env --validation-method CNAME
   ```

Two things that break issuance:

- **CAA records.** If the root domain has any `CAA` record, add `0 issue digicert.com` or
  both issuance and renewal fail.
- **Ordering.** The app must already be publicly reachable when the certificate is
  requested, because DigiCert validates over HTTP. Deploy first, then DNS, then bind.

Set `customDomain` in the parameters file once the hostname is live: the template feeds it
to `XCOACH_CORS_ORIGINS`, which matters only for direct cross-origin calls to the API (the
SPA itself is same-origin through nginx) but costs nothing to have right.

Front Door in front of the frontend is optional. Add it for WAF or global caching; it is
not needed for TLS or for a custom domain.

## Observability

Container logs stream to the Log Analytics workspace created by the template:

```bash
az containerapp logs show -n xcoach-backend -g $RG --follow
```

For request-level tracing, add Application Insights and instrument FastAPI with
OpenTelemetry. The health endpoint is the fastest first check either way — `/api/health`
reports `auth_configured`, `chat_configured`, `line_login_configured`,
`storage_configured` and a `stores` map for the labeled-video, detection, KG and RAG
directories, which between them explain most "why is this feature dead in production"
questions.

## What this does not cover

The research pipelines are not part of this deployment and should not be added to these
images. `requirements-docker.txt` is the web subset of `requirements.txt` — adding `torch`
and `transformers` back would grow the backend image by several GB for code the API never
calls. If those pipelines need cloud compute, run them as Container Apps Jobs or on Azure
Machine Learning, against their own image built from the full `requirements.txt`.
