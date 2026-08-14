// x-coach on Azure Container Apps.
//
// Mirrors docker-compose.yml one-for-one: a `backend` app (FastAPI + the pose/rules/
// retrieval pipeline) behind an INTERNAL ingress, and a `frontend` app (nginx serving the
// built SPA) with an EXTERNAL ingress that proxies /api to it. Postgres/auth stays on
// Supabase and user uploads stay on Cloudflare R2, so neither has a resource here.
//
// Deploy in two passes, because the first one has to create the environment and the data
// share that the container apps mount:
//
//   az deployment group create -g <rg> -f infra/main.bicep -p deployApps=false
//   (images are built by .github/workflows/build-images.yml and pushed to GHCR)
//   az deployment group create -g <rg> -f infra/main.bicep -p @infra/main.parameters.json
//
// See docs/azure-deployment.md for the full walkthrough, including the 240s ingress
// timeout, the Azure Files data share, and custom domains + managed TLS certificates.

targetScope = 'resourceGroup'

// ---------------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------------

@description('Region for every resource. East Asia or Japan East for Taiwan/LINE traffic; keep it close to the Supabase project.')
param location string = resourceGroup().location

@description('Prefix for generated resource names. Lowercase alphanumeric.')
@minLength(3)
@maxLength(11)
param namePrefix string = 'xcoach'

@description('Set false on the first pass to create the registry and environment only; the images do not exist yet.')
param deployApps bool = true

@description('Backend image reference, e.g. ghcr.io/lucas6028/x-coach-backend:<sha>. Required when deployApps is true.')
param backendImage string = ''

@description('Frontend image reference, e.g. ghcr.io/lucas6028/x-coach-frontend:<sha>. Required when deployApps is true.')
param frontendImage string = ''

@description('Public hostname the SPA is served on, e.g. app.example.com. Left blank the app is reachable on its generated *.azurecontainerapps.io FQDN only. The certificate is bound separately -- see docs/azure-deployment.md.')
param customDomain string = ''

// --- Backend sizing --------------------------------------------------------------------
// A single analysis is CPU/RAM-heavy (MediaPipe + rules + RAG) and the in-process cap is
// PER PROCESS (backend/app/config.py), so scale with replicas rather than workers.
// Consumption tops out at 4 vCPU / 8 GiB per replica; memory must be 2 GiB per vCPU.

@description('vCPU per backend replica.')
@allowed([1, 2, 4])
param backendCpu int = 2

@description('Concurrent in-process analyses per backend replica (XCOACH_MAX_CONCURRENT_ANALYSES).')
param maxConcurrentAnalyses int = 2

@description('Backend replica floor. 1 keeps a warm replica and bills it at the idle rate around the clock; 0 costs nothing while nobody is using the app but puts a cold start -- MediaPipe, the KG and the vector DB off the SMB share -- on the first request, which the LINE webhook may time out on. See the cost ladder in docs/azure-deployment.md.')
@minValue(0)
param backendMinReplicas int = 1

@description('Backend replica ceiling.')
param backendMaxReplicas int = 3

@description('Frontend replica floor. nginx starts in a second or two, so 0 is cheap and barely noticeable -- but it is the public entry point, so the first visitor after an idle period waits for it.')
@minValue(0)
param frontendMinReplicas int = 1

// --- Application configuration ---------------------------------------------------------
// Every one of these is optional: the backend boots without them and reports each
// unconfigured subsystem on GET /api/health rather than failing. See .env.example.

@description('Supabase project URL, e.g. https://abcdefgh.supabase.co')
param supabaseUrl string = ''

@secure()
@description('Supabase anon/public key. The backend acts as the user; RLS is the backstop.')
param supabaseAnonKey string = ''

@secure()
@description('Supabase service_role key. Used ONLY by the LINE LIFF bridge to mint a sign-in link.')
param supabaseServiceRoleKey string = ''

@description('LINE Login channel id (LIFF bridge).')
param lineChannelId string = ''

@secure()
@description('LINE Messaging API channel secret (webhook signature verification).')
param lineMessagingChannelSecret string = ''

@secure()
@description('LINE Messaging API channel access token (reply API).')
param lineMessagingAccessToken string = ''

@description('LIFF app id, used to build the "open x-coach" link in bot replies.')
param lineLiffId string = ''

@secure()
@description('OpenAI-compatible API key for conversational coaching. Blank keeps /api/chat disabled (503).')
param llmApiKey string = ''

@description('Comma-separated model ids offered in Settings; the first is the default.')
param llmModels string = ''

@description('OpenAI-compatible base URL. Blank uses the backend default (OpenRouter).')
param llmBaseUrl string = ''

// --- Cloudflare R2 ---------------------------------------------------------------------
// REQUIRED in this topology. /app/data is mounted read-only from Azure Files, so the
// LocalObjectStore fallback has nowhere to write -- and on ephemeral replicas its uploads
// would vanish on restart anyway. All four must be set together: the backend treats a
// partial set as "unconfigured" and falls back SILENTLY.

param r2AccountId string = ''

@description('R2 access key id.')
param r2AccessKeyId string = ''

@secure()
@description('R2 secret access key.')
param r2SecretAccessKey string = ''

@description('R2 bucket name.')
param r2Bucket string = ''

// ---------------------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------------------

var suffix = uniqueString(resourceGroup().id)
// Storage account names are capped at 24 characters and must be lowercase alphanumeric,
// which an 11-character prefix plus the 13-character hash would overrun.
var storageName = toLower(take('${namePrefix}st${suffix}', 24))
var envName = '${namePrefix}-env'
var lawName = '${namePrefix}-logs'
var backendAppName = '${namePrefix}-backend'
var frontendAppName = '${namePrefix}-frontend'
var dataShareName = 'data'
var dataStorageName = 'xcoach-data'

// The frontend's nginx resolves this at config load. Built from the environment's DNS
// suffix rather than from the backend resource, so the frontend can deploy independently.
var backendOrigin = 'http://${backendAppName}.internal.${containerAppsEnv.properties.defaultDomain}'

// Requests reach the API same-origin through nginx, so CORS normally never applies. This
// keeps a direct browser call to the backend FQDN working from the SPA's own origin.
var corsOrigins = empty(customDomain) ? '' : 'https://${customDomain}'

// ---------------------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------------------
//
// There is no registry resource. The images live in GHCR as public packages, built by
// .github/workflows/build-images.yml, and a public registry needs no `registries` block and
// no pull identity at all. An Azure Container Registry was the original design, but ACR
// Tasks -- the only way to build an image without a local Docker daemon -- is disabled on
// subscriptions spending student or trial credit, which left the registry as $5/month of
// storage for images that had to be built on GitHub anyway.

resource law 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: lawName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ---------------------------------------------------------------------------------------
// Data share
// ---------------------------------------------------------------------------------------
//
// Stands in for compose's `./data:/app/data:ro`: the KG graphml, the RAG vector DB and the
// pre-processed demo library. Read-only, exactly as in compose -- the backend only ever
// reads under data/, and the one path it writes (data/runtime/objects, the LocalObjectStore
// fallback) is unused once R2 is configured.

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource dataShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: dataShareName
  properties: {
    // Generous: the labeled demo dataset is the bulk of it. Billed on what is used.
    shareQuota: 100
    enabledProtocols: 'SMB'
  }
}

// ---------------------------------------------------------------------------------------
// Container Apps environment
// ---------------------------------------------------------------------------------------

resource containerAppsEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: law.properties.customerId
        sharedKey: law.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
}

// NOTE: the ingress request timeout is NOT set here. Envoy cancels any request past 240s,
// which a cold analysis can exceed; raising it needs premium ingress, enabled out of band:
//
//   az containerapp env update -n <env> -g <rg> --enable-premium-ingress \
//        --request-idle-timeout 15
//
// See docs/azure-deployment.md for why the durable fix is an async job instead.

resource envDataStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: containerAppsEnv
  name: dataStorageName
  properties: {
    azureFile: {
      accountName: storage.name
      accountKey: storage.listKeys().keys[0].value
      shareName: dataShare.name
      accessMode: 'ReadOnly'
    }
  }
}

// ---------------------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------------------

resource backend 'Microsoft.App/containerApps@2024-03-01' = if (deployApps) {
  name: backendAppName
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnv.id
    workloadProfileName: 'Consumption'
    configuration: {
      // Internal only. Two reasons: nothing outside the environment needs the API, and
      // when the R2 vars are unset the backend exposes an UNAUTHENTICATED
      // GET /api/local-object/{key} that must never face the internet.
      ingress: {
        external: false
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      // No `registries`: the GHCR packages are public, so the platform pulls anonymously.
      secrets: [
        { name: 'supabase-anon-key', value: supabaseAnonKey }
        { name: 'supabase-service-role-key', value: supabaseServiceRoleKey }
        { name: 'line-messaging-channel-secret', value: lineMessagingChannelSecret }
        { name: 'line-messaging-access-token', value: lineMessagingAccessToken }
        { name: 'llm-api-key', value: llmApiKey }
        { name: 'r2-secret-access-key', value: r2SecretAccessKey }
      ]
    }
    template: {
      containers: [
        {
          name: 'backend'
          image: backendImage
          resources: {
            cpu: json(string(backendCpu))
            memory: '${backendCpu * 2}Gi'
          }
          volumeMounts: [
            {
              volumeName: 'data'
              mountPath: '/app/data'
            }
          ]
          env: [
            { name: 'SUPABASE_URL', value: supabaseUrl }
            { name: 'SUPABASE_ANON_KEY', secretRef: 'supabase-anon-key' }
            { name: 'SUPABASE_SERVICE_ROLE_KEY', secretRef: 'supabase-service-role-key' }
            { name: 'LINE_CHANNEL_ID', value: lineChannelId }
            { name: 'LINE_MESSAGING_CHANNEL_SECRET', secretRef: 'line-messaging-channel-secret' }
            { name: 'LINE_MESSAGING_ACCESS_TOKEN', secretRef: 'line-messaging-access-token' }
            { name: 'LINE_LIFF_ID', value: lineLiffId }
            { name: 'LLM_API_KEY', secretRef: 'llm-api-key' }
            { name: 'LLM_MODELS', value: llmModels }
            { name: 'LLM_BASE_URL', value: llmBaseUrl }
            { name: 'R2_ACCOUNT_ID', value: r2AccountId }
            { name: 'R2_ACCESS_KEY_ID', value: r2AccessKeyId }
            { name: 'R2_SECRET_ACCESS_KEY', secretRef: 'r2-secret-access-key' }
            { name: 'R2_BUCKET', value: r2Bucket }
            { name: 'XCOACH_CORS_ORIGINS', value: corsOrigins }
            { name: 'XCOACH_MAX_CONCURRENT_ANALYSES', value: string(maxConcurrentAnalyses) }
          ]
          probes: [
            {
              // Generous failure threshold: the first request pays for loading MediaPipe,
              // the knowledge graph and the RAG vector DB off the Azure Files share.
              type: 'Startup'
              httpGet: {
                path: '/api/health'
                port: 8000
              }
              periodSeconds: 10
              failureThreshold: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/api/health'
                port: 8000
              }
              periodSeconds: 15
              failureThreshold: 3
            }
            {
              // Long period and a high threshold on purpose: a replica saturated with
              // analyses is slow to answer, not dead. Restarting it would discard
              // in-flight work that the user is waiting on.
              type: 'Liveness'
              httpGet: {
                path: '/api/health'
                port: 8000
              }
              periodSeconds: 60
              failureThreshold: 5
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'data'
          storageType: 'AzureFile'
          storageName: envDataStorage.name
        }
      ]
      scale: {
        minReplicas: backendMinReplicas
        maxReplicas: backendMaxReplicas
        rules: [
          {
            // Matched to the in-process semaphore rather than left at the 10-concurrency
            // default: past this the requests queue on the semaphore, not on CPU, so the
            // replica looks healthy while every upload waits.
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: string(maxConcurrentAnalyses)
              }
            }
          }
        ]
      }
    }
  }
}

// ---------------------------------------------------------------------------------------
// Frontend
// ---------------------------------------------------------------------------------------

resource frontend 'Microsoft.App/containerApps@2024-03-01' = if (deployApps) {
  name: frontendAppName
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnv.id
    workloadProfileName: 'Consumption'
    configuration: {
      ingress: {
        external: true
        targetPort: 80
        transport: 'http'
        // The generated *.azurecontainerapps.io FQDN is HTTPS already; this redirects the
        // plain-HTTP port so the LINE webhook and LIFF never see a downgrade.
        allowInsecure: false
      }
      // No `registries`: see the backend app above.
    }
    template: {
      containers: [
        {
          name: 'frontend'
          image: frontendImage
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: [
            // Substituted into the nginx config at container start (see
            // frontend/nginx.conf.template). Defaults to the compose service name.
            {
              name: 'BACKEND_ORIGIN'
              value: backendOrigin
            }
          ]
          probes: [
            {
              type: 'Readiness'
              httpGet: {
                path: '/'
                port: 80
              }
              periodSeconds: 15
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        // Static assets only; one replica saturates long before the backend does.
        minReplicas: frontendMinReplicas
        maxReplicas: 3
      }
    }
  }
}

// ---------------------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------------------

output environmentName string = containerAppsEnv.name
output storageAccountName string = storage.name
output dataShareName string = dataShare.name
output backendInternalOrigin string = backendOrigin
output frontendFqdn string = frontend.?properties.configuration.ingress.fqdn ?? ''

@description('Point an apex A record at this, or a subdomain CNAME at the frontend FQDN. Needed for the free managed certificate.')
output environmentStaticIp string = containerAppsEnv.properties.staticIp
