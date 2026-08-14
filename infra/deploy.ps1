<#
.SYNOPSIS
    Deploy x-coach to Azure Container Apps, one stage at a time.

.DESCRIPTION
    The narrative version of this is docs/azure-deployment.md — read it first. This script
    exists because the walkthrough's snippets are bash, and on Windows Git Bash mangles any
    argument that starts with "/" (resource IDs, --scope), so `az` has to be driven from
    PowerShell. It also reads the ~14 configuration values out of .env instead of asking you
    to paste eight secrets onto a command line.

    Stages are separate so a failure costs you one step, not the whole run:

        providers   register the four resource providers (once per subscription)
        infra       resource group + ACR + Log Analytics + storage + environment (no apps)
        data        upload the KG and the RAG vector DB to the Azure Files share
        build       build both images in ACR (no local Docker needed)
        apps        deploy the two container apps
        update      re-point the existing apps at a freshly built tag
        status      print the FQDN and run the post-deploy health checks

.EXAMPLE
    ./infra/deploy.ps1 -Stage providers
    ./infra/deploy.ps1 -Stage infra
    ./infra/deploy.ps1 -Stage data
    ./infra/deploy.ps1 -Stage build
    ./infra/deploy.ps1 -Stage apps
    ./infra/deploy.ps1 -Stage status

.NOTES
    Run from the repository root. Requires `az login` to have been done already.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('providers', 'infra', 'data', 'build', 'apps', 'update', 'status')]
    [string]$Stage,

    [string]$ResourceGroup = 'xcoach-rg',
    [string]$Location = 'eastasia',
    [string]$EnvFile = '.env',

    # Defaults to the short commit SHA, so a deployed revision is traceable to a commit.
    [string]$Tag = ''
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

function Assert-LastExit {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)." }
}

function Read-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) { throw "No $Path — copy .env.example and fill it in." }
    $map = @{}
    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }
        $i = $trimmed.IndexOf('=')
        if ($i -lt 1) { continue }
        $key = $trimmed.Substring(0, $i).Trim()
        $value = $trimmed.Substring($i + 1).Trim()
        # Strip one layer of surrounding quotes, the way python-dotenv does.
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                                      ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $map[$key] = $value
    }
    return $map
}

function Get-DeploymentOutput {
    param([string]$Name)
    $value = az deployment group show -g $ResourceGroup -n main `
        --query "properties.outputs.$Name.value" -o tsv
    Assert-LastExit "Reading deployment output '$Name'"
    if ([string]::IsNullOrWhiteSpace($value)) {
        # frontendFqdn is empty until deployApps=true, so this is also what a `-Stage status`
        # run before `-Stage apps` looks like.
        throw "Deployment output '$Name' is empty. Run the earlier stages first ('infra', then 'apps')."
    }
    return $value.Trim()
}

function Resolve-Tag {
    if ($Tag) { return $Tag }
    $sha = (git rev-parse --short HEAD).Trim()
    Assert-LastExit 'git rev-parse'
    return $sha
}

function Assert-LoggedIn {
    az account show -o none 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'Not logged in. Run "az login" first, then re-run this script.'
    }
    $name = az account show --query name -o tsv
    $id = az account show --query id -o tsv
    Write-Host "Subscription: $name ($id)" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------

function Invoke-Providers {
    Assert-LoggedIn
    # A fresh subscription has none of these registered, and the first deployment fails with
    # MissingSubscriptionRegistration rather than saying which one is missing.
    foreach ($ns in @('Microsoft.App', 'Microsoft.OperationalInsights',
                      'Microsoft.ContainerRegistry', 'Microsoft.Storage')) {
        Write-Host "Registering $ns ..." -ForegroundColor Cyan
        az provider register --namespace $ns --wait
        Assert-LastExit "Registering $ns"
    }
    Write-Host 'All four providers registered.' -ForegroundColor Green
}

function Invoke-Infra {
    Assert-LoggedIn
    az group create -n $ResourceGroup -l $Location -o none
    Assert-LastExit 'az group create'

    # Pass 1 skips the apps: they reference images that do not exist until the registry does.
    az deployment group create -g $ResourceGroup -n main -f infra/main.bicep `
        -p '@infra/main.parameters.json' `
        -p location=$Location `
        -p deployApps=false `
        -o none
    Assert-LastExit 'Pass 1 (deployApps=false)'

    Write-Host "ACR:     $(Get-DeploymentOutput 'acrName')" -ForegroundColor Green
    Write-Host "Storage: $(Get-DeploymentOutput 'storageAccountName')" -ForegroundColor Green
    Write-Host ''
    Write-Host 'Set a budget alert before going further — on a student subscription, ' -NoNewline
    Write-Host 'hitting $0 disables every resource in it.' -ForegroundColor Yellow
}

function Invoke-Data {
    Assert-LoggedIn
    $storage = Get-DeploymentOutput 'storageAccountName'
    $share = Get-DeploymentOutput 'dataShareName'
    $key = az storage account keys list -g $ResourceGroup -n $storage --query '[0].value' -o tsv
    Assert-LastExit 'Reading storage account key'
    $key = $key.Trim()

    if (-not (Test-Path 'data/kg/sports_kg_v3.graphml')) {
        throw 'data/kg/sports_kg_v3.graphml is missing. It is gitignored and pipeline-built — generate it before deploying, or the knowledge endpoints will report their stores absent.'
    }
    if (-not (Test-Path 'data/rag/vector_db')) {
        throw 'data/rag/vector_db is missing. Build the RAG store before deploying.'
    }

    # Only the graph the backend actually opens (backend/app/config.py:KG_GRAPH_FILE) plus the
    # canonical mapping. data/kg/ also holds eight .bak/.pre-*/.post-*-raw snapshots that are
    # pipeline history, not runtime inputs.
    Write-Host 'Uploading the knowledge graph ...' -ForegroundColor Cyan
    foreach ($f in @('sports_kg_v3.graphml', 'exercise_canonical_mapping_v1.json', 'shared_vocab_v1.json')) {
        if (Test-Path "data/kg/$f") {
            az storage file upload --account-name $storage --account-key $key `
                --share-name $share --path "kg/$f" --source "data/kg/$f" -o none
            Assert-LastExit "Uploading kg/$f"
        }
    }

    Write-Host 'Uploading the RAG vector DB ...' -ForegroundColor Cyan
    az storage file upload-batch --account-name $storage --account-key $key `
        --destination $share --destination-path 'rag/vector_db' --source 'data/rag/vector_db' -o none
    Assert-LastExit 'Uploading the vector DB'

    Write-Host "Share '$share' populated; it mounts read-only at /app/data." -ForegroundColor Green
    Write-Host 'The demo video library (data/Fitness-AQA/...) is optional and large — upload it the same way only if you want the pre-processed demos.'
}

function Invoke-Build {
    Assert-LoggedIn
    $env_ = Read-DotEnv $EnvFile
    $acr = Get-DeploymentOutput 'acrName'
    $t = Resolve-Tag
    Write-Host "Building tag $t in $acr ..." -ForegroundColor Cyan

    # Backend context is the REPO ROOT: the app imports backend.* and src.* by absolute path.
    az acr build -r $acr -t "x-coach-backend:$t" -f backend/Dockerfile .
    Assert-LastExit 'Backend image build'

    # VITE_* are inlined by Vite at BUILD time, so they are build args, not runtime env —
    # changing one needs a rebuild, not a restart. The anon key ships in the bundle by
    # design; row access is governed by Postgres RLS. The service-role key must never
    # appear here.
    # -f is relative to the CONTEXT root, not to the working directory, so it is a bare
    # 'Dockerfile' here even though the backend above needs the path from the repo root.
    az acr build -r $acr -t "x-coach-frontend:$t" -f Dockerfile frontend `
        --build-arg "VITE_SUPABASE_URL=$($env_['SUPABASE_URL'])" `
        --build-arg "VITE_SUPABASE_ANON_KEY=$($env_['SUPABASE_ANON_KEY'])" `
        --build-arg "VITE_LIFF_ID=$($env_['LINE_LIFF_ID'])"
    Assert-LastExit 'Frontend image build'

    Write-Host "Built x-coach-backend:$t and x-coach-frontend:$t" -ForegroundColor Green
}

function Invoke-Apps {
    Assert-LoggedIn
    $env_ = Read-DotEnv $EnvFile
    $acr = Get-DeploymentOutput 'acrName'
    $login = Get-DeploymentOutput 'acrLoginServer'
    $t = Resolve-Tag

    foreach ($required in @('R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET')) {
        if ([string]::IsNullOrWhiteSpace($env_[$required])) {
            # /app/data is read-only and replicas are ephemeral, so the local-store fallback
            # has nowhere to write — and it falls back SILENTLY, at WARNING level.
            throw "$required is empty in $EnvFile. All four R2_* values are required in Azure; without them uploads silently fall back to a read-only local store."
        }
    }

    az deployment group create -g $ResourceGroup -n main -f infra/main.bicep `
        -p '@infra/main.parameters.json' `
        -p location=$Location `
        -p deployApps=true `
        -p backendImage="$login/x-coach-backend:$t" `
        -p frontendImage="$login/x-coach-frontend:$t" `
        -p supabaseUrl="$($env_['SUPABASE_URL'])" `
        -p supabaseAnonKey="$($env_['SUPABASE_ANON_KEY'])" `
        -p supabaseServiceRoleKey="$($env_['SUPABASE_SERVICE_ROLE_KEY'])" `
        -p lineChannelId="$($env_['LINE_CHANNEL_ID'])" `
        -p lineMessagingChannelSecret="$($env_['LINE_MESSAGING_CHANNEL_SECRET'])" `
        -p lineMessagingAccessToken="$($env_['LINE_MESSAGING_ACCESS_TOKEN'])" `
        -p lineLiffId="$($env_['LINE_LIFF_ID'])" `
        -p llmApiKey="$($env_['LLM_API_KEY'])" `
        -p llmModels="$($env_['LLM_MODELS'])" `
        -p llmBaseUrl="$($env_['LLM_BASE_URL'])" `
        -p r2AccountId="$($env_['R2_ACCOUNT_ID'])" `
        -p r2AccessKeyId="$($env_['R2_ACCESS_KEY_ID'])" `
        -p r2SecretAccessKey="$($env_['R2_SECRET_ACCESS_KEY'])" `
        -p r2Bucket="$($env_['R2_BUCKET'])" `
        -o none
    Assert-LastExit 'Pass 2 (deployApps=true)'

    Invoke-Status
}

function Invoke-Update {
    Assert-LoggedIn
    $login = Get-DeploymentOutput 'acrLoginServer'
    $t = Resolve-Tag
    az containerapp update -n xcoach-backend -g $ResourceGroup --image "$login/x-coach-backend:$t" -o none
    Assert-LastExit 'Updating the backend image'
    az containerapp update -n xcoach-frontend -g $ResourceGroup --image "$login/x-coach-frontend:$t" -o none
    Assert-LastExit 'Updating the frontend image'
    Write-Host "Both apps now run tag $t." -ForegroundColor Green
}

function Invoke-Status {
    Assert-LoggedIn
    $fqdn = Get-DeploymentOutput 'frontendFqdn'
    Write-Host ''
    Write-Host "App: https://$fqdn" -ForegroundColor Green

    # Hit it from outside: this proves the nginx /api proxy and the internal ingress, not
    # merely that the backend container came up. On a scaled-to-zero backend the first call
    # pays the cold start, so allow it a generous timeout.
    Write-Host 'Checking /api/health (a cold backend may take a minute to wake) ...' -ForegroundColor Cyan
    try {
        $health = Invoke-RestMethod -Uri "https://$fqdn/api/health" -TimeoutSec 180
    } catch {
        throw "Health check failed: $_"
    }
    $health | ConvertTo-Json -Depth 5 | Write-Host

    if (-not $health.storage_configured) {
        Write-Host 'storage_configured is FALSE — the backend fell back to local storage and uploads will not survive. Check the four R2_* values.' -ForegroundColor Red
    } else {
        Write-Host 'storage_configured: true (R2 is live).' -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------------------

switch ($Stage) {
    'providers' { Invoke-Providers }
    'infra'     { Invoke-Infra }
    'data'      { Invoke-Data }
    'build'     { Invoke-Build }
    'apps'      { Invoke-Apps }
    'update'    { Invoke-Update }
    'status'    { Invoke-Status }
}
