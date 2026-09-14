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

        providers   register the three resource providers (once per subscription)
        infra       resource group + Log Analytics + storage + environment (no apps)
        data        upload the KG and the RAG vector DB to the Azure Files share
        build       wait for the GitHub Actions run that builds and pushes both images
        apps        deploy the two container apps
        dns         print the DNS records each hostname in customDomains needs
        domain      add the hostnames, then issue and bind their managed certificates
        update      re-point the existing apps at a freshly built tag
        status      print the FQDN and run the post-deploy health checks

.EXAMPLE
    ./infra/deploy.ps1 -Stage providers
    ./infra/deploy.ps1 -Stage infra
    ./infra/deploy.ps1 -Stage data
    ./infra/deploy.ps1 -Stage build
    ./infra/deploy.ps1 -Stage apps
    ./infra/deploy.ps1 -Stage dns       # then create the records it prints
    ./infra/deploy.ps1 -Stage domain
    ./infra/deploy.ps1 -Stage status

.NOTES
    Run from the repository root. Requires `az login` to have been done already.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('providers', 'infra', 'data', 'build', 'apps', 'dns', 'domain', 'update', 'status')]
    [string]$Stage,

    [string]$ResourceGroup = 'xcoach-rg',
    # NOT eastasia. The Azure for Students subscription carries a built-in policy assignment
    # (sys.regionrestriction) that allows only malaysiawest, southeastasia, japanwest,
    # japaneast and koreacentral; anything else fails with RequestDisallowedByAzure.
    [string]$Location = 'japaneast',
    [string]$EnvFile = '.env',

    # Where .github/workflows/build-images.yml pushes. Must be lowercase (GHCR requires it).
    [string]$ImageRepo = 'ghcr.io/lucas6028',

    # Defaults to the full commit SHA, because that is what the workflow tags with
    # (github.sha). The commit must be pushed, or no image carries this tag.
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
    if (-not (Test-Path $Path)) { throw "No $Path -- copy .env.example and fill it in." }
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

function Get-CustomDomains {
    # The hostnames live in the parameters file, not in .env: they are infrastructure, not a
    # secret, and the template needs them on every pass or the ingress drops the binding.
    $params = Get-Content 'infra/main.parameters.json' -Raw | ConvertFrom-Json
    $value = $params.parameters.customDomains.value
    if ($null -eq $value) { return @() }
    return @($value)
}

function Test-IsApex {
    # One dot = apex (xcoach.dev). Mirrors the template's rule for the validation method.
    param([string]$Hostname)
    return (($Hostname -split '\.').Count -eq 2)
}

function Get-BoundHostnames {
    # Hostnames the frontend ALREADY serves with a certificate. `apps` must re-declare them
    # as bound, or the redeploy downgrades the binding; before `domain` has run there are
    # none and the template must not ask for certificates yet.
    # Joined before parsing: az emits one line per array element, and Windows PowerShell's
    # ConvertFrom-Json chokes on a multi-line document fed to it line by line.
    $json = (az containerapp hostname list -n xcoach-frontend -g $ResourceGroup -o json 2>$null) -join "`n"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($json)) { return @() }
    $list = $json | ConvertFrom-Json
    return @($list | Where-Object { $_.bindingType -eq 'SniEnabled' } | ForEach-Object { $_.name })
}

function Resolve-Tag {
    if ($Tag) { return $Tag }
    $sha = (git rev-parse HEAD).Trim()
    Assert-LastExit 'git rev-parse'
    return $sha
}

function Get-ImageRef {
    param([string]$Component, [string]$TagValue)
    return "$ImageRepo/x-coach-$Component`:$TagValue"
}

function Assert-ImagePullable {
    # Two ways to reach `apps` with an unpullable image, and both look like a broken template
    # rather than a broken image: the tag defaults to HEAD, but the workflow only builds when
    # something image-relevant changed (a docs-only commit produces no tag); and a GHCR package
    # is PRIVATE until someone flips it in the UI, which no API can do. Anonymous pull is
    # exactly what Container Apps attempts, so ask the registry the same question it will.
    param([string]$Component, [string]$TagValue)
    $path = ($ImageRepo -replace '^ghcr\.io/', '') + "/x-coach-$Component"
    try {
        $token = (Invoke-RestMethod "https://ghcr.io/token?scope=repository:${path}:pull&service=ghcr.io" -TimeoutSec 30).token
        $accept = 'application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'
        Invoke-WebRequest "https://ghcr.io/v2/$path/manifests/$TagValue" `
            -Headers @{ Authorization = "Bearer $token"; Accept = $accept } `
            -Method Head -TimeoutSec 30 -ErrorAction Stop | Out-Null
    } catch {
        $code = $_.Exception.Response.StatusCode.value__
        if ($code -eq 401 -or $code -eq 403) {
            throw "ghcr.io/$path is not anonymously pullable (HTTP $code). Make the package PUBLIC: https://github.com/users/$(($path -split '/')[0])/packages/container/x-coach-$Component/settings -> Danger Zone -> Change visibility."
        }
        throw "No image ghcr.io/${path}:$TagValue (HTTP $code). The workflow only builds on image-relevant paths, so a docs-only commit produces no tag -- pass -Tag <sha of a built commit>."
    }
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
    foreach ($ns in @('Microsoft.App', 'Microsoft.OperationalInsights', 'Microsoft.Storage')) {
        Write-Host "Registering $ns ..." -ForegroundColor Cyan
        az provider register --namespace $ns --wait
        Assert-LastExit "Registering $ns"
    }
    Write-Host 'All three providers registered.' -ForegroundColor Green
}

function Invoke-Infra {
    Assert-LoggedIn
    az group create -n $ResourceGroup -l $Location -o none
    Assert-LastExit 'az group create'

    # Pass 1 skips the apps: they mount the data share this pass creates.
    az deployment group create -g $ResourceGroup -n main -f infra/main.bicep `
        -p '@infra/main.parameters.json' `
        -p location=$Location `
        -p deployApps=false `
        -o none
    Assert-LastExit 'Pass 1 (deployApps=false)'

    Write-Host "Storage: $(Get-DeploymentOutput 'storageAccountName')" -ForegroundColor Green
    Write-Host ''
    Write-Host 'Set a budget alert before going further -- on a student subscription, ' -NoNewline
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
        throw 'data/kg/sports_kg_v3.graphml is missing. It is gitignored and pipeline-built -- generate it before deploying, or the knowledge endpoints will report their stores absent.'
    }
    if (-not (Test-Path 'data/rag/vector_db')) {
        throw 'data/rag/vector_db is missing. Build the RAG store before deploying.'
    }

    # `az storage file upload` will not create intermediate directories, and a missing one
    # surfaces as ParentNotFound rather than as anything mentioning directories. Azure Files
    # has no implicit hierarchy, so each level is created on its own; both calls are
    # idempotent-by-hand, hence -o none with the exit code ignored on "already exists".
    foreach ($dir in @('kg', 'rag', 'rag/vector_db')) {
        az storage directory create --account-name $storage --account-key $key `
            --share-name $share --name $dir -o none 2>$null
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
    Write-Host 'The demo video library (data/Fitness-AQA/...) is optional and large -- upload it the same way only if you want the pre-processed demos.'
}

function Invoke-Build {
    # Images are NOT built here. `az acr build` runs on ACR Tasks, which is disabled for any
    # subscription spending student or trial credit (TasksOperationsNotAllowed), and this
    # machine has no Docker daemon. .github/workflows/build-images.yml builds both images and
    # pushes them to GHCR; this stage only waits for that run.
    $t = Resolve-Tag
    $branch = (git rev-parse --abbrev-ref HEAD).Trim()
    Write-Host "Waiting for the build-images run on $branch ..." -ForegroundColor Cyan

    $runId = gh run list --workflow build-images.yml --branch $branch --limit 1 --json databaseId --jq '.[0].databaseId'
    Assert-LastExit 'Listing workflow runs'
    if ([string]::IsNullOrWhiteSpace($runId)) {
        throw "No build-images run on '$branch'. Push the branch first -- the workflow triggers on push."
    }
    gh run watch $runId.Trim() --exit-status
    Assert-LastExit 'The build-images workflow'

    Write-Host "Built $(Get-ImageRef 'backend' $t)" -ForegroundColor Green
    Write-Host "Built $(Get-ImageRef 'frontend' $t)" -ForegroundColor Green
    Write-Host 'Both GHCR packages must be PUBLIC, or Container Apps cannot pull them.' -ForegroundColor Yellow
}

function Invoke-Apps {
    Assert-LoggedIn
    $env_ = Read-DotEnv $EnvFile
    $t = Resolve-Tag
    Assert-ImagePullable 'backend' $t
    Assert-ImagePullable 'frontend' $t

    # Into variables first: `-p name=(Get-ImageRef ...)` makes PowerShell split the token, so
    # az receives the bare image reference as its own -p and answers "Unable to parse parameter".
    $backendImage = Get-ImageRef 'backend' $t
    $frontendImage = Get-ImageRef 'frontend' $t

    foreach ($required in @('R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET')) {
        if ([string]::IsNullOrWhiteSpace($env_[$required])) {
            # /app/data is read-only and replicas are ephemeral, so the local-store fallback
            # has nowhere to write — and it falls back SILENTLY, at WARNING level.
            throw "$required is empty in $EnvFile. All four R2_* values are required in Azure; without them uploads silently fall back to a read-only local store."
        }
    }

    # Certificates only for hostnames the frontend already serves bound. On a first `apps`
    # (or a hostname newly added to the parameters file) that is none of them: the template
    # adds the hostname unbound and `-Stage domain` issues the certificate afterwards.
    $domains = Get-CustomDomains
    $bound = Get-BoundHostnames
    $bind = ($domains.Count -gt 0) -and (@($domains | Where-Object { $bound -notcontains $_ }).Count -eq 0)
    if ($domains.Count -gt 0 -and -not $bind) {
        Write-Host "customDomains not yet bound on the frontend; deploying them unbound. Run -Stage domain afterwards." -ForegroundColor Yellow
        Assert-DnsRecords $domains
    }

    Invoke-AppsDeployment -Settings $env_ -BackendImage $backendImage -FrontendImage $frontendImage -BindCertificates $bind
    Assert-LastExit 'Pass 2 (deployApps=true)'

    Invoke-Status
}

function Invoke-AppsDeployment {
    param(
        [hashtable]$Settings,
        [string]$BackendImage,
        [string]$FrontendImage,
        [bool]$BindCertificates
    )
    $bindArg = if ($BindCertificates) { 'true' } else { 'false' }
    az deployment group create -g $ResourceGroup -n main -f infra/main.bicep `
        -p '@infra/main.parameters.json' `
        -p location=$Location `
        -p deployApps=true `
        -p bindCertificates=$bindArg `
        -p backendImage="$BackendImage" `
        -p frontendImage="$FrontendImage" `
        -p supabaseUrl="$($Settings['SUPABASE_URL'])" `
        -p supabaseAnonKey="$($Settings['SUPABASE_ANON_KEY'])" `
        -p supabaseServiceRoleKey="$($Settings['SUPABASE_SERVICE_ROLE_KEY'])" `
        -p lineChannelId="$($Settings['LINE_CHANNEL_ID'])" `
        -p lineMessagingChannelSecret="$($Settings['LINE_MESSAGING_CHANNEL_SECRET'])" `
        -p lineMessagingAccessToken="$($Settings['LINE_MESSAGING_ACCESS_TOKEN'])" `
        -p lineLiffId="$($Settings['LINE_LIFF_ID'])" `
        -p llmApiKey="$($Settings['LLM_API_KEY'])" `
        -p llmModels="$($Settings['LLM_MODELS'])" `
        -p llmBaseUrl="$($Settings['LLM_BASE_URL'])" `
        -p r2AccountId="$($Settings['R2_ACCOUNT_ID'])" `
        -p r2AccessKeyId="$($Settings['R2_ACCESS_KEY_ID'])" `
        -p r2SecretAccessKey="$($Settings['R2_SECRET_ACCESS_KEY'])" `
        -p r2Bucket="$($Settings['R2_BUCKET'])" `
        -o none
}

function Get-DnsPlan {
    # One row per record to create at the registrar. Apex hostnames need an A record to the
    # environment's static IP (a CNAME at the apex is not valid DNS); everything deeper is a
    # CNAME to the frontend's generated FQDN. Every hostname also proves ownership with a
    # TXT record at asuid.<hostname>.
    # Read off the environment itself, not the deployment outputs: a deployment made with
    # an older template has no frontendDefaultFqdn / customDomainVerificationId output yet.
    $json = (az containerapp env show -n xcoach-env -g $ResourceGroup `
        --query '{ip:properties.staticIp,domain:properties.defaultDomain,verify:properties.customDomainConfiguration.customDomainVerificationId}' -o json) -join "`n"
    Assert-LastExit 'Reading the environment'
    $envInfo = $json | ConvertFrom-Json
    $ip = $envInfo.ip
    $fqdn = "xcoach-frontend.$($envInfo.domain)"
    $verify = $envInfo.verify
    $rows = @()
    foreach ($d in (Get-CustomDomains)) {
        if (Test-IsApex $d) {
            $rows += [pscustomobject]@{ Type = 'A';     Name = $d;          Value = $ip }
        } else {
            $rows += [pscustomobject]@{ Type = 'CNAME'; Name = $d;          Value = $fqdn }
        }
        $rows += [pscustomobject]@{ Type = 'TXT';       Name = "asuid.$d";  Value = $verify }
    }
    return $rows
}

function Assert-DnsRecords {
    # Ask the public resolvers the same question the platform will. Adding a hostname whose
    # asuid TXT is missing fails the whole deployment with a validation error that names
    # neither DNS nor the record, so check up front.
    param([string[]]$Domains)
    $plan = Get-DnsPlan
    $missing = @()
    foreach ($row in $plan) {
        $ok = $false
        try {
            $answers = @(Resolve-DnsName -Name $row.Name -Type $row.Type -DnsOnly -ErrorAction Stop)
            switch ($row.Type) {
                'A'     { $ok = [bool]($answers | Where-Object { $_.IPAddress -eq $row.Value }) }
                'CNAME' { $ok = [bool]($answers | Where-Object { $_.NameHost -eq $row.Value }) }
                'TXT'   { $ok = [bool]($answers | Where-Object { ($_.Strings -join '') -eq $row.Value }) }
            }
        } catch { $ok = $false }
        if (-not $ok) { $missing += "$($row.Type) $($row.Name) -> $($row.Value)" }
    }
    if ($missing.Count -gt 0) {
        throw "DNS is not ready; create (or wait for) these records first:`n  " + ($missing -join "`n  ")
    }
    Write-Host 'DNS records verified.' -ForegroundColor Green
}

function Invoke-Dns {
    Assert-LoggedIn
    $domains = Get-CustomDomains
    if ($domains.Count -eq 0) { throw 'customDomains is empty in infra/main.parameters.json.' }
    Write-Host 'Create these records at the registrar (name.com for xcoach.dev), replacing any A/CNAME the names already have:' -ForegroundColor Cyan
    Get-DnsPlan | Format-Table -AutoSize | Out-String | Write-Host
    Write-Host 'If the zone carries a CAA record, it must also allow "0 issue digicert.com".'
    Write-Host 'Then: ./infra/deploy.ps1 -Stage domain' -ForegroundColor Cyan
}

function Invoke-Domain {
    Assert-LoggedIn
    $env_ = Read-DotEnv $EnvFile
    $domains = Get-CustomDomains
    if ($domains.Count -eq 0) { throw 'customDomains is empty in infra/main.parameters.json.' }

    # Re-deploy the apps at the image they run NOW, not at HEAD: this stage changes the
    # ingress, not the code, and HEAD may have no built image.
    $backendImage = az containerapp show -n xcoach-backend -g $ResourceGroup --query 'properties.template.containers[0].image' -o tsv
    Assert-LastExit 'Reading the backend image'
    $frontendImage = az containerapp show -n xcoach-frontend -g $ResourceGroup --query 'properties.template.containers[0].image' -o tsv
    Assert-LastExit 'Reading the frontend image'

    Assert-DnsRecords $domains

    # Pass A: hostnames on the frontend, unbound. The platform verifies the asuid TXT here.
    Write-Host 'Adding the hostnames (unbound) ...' -ForegroundColor Cyan
    Invoke-AppsDeployment -Settings $env_ -BackendImage $backendImage.Trim() -FrontendImage $frontendImage.Trim() -BindCertificates $false
    Assert-LastExit 'Adding the hostnames'

    # Pass B: one managed certificate per hostname, then the SNI binding. DigiCert validates
    # the apex over HTTP against the app itself, so this needs pass A to have gone live.
    Write-Host 'Issuing and binding the managed certificates (a few minutes) ...' -ForegroundColor Cyan
    Invoke-AppsDeployment -Settings $env_ -BackendImage $backendImage.Trim() -FrontendImage $frontendImage.Trim() -BindCertificates $true
    Assert-LastExit 'Binding the certificates'

    foreach ($d in $domains) {
        Write-Host "https://$d" -ForegroundColor Green
    }
    Write-Host 'Now update the three things outside Azure: Supabase Site URL / Redirect URLs, the LIFF endpoint, and the LINE webhook URL (docs/azure-deployment.md).' -ForegroundColor Yellow
}

function Invoke-Update {
    Assert-LoggedIn
    $t = Resolve-Tag
    Assert-ImagePullable 'backend' $t
    Assert-ImagePullable 'frontend' $t
    az containerapp update -n xcoach-backend -g $ResourceGroup --image (Get-ImageRef 'backend' $t) -o none
    Assert-LastExit 'Updating the backend image'
    az containerapp update -n xcoach-frontend -g $ResourceGroup --image (Get-ImageRef 'frontend' $t) -o none
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
        Write-Host 'storage_configured is FALSE -- the backend fell back to local storage and uploads will not survive. Check the four R2_* values.' -ForegroundColor Red
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
    'dns'       { Invoke-Dns }
    'domain'    { Invoke-Domain }
    'update'    { Invoke-Update }
    'status'    { Invoke-Status }
}
