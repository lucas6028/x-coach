# 部署 x-coach 到 Azure

同一個環境裡兩個 Container Apps，與 `docker-compose.yml` 一一對應：`backend`（FastAPI 加上
pose/rules/retrieval pipeline）放在**內部** ingress 後面，`frontend`（nginx 服務打包好的 SPA）用
**對外** ingress 並把 `/api` 代理過去。Postgres/auth 留在 Supabase，使用者上傳留在 Cloudflare R2，
所以這兩者在 Azure 上都沒有對應資源。基礎設施定義在 `infra/main.bicep`。

先讀 `docs/docker.md` — 關於這兩個 image 的一切（裡面裝了什麼、為什麼 `data/` 是掛載而不是烘進
image、為什麼 `VITE_*` 是 build args）在這裡完全適用，不需要重講。

## 各部分放在哪裡

| 專案部分 | Azure 服務 | 說明 |
| --- | --- | --- |
| `frontend/`（nginx + 打包好的 SPA） | Container Apps | 保留 `/api` 同源代理，SSE 與 256 MB 上傳都不用改寫 |
| `backend/`（FastAPI + MediaPipe + ffmpeg） | Container Apps，Consumption profile | CPU/RAM 密集、單一 worker、需要調高 request timeout |
| 兩個 image | Azure Container Registry（Basic） | 用 user-assigned managed identity 拉取；admin 帳戶保持關閉 |
| Postgres、auth、RLS | **維持 Supabase 不動** | 沒有 Azure 資源。搬走等於重寫 auth、RLS 與 `supabase-py` |
| 上傳內容（影片、pose JSON、縮圖） | **維持 Cloudflare R2 不動** | `services/storage.py` 走 S3 API；Azure Blob 沒有 S3 相容端點，要換就得寫一份新的 store 實作 |
| `data/` — KG graphml、RAG 向量庫、demo 影片庫 | Azure Files 檔案共用，唯讀掛載於 `/app/data` | 與 compose 的 `./data:/app/data:ro` 同形 |
| 機密設定 | Container Apps secrets（或 Key Vault 參照） | 絕不放進參數檔 — 見[機密設定](#機密設定) |
| Log 與追蹤 | Log Analytics（範本已接好）+ Application Insights | |
| 自訂網域與 TLS | Container Apps 自訂網域 + 免費受管憑證 | 只有 frontend 需要；backend 是內部的 |
| 研究 pipeline（`src/rehab24`、`src/video`、torch/VideoMAE、Gemini KG 抽取） | **不部署** | 真的需要上雲就用 Container Apps *Jobs* 或 Azure ML，不是這個 app。`requirements-docker.txt` 刻意排除了它們的相依套件 |

**區域。** 範本用 `japaneast`。East Asia（香港）離台灣最近，但 **Azure for Students 訂閱不准開在
那裡** — 訂閱上掛著一個 `sys.regionrestriction` policy，只放行 `malaysiawest`、`southeastasia`、
`japanwest`、`japaneast`、`koreacentral`，其餘一律 `RequestDisallowedByAzure`。要確認自己這個
訂閱放行哪些：

```bash
az policy assignment show -n sys.regionrestriction --query parameters.listOfAllowedLocations.value
```

在放行清單裡挑離 Supabase 專案較近的，因為每一次請求都要付那趟來回。

## 拓撲

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

backend 的 ingress 設為內部有兩個理由：環境外面沒有東西需要這個 API；而且當 `R2_*` 變數未設定時，
backend 會暴露一個**未經驗證的** `GET /api/local-object/{key}`，那絕不能面向網際網路。

唯一真的從外面進來的是 LINE Messaging webhook。它打的是 frontend 的公開主機名稱，由 nginx 像其他
`/api` 路由一樣轉發 — 不需要另外開一個公開端點，也沒有理由把 backend 變成對外。

## 為什麼 frontend 放 Container Apps，而不是 Static Web Apps 或 Vercel

frontend 這個 image 不是靜態網站：`nginx.conf.template` 同時也是反向代理，而它做的三件事是
CDN 優先的平台辦不到的。

- **長時間請求。** 冷啟動的分析會跑上幾分鐘；設定裡允許 900 秒。Azure Static Web Apps 與 Vercel
  對代理請求的上限都遠低於此（Vercel 的外部 rewrite 上限是 120 秒，而且不可設定）。
- **大型上傳。** `client_max_body_size 256m`。Static Web Apps 的單次請求上限是 30 MB。
- **SSE。** `/api/chat` 需要 `proxy_buffering off` 才能逐 token 串流；透過第三方代理層的行為
  至少可以說是沒有文件保證的。

要避開這三點，就得放棄 `/api` 代理、改成跨來源呼叫 backend — 但 `frontend/src/api.ts` 裡每一個呼叫
都是寫死的相對路徑（`fetch("/api/...")`，而 `videoFileUrl` 直接回傳 `/api/video-file/{id}` 塞進
`<video src>`）。那是一次橫跨約 30 個呼叫點的 `VITE_API_BASE_URL` 重構，外加 CORS，外加一個公開的
backend，換來的只是用 CDN 遞送一個小型 SPA。以這個規模來說不值得。

## 三件必須弄對的事

### 1. ingress 的請求逾時預設是 240 秒

Container Apps 在每個 app 前面都擺了 Envoy，它會在 240 秒切斷請求。冷啟動的分析
（MediaPipe + rules + RAG）有可能超過，而 `nginx.conf.template` 裡寫 900 秒並不能改變這件事 —
平台會先動手。

三條出路，依「做得徹底」的程度排列：

1. **Premium ingress**，可把逾時設定到最長一小時。沒有寫進 Bicep 範本，因為它會改變環境的計費
   方式；請另外啟用：

   ```bash
   az containerapp env update -n xcoach-env -g <rg> \
       --enable-premium-ingress --request-idle-timeout 15
   ```

2. **優先走客戶端 pose 路徑。** 瀏覽器跑 MediaPipe 並送出 pose JSON（`analyze.py` 已經支援）；
   伺服器端只跑 rules 與 retrieval，秒級就完成。240 秒綽綽有餘。
3. **把分析改成非同步。** 上傳 → Storage Queue 或 Service Bus → 由 KEDA 觸發的
   **Container Apps Job** → 客戶端輪詢。`analyze.py:22` 本來就把行程內的 semaphore 稱為
   「until the Celery/Redis worker queue lands」的權宜之計；Jobs 就是 Azure 上的那個 queue，
   而且不必自己養 broker。

先做 (1) 或 (2)。等分析量真的起來，(3) 才是正解；一開始就做是過度工程。

### 2. R2 是必要的，不是選配

`/app/data` 是唯讀掛載，所以 `LocalObjectStore` 沒有地方可寫 — 而且 replica 是短暫的，就算寫成功
了，下一個修訂版也會讓它消失。四個 `R2_*` 變數必須全部設定。少一個或拼錯一個，app 就會**靜默地**
退回本機儲存。

每次部署後都要驗證：

```bash
curl -s https://<frontend-fqdn>/api/health | grep storage_configured   # 必須是 true
```

並檢查啟動 log 裡有 INFO 等級的 `Object storage: Cloudflare R2 (bucket=...)`。退回本機時記的是
WARNING。

如果你比較想保留本機儲存，可以再加一個 Azure Files 檔案共用、以讀寫模式掛在 `/app/data/runtime` —
但 R2 比較便宜、沒有 egress 費用，而且已經有 `.env.example` 裡描述的 `uploads/anon/` 生命週期規則。

### 3. 用 replica 擴展，不是用 worker

`XCOACH_MAX_CONCURRENT_ANALYSES` 是一個**每行程**的 semaphore（`backend/app/config.py`）。因此
範本讓每個 replica 只跑一個 uvicorn worker，並把 HTTP scale rule 的 `concurrentRequests` 設成同一個
數字，讓 replica 在它的 semaphore 飽和的那一刻才擴展出去，而不是等到 Envoy 預設的 10 — 超過之後
請求會無聲地排隊，而 replica 看起來依然健康。

`backendMinReplicas` 是這份範本裡唯一真正花錢的旋鈕，見下一節。

Consumption 的上限是每個 replica 4 vCPU / 8 GiB，記憶體固定為每 vCPU 2 GiB。超過就需要
dedicated workload profile。

## 成本，以及 Azure 學生方案

Container Apps 按 vCPU-秒與 GiB-秒計費。每個訂閱每月頭 180,000 vCPU-秒、360,000 GiB-秒、
200 萬次請求免費；超出後**執行中**是 $0.000024/vCPU-秒，而 `minReplicas > 0` 但沒有請求在處理的
**閒置** replica 是 $0.000008/vCPU-秒（記憶體 $0.000001/GiB-秒）。scale 到 0 則完全不計。

閒置費率才是重點：一個 `minReplicas: 1` 的 replica 就算整個月沒人用，也是整個月都在計費。

| 設定 | 每月約略 | $100 學生額度可撐 |
| --- | --- | --- |
| backend 2 vCPU、`minReplicas: 1` | $50 + frontend $6 + ACR $5 ≈ **$61** | 約 7 週 |
| backend 1 vCPU、`minReplicas: 1` | $24 + $6 + $5 ≈ **$36** | 約 3 個月 |
| backend `minReplicas: 0`（**範本預設**） | 用多少算多少 + $6 + $5 ≈ **$12** | 約 8 個月 |

所以 `infra/main.parameters.json` 預設 `backendMinReplicas: 0`。代價是冷啟動：第一個請求要等
容器起來、載入 MediaPipe、再從 SMB 共用讀知識圖譜與向量庫。兩個後果要知道：

- **LINE Messaging webhook 可能逾時重試。** LINE 對 webhook 的等待很短，冷的 backend 接不下。
  如果 bot 是要給真人用的，把 `backendMinReplicas` 調回 1，接受上面那張表的第二列。
- **demo 前先把它叫醒。** 口試或展示前打一次 `https://<fqdn>/api/health`，之後只要有流量
  replica 就會維持喚醒。

frontend 維持 `minReplicas: 1`：它只是 nginx，0.25 vCPU 一個月約 $6，換來的是網站本身永遠是熱的。
真的要再省，把 `frontendMinReplicas` 也設成 0，第一位訪客多等兩三秒。

學生訂閱另外三件事：

- **先設預算警示。** 額度歸零時資源會被停用，不是寄帳單給你。在 Cost Management 裡對
  resource group 設 50% 與 80% 的 alert，在跑第二趟部署之前就設好。
- **配額**。免費／學生訂閱不能申請提高配額，Container Apps environment 的數量上限很低
  （常見錯誤訊息是 `Environment limit reached`）。這份範本只建一個環境。
- **區域限制**。學生訂閱在部分區域不能開資源。`location` 是參數，而 ACR、儲存體與環境共用它，
  所以換區域只是改一個參數，不是逐一搬資源。實測 `eastasia` 就是被擋的那個，見上面的區域說明。

## 部署

在 Windows 上請跑 `infra/deploy.ps1`，它把下面每一段包成一個 stage，並且從 `.env` 讀出那十幾個
設定值，不用把八個機密貼到命令列上：

```powershell
az login                              # 互動式，只有你能跑
./infra/deploy.ps1 -Stage providers   # 註冊四個 resource provider
./infra/deploy.ps1 -Stage infra       # 第一趟：registry、環境、儲存體
./infra/deploy.ps1 -Stage data        # 灌 KG 與向量庫
./infra/deploy.ps1 -Stage build       # 在 ACR 裡建兩個 image
./infra/deploy.ps1 -Stage apps        # 第二趟：兩個 container app
```

下面的 bash 版本是同一件事，給非 Windows 環境、也給你知道每個 stage 實際做了什麼。**不要在
Git Bash 裡跑 `az`**：MSYS 會把任何以 `/` 開頭的參數當成路徑改寫，resource ID 與 `--scope`
會被無聲地改壞。

### 第零步：註冊 resource provider

全新的訂閱一個都沒註冊，而第一趟部署只會回 `MissingSubscriptionRegistration`，不會告訴你少哪個。

```bash
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.Storage; do
    az provider register --namespace $ns --wait
done
```

### 第一趟：registry 與環境

container apps 引用的 image 在 registry 存在之前並不存在，所以第一次部署要跳過它們。

```bash
RG=xcoach-rg
az group create -n $RG -l japaneast

az deployment group create -g $RG -n main -f infra/main.bicep \
    -p @infra/main.parameters.json -p deployApps=false
ACR=$(az deployment group show -g $RG -n main --query properties.outputs.acrName.value -o tsv)
```

### 灌入資料共用

KG 與 RAG 儲存是由 pipeline 產生且被 gitignore 的，所以是上傳而不是在雲端建置。在 pipeline 跑完
之後，從 repo 根目錄執行：

```bash
STORAGE=$(az deployment group show -g $RG -n main \
    --query properties.outputs.storageAccountName.value -o tsv)
SHARE=$(az deployment group show -g $RG -n main \
    --query properties.outputs.dataShareName.value -o tsv)
KEY=$(az storage account keys list -g $RG -n $STORAGE --query '[0].value' -o tsv)

# 只送 backend 真的會開的那個圖：data/kg/ 另外那八個 .bak / .pre-* / .post-*-raw 是
# pipeline 的歷史快照，不是執行時的輸入。路徑對應 backend/app/config.py 的 KG_GRAPH_FILE。
az storage file upload --account-name $STORAGE --account-key $KEY \
    --share-name $SHARE --path kg/sports_kg_v3.graphml --source data/kg/sports_kg_v3.graphml

az storage file upload-batch --account-name $STORAGE --account-key $KEY \
    --destination $SHARE --destination-path rag/vector_db --source data/rag/vector_db
```

demo 影片庫（`data/Fitness-AQA/Squat/Labeled_Dataset/`）是選配而且很大；只有在你要那些即時 demo
影片時才用同樣方式上傳。沒有它 API 一樣能服務 — `/api/health` 會回報那些 store 不存在，就跟一個
什麼都沒掛載的 `docker run` 一樣。

> 冷啟動的替代方案：如果讀取 SMB 上的向量庫真的在啟動延遲上顯現出來，改成把 `data/kg/` 與
> `data/rag/vector_db/` 烘進 backend image。那需要改 `.dockerignore` — Docker 無法重新納入
> 一個父目錄已被排除的路徑，所以 `data` 那行得先改成 `data/*`，`!data/kg` 才會生效。只有在量測
> 證明有必要時才做；KG 與向量庫都很小，而且 embedder 是 hash-based 的
> （`src/knowledge/rag_vector_db.py`），不會去下載任何模型。

### 建置與推送

兩個 image 都在 ACR 裡建置，所以不需要本機的 Docker daemon。backend 的 build context 是
**repo 根目錄**（它以絕對套件路徑 import `backend.*` 與 `src.*`）；frontend 的是 `frontend/`。

```bash
TAG=$(git rev-parse --short HEAD)

az acr build -r $ACR -t x-coach-backend:$TAG -f backend/Dockerfile .

# -f 是相對於 context 根目錄，不是相對於工作目錄 — 所以 backend 那行寫得出路徑，
# frontend 這行就只能是裸的 Dockerfile。
az acr build -r $ACR -t x-coach-frontend:$TAG -f Dockerfile frontend \
    --build-arg VITE_SUPABASE_URL=https://xxxx.supabase.co \
    --build-arg VITE_SUPABASE_ANON_KEY=eyJ... \
    --build-arg VITE_LIFF_ID=1234567890-Abcdefgh
```

Vite 會在建置時把 `VITE_*` 內嵌進 bundle，所以它們是 **build args，不是 runtime env** — 改動任何
一個都需要重新建置，不是重啟。anon key 可以安心隨 bundle 出貨；資料列的存取由 Postgres RLS 控管。

### 第二趟：兩個 app

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

frontend 會拿到指向 backend 內部 FQDN 的 `BACKEND_ORIGIN`。nginx 在容器啟動時把它代入設定檔
（`frontend/nginx.conf.template`），所以要重新指向代理目標是改 env，不是重新建置。

注意那份設定裡的 `Host` header 設的是 `$proxy_host`，不是 `$host`。內部 ingress 是按 Host 路由的，
把瀏覽器的主機名稱轉過去會讓環境找不到 backend app — 那是平台回的 404，但看起來像 API 的路由 bug。
瀏覽器原本的主機名稱保留在 `X-Forwarded-Host`。

### Supabase migrations 不在 Azure 這邊

`db/migrations/` 底下的 SQL 是手動套到 Supabase 的，跟這次部署完全沒有交集。上線前確認最新那幾支
（`20260813000000_training_plans.sql`、`20260725000000_analysis_movement.sql`）已經跑過 —
少了資料表的話，前端會壞在看起來像 Azure 問題的地方。

### 後續部署

只有 image tag 會變：

```powershell
./infra/deploy.ps1 -Stage build
./infra/deploy.ps1 -Stage update
```

```bash
az containerapp update -n xcoach-backend  -g $RG --image $ACR.azurecr.io/x-coach-backend:$TAG
az containerapp update -n xcoach-frontend -g $RG --image $ACR.azurecr.io/x-coach-frontend:$TAG
```

在 GitHub Actions 裡，用 OIDC federated credential 驗證（`azure/login@v2` 搭配
`client-id`/`tenant-id`/`subscription-id`，不存任何 secret），並在現有 `ci.yml` 的測試與覆蓋率
關卡通過之後執行同樣這兩行。

## 機密設定

`@secure()` 參數刻意不放在 `infra/main.parameters.json`，這樣那個檔案才能安心進版控。請如上面那樣
用命令列傳入，或者 — 對任何長期存在的機密而言更好 — 放進 Key Vault 再參照：

```json
"llmApiKey": {
  "reference": {
    "keyVault": { "id": "/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.KeyVault/vaults/<vault>" },
    "secretName": "llm-api-key"
  }
}
```

`SUPABASE_SERVICE_ROLE_KEY` 是最需要小心的一個：它的存在只是為了讓 LINE LIFF 橋接能鑄出一個登入
連結，絕不用於資料存取。它不能進到 frontend image，也不能出現在任何 build arg。

## 自訂網域與 TLS

系統產生的 `*.azurecontainerapps.io` FQDN 本來就提供 HTTPS，所以在你想用自己的主機名稱之前，TLS
不需要做任何事。之後 Container Apps 會簽發一張**免費的受管憑證**（DigiCert）並自動續期 — 沒有
certbot 要跑。

只有 frontend 需要；backend 是內部的。

1. 加上 DNS 記錄。子網域需要一筆指向 frontend FQDN 的 `CNAME`，再加一筆位於 `asuid.<sub>`、
   內容為驗證碼的 `TXT`。根網域（apex）需要一筆指向環境靜態 IP 的 `A` 記錄（範本輸出裡的
   `environmentStaticIp`），再加一筆位於 `asuid` 的 `TXT`。
2. 繫結網域並讓 Azure 簽發憑證：

   ```bash
   az containerapp hostname add -n xcoach-frontend -g $RG --hostname app.example.com
   az containerapp hostname bind -n xcoach-frontend -g $RG --hostname app.example.com \
       --environment xcoach-env --validation-method CNAME
   ```

兩件會讓簽發失敗的事：

- **CAA 記錄。** 如果根網域上存在任何 `CAA` 記錄，必須加上 `0 issue digicert.com`，否則簽發與
  續期都會失敗。
- **順序。** 請求憑證時 app 必須已經是公開可達的，因為 DigiCert 是透過 HTTP 驗證的。先部署，
  再設 DNS，最後才繫結。

主機名稱上線後，在參數檔裡設定 `customDomain`：範本會把它餵給 `XCOACH_CORS_ORIGINS`。這只有在
直接跨來源呼叫 API 時才有意義（SPA 本身透過 nginx 是同源的），但設對了也不花什麼成本。

在 frontend 前面加 Front Door 是選配。要 WAF 或全球快取時才加；TLS 與自訂網域都不需要它。

## 可觀測性

容器 log 會串流到範本建立的 Log Analytics workspace：

```bash
az containerapp logs show -n xcoach-backend -g $RG --follow
```

要做請求層級的追蹤，就加上 Application Insights 並用 OpenTelemetry 為 FastAPI 埋點。不論如何，
健康端點都是最快的第一道檢查 — `/api/health` 回報 `auth_configured`、`chat_configured`、
`line_login_configured`、`storage_configured`，以及一個涵蓋 labeled-video、detection、KG 與 RAG
目錄的 `stores` 對照表，這些加起來能解釋大部分「為什麼這個功能在正式環境是死的」這類問題。

## 這份文件不涵蓋的部分

研究 pipeline 不屬於這次部署，也不該被加進這兩個 image。`requirements-docker.txt` 是
`requirements.txt` 的 web 子集 — 把 `torch` 與 `transformers` 加回去，會為了 API 從不呼叫的程式碼
讓 backend image 膨脹好幾 GB。如果那些 pipeline 需要雲端運算資源，請用 Container Apps Jobs 或
Azure Machine Learning 執行，並使用它們自己、以完整 `requirements.txt` 建置的 image。
