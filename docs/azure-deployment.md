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
| 兩個 image | **GHCR（公開 package）** | Azure 建不了 image — ACR Tasks 對學生額度停用，見[建置與推送](#建置與推送在-github-actions不在-azure) |
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
| backend 2 vCPU、`minReplicas: 1` | $50 + frontend $6 ≈ **$56** | 約 8 週 |
| backend 1 vCPU、`minReplicas: 1` | $24 + $6 ≈ **$30** | 約 3 個月 |
| backend `minReplicas: 0`（**範本預設**） | 用多少算多少 + frontend $6 ≈ **$7** | 用不完（見下） |

第三列的額度不是被燒完的，是**過期**的：學生額度給 12 個月，到期就歸零，跟還剩多少無關。所以
scale-to-zero 的真正意義不是「撐更久」，而是「這 12 個月不必再想這件事」。

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
- **區域限制**。學生訂閱在部分區域不能開資源。`location` 是參數，而儲存體與環境共用它，
  所以換區域只是改一個參數，不是逐一搬資源。實測 `eastasia` 就是被擋的那個，見上面的區域說明。

## 部署

在 Windows 上請跑 `infra/deploy.ps1`，它把下面每一段包成一個 stage，並且從 `.env` 讀出那十幾個
設定值，不用把八個機密貼到命令列上：

```powershell
az login                              # 互動式，只有你能跑
./infra/deploy.ps1 -Stage providers   # 註冊三個 resource provider
./infra/deploy.ps1 -Stage infra       # 第一趟：環境、儲存體、Log Analytics
./infra/deploy.ps1 -Stage data        # 灌 KG 與向量庫
./infra/deploy.ps1 -Stage build       # 等 GitHub Actions 把兩個 image 建好
./infra/deploy.ps1 -Stage apps        # 第二趟：兩個 container app
./infra/deploy.ps1 -Stage dns         # 印出 xcoach.dev 需要的 DNS 記錄，去 name.com 建
./infra/deploy.ps1 -Stage domain      # 第三、四趟：掛上主機名稱，簽發並繫結憑證
```

下面的 bash 版本是同一件事，給非 Windows 環境、也給你知道每個 stage 實際做了什麼。**不要在
Git Bash 裡跑 `az`**：MSYS 會把任何以 `/` 開頭的參數當成路徑改寫，resource ID 與 `--scope`
會被無聲地改壞。

### 第零步：註冊 resource provider

全新的訂閱一個都沒註冊，而第一趟部署只會回 `MissingSubscriptionRegistration`，不會告訴你少哪個。

```bash
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.Storage; do
    az provider register --namespace $ns --wait
done
```

### 第一趟：環境與儲存體

container apps 要掛第一趟才建出來的資料共用，所以第一次部署先跳過它們。

```bash
RG=xcoach-rg
az group create -n $RG -l japaneast

az deployment group create -g $RG -n main -f infra/main.bicep \
    -p @infra/main.parameters.json -p deployApps=false
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

# 只送 backend 真的會開的那個圖：data/kg/ 其餘的 .bak / .pre-* / .post-*-raw 是
# pipeline 的歷史快照（每跑一支 author_*/reconcile_* 就多一個），不是執行時的輸入。
# 路徑對應 backend/app/config.py 的 KG_GRAPH_FILE。改完圖要重跑這一行才會進 prod。
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

### 建置與推送：在 GitHub Actions，不在 Azure

**Azure 建不了這兩個 image。** `az acr build` 跑在 ACR Tasks 上，而使用學生／試用額度的訂閱一律
被停用該功能：

```
(TasksOperationsNotAllowed) ACR Tasks requests for the registry ... are not permitted
```

官方唯一解法是升級成 Pay-As-You-Go，那就沒有 $100 額度了。所以 image 由
`.github/workflows/build-images.yml` 在 GitHub Actions 裡建，推到 **GHCR**：

```
ghcr.io/lucas6028/x-coach-backend:<commit sha>
ghcr.io/lucas6028/x-coach-frontend:<commit sha>
```

推送只用 Actions 內建的 `GITHUB_TOKEN`，不需要任何 Azure 憑證。**兩個 package 必須是 public**，
Container Apps 才拉得到 —— 這也是範本裡沒有 `registries` 區塊、沒有 pull identity、也沒有 ACR
資源的原因。第一次推完之後要到 GitHub 的 package 設定把可見度改成 public（package 預設是 private，
即使 repo 是 public）。

需要三個 repository secret，因為 Vite 在**建置時**就把 `VITE_*` 內嵌進 bundle（改動任何一個都要
重新建置，不是重啟）：`VITE_SUPABASE_URL`、`VITE_SUPABASE_ANON_KEY`、`VITE_LIFF_ID`。anon key 可以
安心隨 bundle 出貨，資料列存取由 Postgres RLS 控管；`SUPABASE_SERVICE_ROLE_KEY` **絕不可以**
出現在 build arg 裡。

backend 的 build context 是 **repo 根目錄**（它以絕對套件路徑 import `backend.*` 與 `src.*`）；
frontend 的是 `frontend/`。

### 第二趟：兩個 app

```bash
az deployment group create -g $RG -f infra/main.bicep \
    -p @infra/main.parameters.json \
    -p backendImage=ghcr.io/lucas6028/x-coach-backend:$TAG \
    -p frontendImage=ghcr.io/lucas6028/x-coach-frontend:$TAG \
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

### 部署完之後，三件不在 Azure 裡的事

這三件全都會表現成「這個功能在正式環境是死的」，但沒有一件是 Azure 的問題，log 裡也看不到。

1. **Supabase Auth 的 redirect URL。** 新的 `https://<fqdn>` 這個來源不在 Supabase 的
   Site URL / Redirect URLs 清單裡，登入會被彈回去。到 Authentication → URL Configuration 加上去。
2. **LIFF endpoint。** 要讓 LINE 那條路走得通，LIFF 的 endpoint 要指向新網址的**站台根目錄** —
   LIFF 深層連結是接在 endpoint 的完整路徑後面的，指到子路徑會整個錯位。LINE Messaging 的
   webhook URL 同理，指到 `https://<fqdn>/api/line/webhook`。
3. **Supabase migrations。** `db/migrations/` 底下的 SQL 是手動套的，跟這次部署完全沒有交集。
   確認最新那幾支（`20260813000000_training_plans.sql`、`20260725000000_analysis_movement.sql`）
   已經跑過，少了資料表前端就會壞。

另外值得當場看一眼的：前端 bundle 裡的 `VITE_*` 是**建置時**由 GitHub secret 內嵌的，如果哪個
secret 是空的，build 一樣會成功，只有登入會在執行期壞掉，而且伺服器端不會有任何錯誤。開頁面確認
SPA 真的連得到 Supabase，比讀 log 快。

### 後續部署

只有 image tag 會變：

```powershell
./infra/deploy.ps1 -Stage build
./infra/deploy.ps1 -Stage update
```

```bash
az containerapp update -n xcoach-backend  -g $RG --image ghcr.io/lucas6028/x-coach-backend:$TAG
az containerapp update -n xcoach-frontend -g $RG --image ghcr.io/lucas6028/x-coach-frontend:$TAG
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

## 自訂網域與 TLS：xcoach.dev

系統產生的 `*.azurecontainerapps.io` FQDN 本來就提供 HTTPS，所以在你想用自己的主機名稱之前，TLS
不需要做任何事。之後 Container Apps 會簽發一張**免費的受管憑證**（DigiCert）並自動續期 — 沒有
certbot 要跑。

只有 frontend 需要；backend 是內部的。

主機名稱**寫在範本裡**（`infra/main.parameters.json` 的 `customDomains`，目前是 `xcoach.dev` 與
`www.xcoach.dev`），不是用 `az containerapp hostname add` 另外加：範本每一趟都會整個覆寫
frontend 的 ingress，只存在 CLI 裡的主機名稱會在下一次 `-Stage apps` 被無聲地拿掉。憑證也是
範本管的（每個主機名稱一個 `managedCertificates` 資源）。

### 順序：DNS → 主機名稱 → 憑證

平台在**加上**主機名稱時就會驗證 `asuid.<host>` 的 TXT，而憑證又只能簽給 app **已經掛著**的主機
名稱，所以是三步，缺一步就會得到一個既不提 DNS 也不提記錄的 validation error：

1. **DNS 記錄。** `-Stage dns` 會印出下面這張表（值從環境本身讀出）。到 name.com 建，**取代**
   這些名稱現有的 A / CNAME（xcoach.dev 目前指向 Vercel）：

   | 類型  | 名稱                  | 值                                                 |
   | ----- | --------------------- | -------------------------------------------------- |
   | A     | `xcoach.dev`          | 環境的靜態 IP（輸出 `environmentStaticIp`）          |
   | TXT   | `asuid.xcoach.dev`    | 環境的 `customDomainVerificationId`                |
   | CNAME | `www.xcoach.dev`      | `xcoach-frontend.<環境 defaultDomain>`（輸出 `frontendDefaultFqdn`） |
   | TXT   | `asuid.www.xcoach.dev`| 同上的 verification id                              |

   根網域不能放 CNAME，所以 apex 走 A 記錄指向環境的靜態 IP；子網域走 CNAME 指向 frontend 的
   系統 FQDN。範本用同一條規則挑驗證方式：一個點的（apex）用 HTTP 驗證，其餘用 CNAME 驗證。
2. **掛上主機名稱（未繫結）。** `bindCertificates=false` 那一趟：主機名稱進到 ingress，
   `bindingType: Disabled`。
3. **簽發並繫結。** `bindCertificates=true` 那一趟：建立受管憑證，等 DigiCert 驗證通過（apex 是對
   app 本身做 HTTP 驗證，所以第 2 步一定要先上線），再把繫結切到 `SniEnabled`。通常幾分鐘。

`-Stage domain` 會先用 `Resolve-DnsName` 逐筆核對第 1 步，然後連跑第 2、3 步，並且用兩個 app
**現在跑的 image** 重新部署，不是 HEAD — 這一步改的是 ingress，不是程式碼。bash 版本：

```bash
# 第 1 步的值
az containerapp env show -n xcoach-env -g $RG \
    --query '{ip:properties.staticIp,domain:properties.defaultDomain,verify:properties.customDomainConfiguration.customDomainVerificationId}'

# 第 2、3 步：同「第二趟：兩個 app」那條指令，各加一個參數
az deployment group create ... -p bindCertificates=false
az deployment group create ... -p bindCertificates=true
```

**沒有 `.env` 的機器**（範本每一趟都要重送全部機密，不能空跑）改用下面三組指令，效果一樣，
但憑證名稱**必須**跟範本產生的一致（`mc-` 加上把 `.` 換成 `-` 的主機名稱），下一次 `-Stage apps`
才會把它們當成同一組資源而不是再簽一張：

```bash
az containerapp hostname add -n xcoach-frontend -g $RG --hostname xcoach.dev
az containerapp hostname add -n xcoach-frontend -g $RG --hostname www.xcoach.dev
az containerapp env certificate create -g $RG -n xcoach-env -c mc-xcoach-dev     --hostname xcoach.dev     --validation-method HTTP
az containerapp env certificate create -g $RG -n xcoach-env -c mc-www-xcoach-dev --hostname www.xcoach.dev --validation-method CNAME
# 等 `az containerapp env certificate list ... --managed-certificates-only` 兩張都 Succeeded
az containerapp hostname bind -n xcoach-frontend -g $RG --hostname xcoach.dev     --environment xcoach-env --certificate mc-xcoach-dev
az containerapp hostname bind -n xcoach-frontend -g $RG --hostname www.xcoach.dev --environment xcoach-env --certificate mc-www-xcoach-dev
```

2026-09-14 xcoach.dev 就是這樣掛上去的。

之後每一次 `-Stage apps` 都會先問 frontend 目前哪些主機名稱已經是 `SniEnabled`：全部都是就帶
`bindCertificates=true`（否則重新部署會把繫結降回 Disabled）；有任何一個不是 — 第一次部署，或
剛在參數檔加了新的主機名稱 — 就先核對 DNS、以未繫結的方式帶上去，並提醒你接著跑 `-Stage domain`。

兩件會讓簽發失敗的事：

- **CAA 記錄。** 如果根網域上存在任何 `CAA` 記錄，必須加上 `0 issue digicert.com`，否則簽發與
  續期都會失敗。（2026-09 檢查時 xcoach.dev 沒有 CAA。）
- **順序。** 見上。DNS 還沒生效就跑第 2 步，或 app 還沒公開可達就跑第 3 步，都會失敗。

`customDomains` 同時餵給 `XCOACH_CORS_ORIGINS`（每個主機名稱一個 `https://` 來源）。這只有在直接
跨來源呼叫 API 時才有意義（SPA 本身透過 nginx 是同源的），但設對了也不花什麼成本。

網域切過來之後，「部署完之後，三件不在 Azure 裡的事」那一節的前兩件要用 `https://xcoach.dev`
再做一次：Supabase 的 Site URL / Redirect URLs、LIFF endpoint、LINE webhook URL。Vercel 那邊的
專案不會自動知道網域已經搬走，把它從 Vercel 專案的 Domains 移掉，免得之後誤判。

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
