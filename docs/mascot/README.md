# Lumen 吉祥物素材

x-coach AI 教練吉祥物 **Lumen**（火焰精靈）的定稿素材與生成腳本。
角色設定完整版見 `lumen.html`（用瀏覽器開）。

## 內容

| 檔案 | 說明 |
|---|---|
| `lumen.html` | 角色設定書 v3（形象、表情組、姿態、貼紙、語氣守則、視覺規格） |
| `lumen-icon-loader.html` | App icon 套組＋載入動畫展示頁 |
| `set-C-*.png` | 主風格「圓潤立體（軟 3D）」表情／姿態（seed 33） |
| `set-A-*.png` | 副風格「軟萌貼紙」表情（seed 7） |
| `icons/` | App icon 各底色×各尺寸、favicon.ico、去背 cutout |
| `LumenLoader.tsx` | 載入動畫的 lift-ready React 元件（尚未接進 app） |
| `gen_*.py` | 生成腳本（FLUX.1-dev via NVIDIA NIM；讀 repo 根 `.env` 的 LLM_API_KEY） |

## App icon（`icons/`）

裁切自「元氣」主視覺去背後合成。三種底：

- `lumen-icon-navy-*.png` — 深藍，識別首選（iOS 主圖標）
- `lumen-icon-cream-*.png` — 奶油，淺色情境
- `lumen-icon-maskable-*.png` — Android／PWA maskable（安全區滿版、去圓角）

尺寸：512 / 192 / 180 / 120 / 32；另有 `favicon.ico`（16–64 多解析度）與
透明去背 `lumen-head.png`、`lumen-full.png`、`web-full.png`（載入動畫用）。

## 接進前端

- ✅ **載入動畫 ＋ 教練面板（2026-07-11 完成）**：`web-full.png`／`web-head.png` 已複製到
  `frontend/public/lumen/`；正式元件在 `frontend/src/components/LumenLoader.tsx`（CSS 移進
  `index.css`，非本資料夾版本的內嵌 `lumenLoaderCss`）。分析等待態（`DemoIntro`，`loading`）
  改用 `<LumenLoader variant="scan" caption={statusMsg} />`；AI 教練已改名 **Lumen**，
  `CoachTray` 的標題／登入提示／每則回答署名用 `LumenAvatar`，思考中用 `variant="dots"`。
  金色只作 Lumen 識別（頭像環、光暈），teal `primary` 仍是操作色。
- ⏳ **Favicon / PWA（尚未做）**：把 `icons/` 對應檔放進 `frontend/public/`，於 `index.html` 加
  `<link rel="icon" href="/favicon.ico">`、`<link rel="apple-touch-icon" href="/lumen-icon-navy-180.png">`，
  並在 `manifest.webmanifest` 註冊 192／512（含 `"purpose":"maskable"` 指向 maskable 版）。

> 註：本資料夾版的 `LumenLoader.tsx` 是 lift-ready 範本；實際接入版在 `frontend/src/components/`。

## 重新生成／擴充

所有圖由 FLUX.1-dev 生成（見 memory `nim-image-gen-recipe`）。prompt 骨架在 `gen_set.py`；
新表情沿用同 seed（主 33／貼紙 7）即維持一致。App icon 重出：
`python gen_icon.py --build`（讀 `set-C-genki.png`，輸出到 `icons/`）。
