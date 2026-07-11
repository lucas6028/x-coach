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

## 接進前端（下一步，尚未做）

1. **Favicon / PWA**：把 `icons/` 對應檔放進 `frontend/public/`，於 `index.html` 加
   `<link rel="icon" href="/favicon.ico">`、`<link rel="apple-touch-icon" href="/lumen-icon-navy-180.png">`，
   並在 `manifest.webmanifest` 註冊 192／512（含 `"purpose":"maskable"` 指向 maskable 版）。
2. **載入動畫**：把 `web-full.png` 複製成 `frontend/public/lumen/lumen-full.png`，
   引入 `LumenLoader.tsx`，注入一次 `lumenLoaderCss`，即可 `<LumenLoader variant="scan" />`。

## 重新生成／擴充

所有圖由 FLUX.1-dev 生成（見 memory `nim-image-gen-recipe`）。prompt 骨架在 `gen_set.py`；
新表情沿用同 seed（主 33／貼紙 7）即維持一致。App icon 重出：
`python gen_icon.py --build`（讀 `set-C-genki.png`，輸出到 `icons/`）。
