# Raspberry Pi 3 火焰／煙霧偵測專題

## 目錄

1. [需求](#需求)
2. [分析](#分析)
3. [設計](#設計)
4. [程式結構](#程式結構)
5. [驗證計畫](#驗證計畫)
6. [參數調整](#參數調整)

---

## 需求

### 功能需求

| 項目 | 說明 |
|---|---|
| 輸入 | 影片檔案（`.mp4` 等）或即時相機串流 |
| 輸出 | 畫面上的火焰 / 煙霧 Bounding Box，並顯示警報狀態 |
| 警報機制 | 連續多幀偵測到後觸發警報，具冷卻時間避免重複警報 |
| 無頭模式 | 支援 `--no-display` 於無螢幕的 Raspberry Pi 上執行 |
| 影片輸出 | 可選擇將標註結果儲存為 `.mp4` |

### 規格需求

| 項目 | 規格 |
|---|---|
| 目標平台 | Raspberry Pi 4B 或同級設備 |
| 解析度 | 640 × 480 |
| 目標幀率 | 15 FPS |
| 依賴套件 | `opencv-python`、`numpy` |
| 語言 | Python 3 |

### Bonus 目標

- 不同光影變化下仍能正確偵測（降低誤判）
- 效能優化，達到目標 FPS@resolution

---

## 分析

### 大題拆解

```
火焰 / 煙霧偵測
├── 火焰偵測
│   ├── 顏色特徵（HSV：紅橘色範圍）
│   ├── 亮度 / 色度條件（YCrCb：高亮度、高 Cr、低 Cb）
│   └── 形態學後處理（去雜訊、填補）
├── 煙霧偵測
│   ├── 顏色特徵（低飽和、灰白色）
│   ├── 動態背景分離（MOG2 背景消去）
│   └── 邊緣模糊度判斷（Laplacian 變異數）
└── 警報狀態機
    ├── 連續幀計數
    ├── 警報觸發門檻
    └── 冷卻時間控制
```

### 誤判來源分析

| 誤判來源 | 應對方式 |
|---|---|
| 日光燈、手電筒 | YCrCb 條件篩除高亮但低 Cr 的光源 |
| 白牆、白紙移動 | 煙霧需同時滿足動態（MOG2）＋低飽和條件 |
| 噪點、短暫閃爍 | 連續幀計數（fire: 4 幀、smoke: 12 幀）才觸發 |

---

## 設計

### 環境圖（Context Diagram）

```
[相機 / 影片] ──→ [ FireSmokeDetector ] ──→ [畫面顯示 / 影片輸出]
       ↑                  ↑                          ↓
  config.json         MOG2 背景模型               [警報輸出 (console)]
```

### 偵測流程

```
讀取影片幀
    │
    ├─→ detect_fire()
    │       ├── BGR → HSV → 顏色遮罩
    │       ├── BGR → YCrCb → 亮度/色度遮罩
    │       ├── AND 合併 → 形態學後處理
    │       └── 找輪廓 → 判斷面積 ≥ min_area
    │
    ├─→ detect_smoke()
    │       ├── BGR → HSV → 低飽和遮罩
    │       ├── MOG2 → 動態遮罩
    │       ├── AND 合併 → 形態學後處理
    │       └── 面積 ≥ min_area AND Laplacian ≤ max_laplacian_var
    │
    └─→ update_alarm_state()
            ├── 連續幀計數
            ├── 超過門檻 → fire_alarm / smoke_alarm
            └── 冷卻時間檢查 → 輸出 alarm
```

### 警報狀態機（FSM）

```
[SAFE] ──(連續偵測 ≥ 門檻)──→ [ALARM]
   ↑                                │
   └────── (冷卻時間到 + 無偵測) ───┘
```

| 狀態 | 說明 |
|---|---|
| `SAFE` | 未偵測到火焰或煙霧 |
| `FIRE ALARM` | 連續 ≥ 4 幀偵測到火焰 |
| `SMOKE ALARM` | 連續 ≥ 12 幀偵測到煙霧 |

### 核心 API（模組介面）

| 方法 | 輸入 | 輸出 |
|---|---|---|
| `detect_fire(frame_bgr)` | BGR 影像幀 | `(is_fire, mask, contours, area)` |
| `detect_smoke(frame_bgr)` | BGR 影像幀 | `(is_smoke, mask, contours, area, blur)` |
| `update_alarm_state(fire, smoke)` | 當前幀偵測結果 | `(alarm, fire_alarm, smoke_alarm)` |

---

## 程式結構

```
.
├── main.py          # 主程式：讀取影片、驅動偵測、顯示結果
├── config.json      # 所有可調參數（HSV 範圍、面積門檻、警報設定等）
├── requirements.txt # 依賴套件
└── README.md        # 本文件
```

### config.json 參數說明

```jsonc
{
  "camera":  { "width": 640, "height": 480, "fps": 15 },
  "fire": {
    "hsv_lower1/upper1": "紅橘色 HSV 範圍（低段）",
    "hsv_lower2/upper2": "紅色 HSV 範圍（高段，處理 Hue 環繞）",
    "min_y":  "YCrCb 最低亮度",
    "min_cr": "最低紅色分量",
    "max_cb": "最高藍色分量",
    "min_area": "最小有效輪廓面積（像素²）"
  },
  "smoke": {
    "max_saturation": "煙霧最大飽和度",
    "min/max_value":  "灰白亮度範圍",
    "min_area":       "最小有效輪廓面積",
    "max_laplacian_var": "最大銳利度（超過代表非煙霧）"
  },
  "motion": { "history": 300, "var_threshold": 25 },
  "alarm": {
    "fire_consecutive_frames":  "觸發火焰警報所需連續幀數",
    "smoke_consecutive_frames": "觸發煙霧警報所需連續幀數",
    "cooldown_sec": "兩次警報間的最短間隔（秒）"
  }
}
```

---

## 驗證計畫

| 測試項目 | 測試方法 | 預期結果 |
|---|---|---|
| 相機開啟 | `python main.py --source 0` | 可看到即時畫面 |
| 火焰偵測 | 使用安全測試影片或小範圍火焰影片 | 顯示 `FIRE ALARM` |
| 煙霧偵測 | 使用煙霧測試影片 | 顯示 `SMOKE ALARM` |
| 燈光誤判 | 手電筒、日光燈、螢幕亮光 | **不應**連續觸發警報 |
| 白色物體誤判 | 白紙、白牆移動 | **不應**觸發煙霧警報 |
| 低效能測試 | Raspberry Pi 4B 執行 640×480 @ 15 FPS | 可穩定執行 |
| 長時間測試 | 連續執行 30 分鐘 | 不當機、警報冷卻正常 |

---


## 參數調整

| 問題 | 調整方式 |
|---|---|
| 火焰太敏感（誤報多） | 提高 `fire.min_area` 或 `fire.min_cr` |
| 火焰偵測不到 | 降低 `fire.min_area` 或 `fire.min_y` |
| 煙霧太敏感（誤報多） | 提高 `smoke.min_area` 或降低 `smoke.max_laplacian_var` |
| 煙霧偵測不到 | 降低 `smoke.min_area` 或提高 `smoke.max_saturation` |
| 警報太頻繁 | 提高 `alarm.cooldown_sec` |
| 警報反應太慢 | 降低 `alarm.fire_consecutive_frames` |
