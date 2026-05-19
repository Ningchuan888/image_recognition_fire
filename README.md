# Raspberry Pi 4 火焰／煙霧偵測專題

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
| 目標平台 | Raspberry Pi 4 或同級設備 |
| 解析度 | 640 × 480 |
| 目標幀率 | 15 FPS |
| 依賴套件 | `opencv-python`、`numpy` |
| 語言 | Python 3 |

### Bonus 目標

- 不同光影變化下仍能正確偵測（降低誤判）
- 效能優化，達到目標 FPS@resolution

---

## 分析

### breakdown

<img width="10852" height="820" alt="mermaid-diagram-2026-05-19-164627" src="https://github.com/user-attachments/assets/2d540cca-8862-4ad1-83a2-98a293c3b015" />



### 誤判來源分析

| 誤判來源 | 應對方式 |
|---|---|
| 日光燈、手電筒 | YCrCb 條件篩除高亮但低 Cr 的光源 |
| 白牆、白紙移動 | 煙霧需同時滿足動態（MOG2）＋低飽和條件 |
| 噪點、短暫閃爍 | 連續幀計數（fire: 4 幀、smoke: 12 幀）才觸發 |

---

## 設計

### 環境圖（Context Diagram）

<img width="1915" height="652" alt="mermaid-diagram-2026-05-06-142224" src="https://github.com/user-attachments/assets/84cc89cf-417c-4b73-a20e-d28b3383c8bd" />


### 偵測流程

<img width="3620" height="3854" alt="mermaid-diagram-2026-05-19-161703" src="https://github.com/user-attachments/assets/18ea30cd-e219-49e2-a240-2d84deeb2729" />


### 警報狀態機（FSM）

<img width="1038" height="716" alt="mermaid-diagram-2026-05-06-142411" src="https://github.com/user-attachments/assets/dff504f9-0f7e-4249-9bd7-5aa79230d6f0" />



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
