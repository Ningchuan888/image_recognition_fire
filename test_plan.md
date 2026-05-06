# 驗證計畫

| 測試項目 | 測試方法 | 預期結果 |
|---|---|---|
| 相機開啟 | `python main.py --source 0` | 可以看到即時畫面 |
| 火焰偵測 | 用安全測試影片或小範圍火焰影片 | FIRE ALARM |
| 煙霧偵測 | 使用煙霧測試影片 | SMOKE ALARM |
| 燈光誤判 | 手電筒、日光燈、螢幕亮光 | 不應連續觸發警報 |
| 白色物體誤判 | 白紙、白牆移動 | 不應觸發煙霧警報 |
| 低效能測試 | Raspberry Pi 3 執行 640x480, 15 FPS | 可穩定執行 |
| 長時間測試 | 連續執行 30 分鐘 | 不當機、警報冷卻正常 |

## 調參建議
- 火焰太敏感：提高 `fire.min_area` 或 `fire.min_cr`
- 火焰偵測不到：降低 `fire.min_area` 或 `fire.min_y`
- 煙霧太敏感：提高 `smoke.min_area` 或降低 `smoke.max_laplacian_var`
- 煙霧偵測不到：降低 `smoke.min_area` 或提高 `smoke.max_saturation`
