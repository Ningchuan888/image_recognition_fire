# Raspberry Pi 3 火焰／煙霧偵測專題

## 需求
- 使用相機即時偵測火焰與煙霧。
- 可先在電腦端以 VSCode + Python 測試。
- 之後移植到 Raspberry Pi 3 執行。
- 需降低燈光、車燈、反光、白牆、陰影造成的誤判。
- Pi 3 效能有限，因此採用 OpenCV 傳統影像處理，不使用大型 AI 模型。

## 安裝

### 電腦端
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py --source 0
```

若測試影片：
```bash
python main.py --source test_fire.mp4
```

### Raspberry Pi 3
```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy
python3 main.py --source 0 --no-display
```

若有桌面畫面：
```bash
python3 main.py --source 0
```

## 檔案說明
- `main.py`：主程式
- `config.json`：門檻值設定
- `requirements.txt`：電腦端套件
- `test_plan.md`：驗證表
