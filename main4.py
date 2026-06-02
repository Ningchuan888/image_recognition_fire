import cv2
import time
import argparse
import json
import numpy as np

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FireSmokeDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fire_count = 0
        self.smoke_count = 0
        self.last_alarm_time = 0

        # 光流用：前一幀灰階
        self.prev_gray = None

        # MOG2 用於煙霧輔助，去除靜態高光背景
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg["motion"].get("history", 300),
            varThreshold=cfg["motion"].get("var_threshold", 100),
            detectShadows=False
        )

    # ──────────────────────────────────────────
    # 工具：取輪廓
    # ──────────────────────────────────────────
    def _get_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # ──────────────────────────────────────────
    # 工具：Farneback 光流 → 動態強度圖與角度圖
    # ──────────────────────────────────────────
    def _optical_flow_features(self, gray):
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            return np.zeros_like(gray), np.zeros(gray.shape, dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        self.prev_gray = gray.copy()

        # angleInDegrees=True 讓回傳的角度為 0~360 度
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        mag = np.clip(mag * 20, 0, 255).astype(np.uint8)   # 放大後截斷
        return mag, ang

    # ──────────────────────────────────────────
    # 火焰偵測
    # ──────────────────────────────────────────
    def detect_fire(self, frame_bgr):
        fc = self.cfg["fire"]

        # 1. 高斯模糊
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

        # 2. 顏色遮罩
        hsv    = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ycrcb  = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv,
                        np.array(fc["hsv_lower1"], dtype=np.uint8),
                        np.array(fc["hsv_upper1"], dtype=np.uint8)),
            cv2.inRange(hsv,
                        np.array(fc["hsv_lower2"], dtype=np.uint8),
                        np.array(fc["hsv_upper2"], dtype=np.uint8))
        )

        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
        ycrcb_mask = (
            (y_ch  > fc["min_y"])  &
            (cr_ch > fc["min_cr"]) &
            (cb_ch < fc["max_cb"])
        ).astype(np.uint8) * 255

        color_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

        # 3. Canny 邊緣（在模糊後的灰階圖上做）
        gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges   = cv2.Canny(gray,
                            fc.get("canny_low",  50),
                            fc.get("canny_high", 150))
        # 膨脹讓邊緣變厚，方便與顏色遮罩重疊
        edges   = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

        # 4. 光流幅度遮罩（火焰只需要看大小，不用看方向）
        mag, _  = self._optical_flow_features(gray)
        flow_th = fc.get("flow_threshold", 15)
        _, flow_mask = cv2.threshold(mag, flow_th, 255, cv2.THRESH_BINARY)

        # 5. 合併：顏色 AND 邊緣 AND 光流
        mask = cv2.bitwise_and(color_mask, edges)
        mask = cv2.bitwise_and(mask, flow_mask)

        # 6. 形態學：先開（去小雜點）再閉（填補內部空洞）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,  3),  np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        # 7. 輪廓過濾（面積 + 長寬比）
        contours = self._get_contours(mask)
        valid, total_area = [], 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < fc["min_area"]:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = float(w) / max(h, 1)
            if 0.2 < ar < 4.0:
                valid.append(c)
                total_area += area

        return len(valid) > 0, mask, valid, total_area

    # ──────────────────────────────────────────
    # 煙霧偵測 (優化版)
    # ──────────────────────────────────────────
    def detect_smoke(self, frame_bgr):
        sc = self.cfg["smoke"]
        h_frame, w_frame = frame_bgr.shape[:2]

        # 1. 高斯模糊
        blurred = cv2.GaussianBlur(frame_bgr, (9, 9), 0)
        gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 2. 啟用 MOG2 動態前景遮罩 (去除靜態高光背景)
        fg_mask = self.bg.apply(blurred)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY) 

        # 3. 顏色條件（低飽和、中高亮度）
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        _, s_ch, v_ch = cv2.split(hsv)
        color_mask = (
            (s_ch < sc["max_saturation"]) &
            (v_ch > sc["min_value"])      &
            (v_ch < sc["max_value"])
        ).astype(np.uint8) * 255

        # 4. 光流特徵 (大小 + 方向)
        mag, ang = self._optical_flow_features(gray)
        flow_lo  = sc.get("flow_low",  3)    
        flow_hi  = sc.get("flow_high", 40)   
        
        # 4.1 光流強度遮罩
        flow_mag_mask = ((mag > flow_lo) & (mag < flow_hi)).astype(np.uint8) * 255
        
        # 4.2 光流方向遮罩 (影像 Y 軸向下，270度為正上方)
        min_ang = sc.get("angle_low", 210)
        max_ang = sc.get("angle_high", 330)
        flow_dir_mask = ((ang > min_ang) & (ang < max_ang)).astype(np.uint8) * 255

        # 5. Canny 邊緣密度「低」判定
        edges      = cv2.Canny(gray, sc.get("canny_low", 30), sc.get("canny_high", 80))
        edges_d    = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
        not_sharp  = cv2.bitwise_not(edges_d)

        # 6. 嚴格合併條件：動態前景 + 顏色相符 + 光流強度夠 + 光流方向向上 + 邊緣不銳利
        mask = cv2.bitwise_and(fg_mask, color_mask)
        mask = cv2.bitwise_and(mask, flow_mag_mask)
        mask = cv2.bitwise_and(mask, flow_dir_mask)
        mask = cv2.bitwise_and(mask, not_sharp)

        # 7. 形態學過濾雜訊
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 8. 輪廓過濾
        contours = self._get_contours(mask)
        valid, total_area, avg_blur = [], 0, 0.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < sc["min_area"]:
                continue
            x, y, w, h_box = cv2.boundingRect(c)
            ar = float(w) / max(h_box, 1)
            if not (0.1 < ar < 5.0):
                continue

            # Laplacian 再次確認模糊度
            roi = gray[max(0, y):y + h_box, max(0, x):x + w]
            if roi.size == 0:
                continue
            blur_val = cv2.Laplacian(roi, cv2.CV_64F).var()
            
            # 加入下限(例如 15)，過濾掉變異數接近 0 的「曝光純白死白區塊」
            if 15 <= blur_val <= sc["max_laplacian_var"]:
                valid.append(c)
                total_area += area
                avg_blur   += blur_val

        if valid:
            avg_blur /= len(valid)

        return len(valid) > 0, mask, valid, total_area, avg_blur

    # ──────────────────────────────────────────
    # 警報狀態機
    # ──────────────────────────────────────────
    def update_alarm_state(self, fire_now, smoke_now):
        self.fire_count  = self.fire_count  + 1 if fire_now  else 0
        self.smoke_count = self.smoke_count + 1 if smoke_now else 0

        fire_alarm  = self.fire_count  >= self.cfg["alarm"]["fire_consecutive_frames"]
        smoke_alarm = self.smoke_count >= self.cfg["alarm"]["smoke_consecutive_frames"]

        now   = time.time()
        alarm = (fire_alarm or smoke_alarm) and \
                (now - self.last_alarm_time > self.cfg["alarm"]["cooldown_sec"])
        if alarm:
            self.last_alarm_time = now
        return alarm, fire_alarm, smoke_alarm


# ──────────────────────────────────────────
# 畫 Bounding Box
# ──────────────────────────────────────────
def draw_contours(frame, contours, color, label):
    if not contours:
        return
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x);       y_min = min(y_min, y)
        x_max = max(x_max, x + w);   y_max = max(y_max, y + h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
    cv2.putText(frame, f"{label} ZONE",
                (x_min, max(20, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ──────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.json")
    parser.add_argument("--source",     default="0")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-video", default="")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        print("=" * 45)
        print("✅ 成功讀取 config.json")
        print(f"   煙霧模糊度門檻 : 15 ~ {cfg['smoke']['max_laplacian_var']}")
        print(f"   火焰最小面積   : {cfg['fire']['min_area']}")
        print(f"   ROI            : {cfg['camera'].get('roi', '全畫面')}")
        print("=" * 45)
    except FileNotFoundError:
        print(f"找不到 {args.config}")
        return

    cam_w   = cfg["camera"].get("width",  640)
    cam_h   = cfg["camera"].get("height", 480)
    fps_set = cfg["camera"].get("fps",     15)
    roi_box = cfg["camera"].get("roi", [0, 0, cam_w, cam_h])
    rx1, ry1, rx2, ry2 = roi_box

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
    cap.set(cv2.CAP_PROP_FPS,          fps_set)

    if not cap.isOpened():
        print(f"無法開啟影像來源：{args.source}")
        return

    detector