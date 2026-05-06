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
        # MOG2 背景相減器：設定變極度遲鈍 (varThreshold=100) 以抵抗光線閃爍
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg["motion"].get("history", 300),
            varThreshold=cfg["motion"].get("var_threshold", 100), 
            detectShadows=False
        )
        self.fire_count = 0
        self.smoke_count = 0
        self.last_alarm_time = 0

    def _get_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def detect_fire(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

        lower1 = np.array(self.cfg["fire"]["hsv_lower1"], dtype=np.uint8)
        upper1 = np.array(self.cfg["fire"]["hsv_upper1"], dtype=np.uint8)
        lower2 = np.array(self.cfg["fire"]["hsv_lower2"], dtype=np.uint8)
        upper2 = np.array(self.cfg["fire"]["hsv_upper2"], dtype=np.uint8)

        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )

        y, cr, cb = cv2.split(ycrcb)
        ycrcb_mask = ((y > self.cfg["fire"]["min_y"]) &
                      (cr > self.cfg["fire"]["min_cr"]) &
                      (cb < self.cfg["fire"]["max_cb"])).astype(np.uint8) * 255

        mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        
        # 形態學去噪與黏合
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        kernel_close = np.ones((11, 11), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        contours = self._get_contours(mask)
        valid_contours = []
        total_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.cfg["fire"]["min_area"]:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / max(h, 1)
                if 0.2 < aspect_ratio < 4.0:
                    valid_contours.append(c)
                    total_area += area
        
        return len(valid_contours) > 0, mask, valid_contours, total_area

    def detect_smoke(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        motion = self.bg.apply(frame_bgr)
        _, motion = cv2.threshold(motion, 200, 255, cv2.THRESH_BINARY)

        smoke_color = ((s < self.cfg["smoke"]["max_saturation"]) &
                       (v > self.cfg["smoke"]["min_value"]) &
                       (v < self.cfg["smoke"]["max_value"])).astype(np.uint8) * 255

        mask = cv2.bitwise_and(smoke_color, motion)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours = self._get_contours(mask)
        valid_contours = []
        total_area = 0
        avg_blur = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.cfg["smoke"]["min_area"]:
                x, y, w, h_box = cv2.boundingRect(c)
                aspect_ratio = float(w) / max(h_box, 1)

                if 0.1 < aspect_ratio < 5.0:
                    roi_gray = gray[max(0, y):y+h_box, max(0, x):x+w]
                    if roi_gray.size > 0:
                        blur_val = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                        
                        if blur_val <= self.cfg["smoke"]["max_laplacian_var"]:
                            valid_contours.append(c)
                            total_area += area
                            avg_blur += blur_val

        if valid_contours:
            avg_blur /= len(valid_contours)

        return len(valid_contours) > 0, mask, valid_contours, total_area, avg_blur

    def update_alarm_state(self, fire_now, smoke_now):
        self.fire_count = self.fire_count + 1 if fire_now else 0
        self.smoke_count = self.smoke_count + 1 if smoke_now else 0

        fire_alarm = self.fire_count >= self.cfg["alarm"]["fire_consecutive_frames"]
        smoke_alarm = self.smoke_count >= self.cfg["alarm"]["smoke_consecutive_frames"]

        now = time.time()
        alarm = (fire_alarm or smoke_alarm) and (now - self.last_alarm_time > self.cfg["alarm"]["cooldown_sec"])
        if alarm:
            self.last_alarm_time = now
        return alarm, fire_alarm, smoke_alarm

def draw_contours(frame, contours, color, label):
    if not contours:
        return

    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    valid_count = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
        valid_count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

    if valid_count > 0:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
        cv2.putText(frame, f"{label} ZONE", (x_min, max(20, y_min - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="設定檔路徑")
    parser.add_argument("--source", default="0", help="攝影機代號或影片檔案路徑")
    parser.add_argument("--no-display", action="store_true", help="關閉畫面顯示")
    parser.add_argument("--save-video", default="", help="輸出錄影檔案路徑")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        # --- 加入這段測試印出 ---
        print("===" * 15)
        print("✅ 成功讀取 config.json！目前設定值如下：")
        print(f"煙霧模糊度門檻 (max_laplacian_var): {cfg['smoke']['max_laplacian_var']}")
        print(f"火焰最小面積 (min_area): {cfg['fire']['min_area']}")
        print(f"掃描區域 (ROI): {cfg['camera'].get('roi', '全畫面')}")
        print("===" * 15)
        # ------------------------
    except FileNotFoundError:
        print(f"錯誤：找不到設定檔 {args.config}，請確認檔案存在。")
        return

    cam_width = cfg["camera"].get("width", 640)
    cam_height = cfg["camera"].get("height", 480)
    fps_setting = cfg["camera"].get("fps", 15)
    
    # 取得 ROI 範圍，如果 config 沒寫，預設掃描全畫面
    roi_box = cfg["camera"].get("roi", [0, 0, cam_width, cam_height])
    rx1, ry1, rx2, ry2 = roi_box
    
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, fps_setting)

    if not cap.isOpened():
        print(f"錯誤：無法開啟影像來源 {args.source}。")
        return

    detector = FireSmokeDetector(cfg)
    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.save_video, fourcc, fps_setting, (cam_width, cam_height))

    prev_time = time.time()
    frame_id = 0
    process_every_n_frames = cfg["camera"].get("frame_skip", 2) 
    
    last_fire_contours, last_smoke_contours = [], []
    fire_area, smoke_area, smoke_blur = 0, 0, 0
    fire_alarm, smoke_alarm = False, False

    print("系統啟動，初始化背景中...")
    
    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.resize(frame, (cam_width, cam_height))
        frame_id += 1
        display = frame.copy()

        # ==========================================
        # 【核心新增】建立 ROI 遮罩 (Region of Interest)
        # ==========================================
        # 建立一個全黑的遮罩
        roi_mask = np.zeros((cam_height, cam_width), dtype=np.uint8)
        # 把火爐的區域塗成白色 (255)
        cv2.rectangle(roi_mask, (rx1, ry1), (rx2, ry2), 255, -1)
        # 套用遮罩：保留框內畫面，框外全黑 (這樣外圍的石頭反光就完全消失了)
        frame_roi = cv2.bitwise_and(frame, frame, mask=roi_mask)
        # ==========================================

        # 注意這裡：把「已經塗黑外圍」的 frame_roi 交給 detector 辨識
        if frame_id % process_every_n_frames == 0:
            fire_now, fire_mask, last_fire_contours, fire_area = detector.detect_fire(frame_roi)
            smoke_now, smoke_mask, last_smoke_contours, smoke_area, smoke_blur = detector.detect_smoke(frame_roi)
            alarm, fire_alarm, smoke_alarm = detector.update_alarm_state(fire_now, smoke_now)
            
            if alarm:
                status = "FIRE ALARM" if fire_alarm else "SMOKE ALARM"
                print(f"[ALARM] 影格={frame_id}, 狀態={status}, 火焰面積={fire_area:.0f}, 煙霧面積={smoke_area:.0f}")

        # 畫圖時，我們畫在原本的 display 上，這樣畫面還是彩色的，但辨識區只在中央
        # 畫出 ROI 的藍色掃描框，讓你知道系統現在只看哪裡
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(display, "Scan Area", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        draw_contours(display, last_fire_contours, (0, 0, 255), "FIRE")
        draw_contours(display, last_smoke_contours, (200, 200, 200), "SMOKE")

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        status_text = "SAFE"
        color = (0, 255, 0)
        if fire_alarm:
            status_text, color = "FIRE ALARM", (0, 0, 255)
        elif smoke_alarm:
            status_text, color = "SMOKE ALARM", (0, 165, 255)

        cv2.putText(display, f"{status_text} | FPS:{fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        debug_text = f"F_Area:{fire_area:.0f} | S_Area:{smoke_area:.0f} | S_Blur:{smoke_blur:.0f}"
        cv2.putText(display, debug_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if out is not None:
            out.write(display)

        if not args.no_display:
            cv2.imshow("Fire & Smoke Detection", display)
            # 你可以打開這行，看看系統眼中的畫面是不是「外圍全黑」
            # cv2.imshow("ROI View (What System Sees)", frame_roi)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()