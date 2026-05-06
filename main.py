import cv2
import time
import argparse
import json
from pathlib import Path
import numpy as np


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FireSmokeDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg["motion"]["history"],
            varThreshold=cfg["motion"]["var_threshold"],
            detectShadows=True
        )
        self.fire_count = 0
        self.smoke_count = 0
        self.last_alarm_time = 0

    def _largest_area(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, []
        areas = [cv2.contourArea(c) for c in contours]
        return max(areas), contours

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

        # 火焰常見條件：亮度高、紅色/橘色分量高。可降低燈光、白牆誤判。
        y, cr, cb = cv2.split(ycrcb)
        ycrcb_mask = ((y > self.cfg["fire"]["min_y"]) &
                      (cr > self.cfg["fire"]["min_cr"]) &
                      (cb < self.cfg["fire"]["max_cb"])).astype(np.uint8) * 255

        mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        area, contours = self._largest_area(mask)
        return area >= self.cfg["fire"]["min_area"], mask, contours, area

    def detect_smoke(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        motion = self.bg.apply(frame_bgr)
        _, motion = cv2.threshold(motion, 200, 255, cv2.THRESH_BINARY)

        # 煙霧通常低飽和、灰白、邊界模糊且有緩慢移動。
        smoke_color = ((s < self.cfg["smoke"]["max_saturation"]) &
                       (v > self.cfg["smoke"]["min_value"]) &
                       (v < self.cfg["smoke"]["max_value"])).astype(np.uint8) * 255

        # 排除太銳利/太黑的物體。
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        mask = cv2.bitwise_and(smoke_color, motion)
        mask = cv2.medianBlur(mask, 7)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        area, contours = self._largest_area(mask)
        is_smoke = (area >= self.cfg["smoke"]["min_area"] and
                    blur <= self.cfg["smoke"]["max_laplacian_var"])
        return is_smoke, mask, contours, area, blur

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
    for c in contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def open_capture(source, width, height, fps):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--source", default="0", help="0 for webcam, or video path")
    parser.add_argument("--no-display", action="store_true", help="for Raspberry Pi headless mode")
    parser.add_argument("--save-video", default="", help="optional output mp4 path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cap = open_capture(args.source, cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fps"])

    if not cap.isOpened():
        raise RuntimeError("無法開啟相機或影片來源，請確認 --source 是否正確。")

    detector = FireSmokeDetector(cfg)
    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            args.save_video, fourcc, cfg["camera"]["fps"],
            (cfg["camera"]["width"], cfg["camera"]["height"])
        )

    prev = time.time()
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (cfg["camera"]["width"], cfg["camera"]["height"]))
        frame_id += 1

        fire_now, fire_mask, fire_contours, fire_area = detector.detect_fire(frame)
        smoke_now, smoke_mask, smoke_contours, smoke_area, smoke_blur = detector.detect_smoke(frame)
        alarm, fire_alarm, smoke_alarm = detector.update_alarm_state(fire_now, smoke_now)

        display = frame.copy()
        draw_contours(display, fire_contours, (0, 0, 255), "FIRE")
        draw_contours(display, smoke_contours, (180, 180, 180), "SMOKE")

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        status = "SAFE"
        if fire_alarm:
            status = "FIRE ALARM"
        elif smoke_alarm:
            status = "SMOKE ALARM"

        cv2.putText(display, f"{status} | FPS:{fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0) if status == "SAFE" else (0, 0, 255), 2)
        cv2.putText(display, f"fire_area={fire_area:.0f} smoke_area={smoke_area:.0f} blur={smoke_blur:.1f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if alarm:
            print(f"[ALARM] frame={frame_id}, status={status}, fire_area={fire_area:.0f}, smoke_area={smoke_area:.0f}")

        if out is not None:
            out.write(display)

        if not args.no_display:
            cv2.imshow("Fire/Smoke Detection", display)
            cv2.imshow("fire_mask", fire_mask)
            cv2.imshow("smoke_mask", smoke_mask)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
