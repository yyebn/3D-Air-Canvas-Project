"""
Air Canvas 3D - v2 (One Euro Filter 적용)
==========================================
변경점:
  - One Euro filter로 3D 좌표 스무딩
  - x/y/z 각 축 독립 필터 (z축에 더 강한 스무딩)
  - 't' 키로 raw vs smoothed 궤적 비교 토글
  - 's' 키로 현재 궤적 PNG로 저장 (리포트 figure용)
  - 'r' 키 시점 리셋 강화

제스처:
  - 검지만 펼침         → drawing mode
  - 검지+중지 펼침      → pen up

3D 창 키:
  - 'c'                 → 모두 지우기
  - 'r'                 → 시점 완전 리셋
  - 't'                 → raw 궤적 표시 토글 (비교용)
  - 'q' / ESC           → 종료
  - (웹캠 창에서도 동일)
"""

import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import time
import math

# ============================================================
# 설정
# ============================================================
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
Z_AMPLIFY = 5.0
MAX_POINTS_PER_STROKE = 5000
MIN_POINT_DISTANCE = 0.003       # 너무 촘촘한 점 제거

# One Euro filter 파라미터 (튜닝 가능)
# x/y는 비교적 정확하니 약하게, z는 노이즈 많으니 강하게
ONE_EURO_PARAMS_XY = {"min_cutoff": 1.0, "beta": 0.05, "d_cutoff": 1.0}
ONE_EURO_PARAMS_Z  = {"min_cutoff": 0.3, "beta": 0.02, "d_cutoff": 1.0}


# ============================================================
# One Euro Filter 구현
# ============================================================
class OneEuroFilter:
    """
    Géry Casiez et al. (2012) "1€ Filter".
    저속에선 강한 lowpass, 고속에선 약한 lowpass로 자동 적응.
    """
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x
        dt = max(t - self.t_prev, 1e-6)
        # 미분값 추정
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # 적응적 cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        # 상태 갱신
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class Vec3OneEuro:
    """x/y/z 각각 독립 OneEuroFilter."""
    def __init__(self, params_xy, params_z):
        self.fx = OneEuroFilter(**params_xy)
        self.fy = OneEuroFilter(**params_xy)
        self.fz = OneEuroFilter(**params_z)

    def __call__(self, vec, t):
        return np.array([
            self.fx(vec[0], t),
            self.fy(vec[1], t),
            self.fz(vec[2], t),
        ])

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()


# ============================================================
# MediaPipe 초기화
# ============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


def is_finger_extended(landmarks, finger_name):
    tip = landmarks[TIP_IDS[finger_name]]
    pip = landmarks[PIP_IDS[finger_name]]
    return tip.y < pip.y


def classify_gesture(landmarks):
    idx = is_finger_extended(landmarks, "index")
    mid = is_finger_extended(landmarks, "middle")
    ring = is_finger_extended(landmarks, "ring")
    pinky = is_finger_extended(landmarks, "pinky")
    if idx and not mid and not ring and not pinky:
        return "draw"
    if idx and mid and not ring and not pinky:
        return "pen_up"
    return "other"


def landmark_to_3d(lm):
    x = lm.x - 0.5
    y = -(lm.y - 0.5)
    z = lm.z * Z_AMPLIFY
    return np.array([x, y, z])


# ============================================================
# Open3D 시각화 초기화
# ============================================================
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="3D Trajectory (v2)",
                  width=900, height=700,
                  left=100, top=100)

# 좌표축 (작게, 좌하단)
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
axis.translate([-0.5, -0.5, 0])
vis.add_geometry(axis)

# 데이터 구조
strokes_smooth = [[]]   # smoothed 궤적
strokes_raw = [[]]      # raw 궤적 (비교용)

# LineSet: smoothed 메인 궤적
line_set_smooth = o3d.geometry.LineSet()
vis.add_geometry(line_set_smooth)

# LineSet: raw 궤적 (토글로 표시)
line_set_raw = o3d.geometry.LineSet()
vis.add_geometry(line_set_raw)
show_raw = False  # 초기엔 raw 숨김

# 핑거팁 마커
finger_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
finger_marker.paint_uniform_color([1.0, 0.2, 0.2])
vis.add_geometry(finger_marker)
prev_marker_pos = np.array([0.0, 0.0, 0.0])

# 필터 인스턴스
euro = Vec3OneEuro(ONE_EURO_PARAMS_XY, ONE_EURO_PARAMS_Z)


def update_line_set_smooth():
    pts, lines, colors = [], [], []
    offset = 0
    for stroke in strokes_smooth:
        if len(stroke) < 2:
            offset += len(stroke)
            continue
        for i, p in enumerate(stroke):
            pts.append(p)
            if i > 0:
                lines.append([offset + i - 1, offset + i])
                z_norm = np.clip(p[2] / 0.5 + 0.5, 0, 1)
                colors.append([z_norm, 0.4, 1.0 - z_norm])
        offset += len(stroke)

    line_set_smooth.points = o3d.utility.Vector3dVector(
        np.array(pts) if pts else np.zeros((0, 3))
    )
    if lines:
        line_set_smooth.lines = o3d.utility.Vector2iVector(np.array(lines))
        line_set_smooth.colors = o3d.utility.Vector3dVector(np.array(colors))
    else:
        line_set_smooth.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=int))
        line_set_smooth.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))


def update_line_set_raw():
    pts, lines, colors = [], [], []
    offset = 0

    if show_raw:
        for stroke in strokes_raw:
            if len(stroke) < 2:
                offset += len(stroke)
                continue
            for i, p in enumerate(stroke):
                pts.append(p)
                if i > 0:
                    lines.append([offset + i - 1, offset + i])
                    colors.append([0.5, 0.5, 0.5])  # 회색 (대조)
            offset += len(stroke)

    line_set_raw.points = o3d.utility.Vector3dVector(
        np.array(pts) if pts else np.zeros((0, 3))
    )
    if lines:
        line_set_raw.lines = o3d.utility.Vector2iVector(np.array(lines))
        line_set_raw.colors = o3d.utility.Vector3dVector(np.array(colors))
    else:
        line_set_raw.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=int))
        line_set_raw.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))


# 초기 시점 저장 (리셋용)
def set_default_view():
    ctl = vis.get_view_control()
    ctl.set_front([0.0, 0.0, 1.0])
    ctl.set_up([0.0, 1.0, 0.0])
    ctl.set_lookat([0.0, 0.0, 0.0])
    ctl.set_zoom(0.6)


set_default_view()


# ---------- 키 콜백 ----------
def cb_clear(vis_obj):
    global strokes_smooth, strokes_raw
    strokes_smooth = [[]]
    strokes_raw = [[]]
    euro.reset()
    update_line_set_smooth()
    update_line_set_raw()
    vis_obj.update_geometry(line_set_smooth)
    vis_obj.update_geometry(line_set_raw)
    print("[clear]")
    return False


def cb_reset(vis_obj):
    set_default_view()
    print("[reset view]")
    return False


def cb_toggle_raw(vis_obj):
    global show_raw
    show_raw = not show_raw
    update_line_set_raw()
    vis_obj.update_geometry(line_set_raw)
    print(f"[toggle raw] show_raw={show_raw}")
    return False


def cb_save(vis_obj):
    ts = int(time.time())
    fname = f"trajectory_{ts}.png"
    vis_obj.capture_screen_image(fname, do_render=True)
    print(f"[saved] {fname}")
    return False


vis.register_key_callback(ord('C'), cb_clear)
vis.register_key_callback(ord('R'), cb_reset)
vis.register_key_callback(ord('T'), cb_toggle_raw)
vis.register_key_callback(ord('S'), cb_save)


# ============================================================
# 웹캠
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"카메라 {CAMERA_INDEX} 열기 실패")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)


# ============================================================
# 메인 루프
# ============================================================
prev_time = time.perf_counter()
fps = 0.0
need_update_smooth = False
need_update_raw = False

print("=" * 60)
print("Air Canvas 3D v2 - One Euro Filter")
print("=" * 60)
print("  검지만 펼침      → 그리기")
print("  검지+중지 펼침   → pen up")
print("  3D창 'C'         → 지우기")
print("  3D창 'R'         → 시점 리셋")
print("  3D창 'T'         → raw 궤적 토글")
print("  3D창 'S'         → 스크린샷 저장")
print("  'q' / ESC        → 종료")
print("=" * 60)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        gesture = "no_hand"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            gesture = classify_gesture(hand_landmarks.landmark)

            tip_lm = hand_landmarks.landmark[TIP_IDS["index"]]
            tip_raw = landmark_to_3d(tip_lm)
            tip_smooth = euro(tip_raw, now)

            # 마커는 smoothed 좌표로
            delta = tip_smooth - prev_marker_pos
            finger_marker.translate(delta)
            prev_marker_pos = tip_smooth.copy()
            vis.update_geometry(finger_marker)

            cx, cy = int(tip_lm.x * w), int(tip_lm.y * h)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)

            if gesture == "draw":
                cs_s = strokes_smooth[-1]
                cs_r = strokes_raw[-1]
                if len(cs_s) < MAX_POINTS_PER_STROKE:
                    if len(cs_s) == 0 or np.linalg.norm(tip_smooth - cs_s[-1]) > MIN_POINT_DISTANCE:
                        cs_s.append(tip_smooth.copy())
                        cs_r.append(tip_raw.copy())
                        need_update_smooth = True
                        if show_raw:
                            need_update_raw = True
            elif gesture == "pen_up":
                if len(strokes_smooth[-1]) > 0:
                    strokes_smooth.append([])
                    strokes_raw.append([])
                    euro.reset()  # 새 stroke마다 필터 초기화 → latency 감소
        else:
            # 손 사라지면 필터 리셋 (다시 잡혔을 때 점프 방지)
            euro.reset()

        # HUD
        mode_color = {
            "draw": (0, 255, 0), "pen_up": (0, 255, 255),
            "other": (180, 180, 180), "no_hand": (0, 0, 255),
        }.get(gesture, (255, 255, 255))
        total = sum(len(s) for s in strokes_smooth)
        info = [
            f"FPS: {fps:5.1f}",
            f"Mode: {gesture}",
            f"Strokes: {len(strokes_smooth)}  Pts: {total}",
            f"Show raw: {show_raw} (3D win: T)",
        ]
        for i, t in enumerate(info):
            y = 30 + i * 30
            cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        cv2.imshow("Webcam", frame)

        if need_update_smooth:
            update_line_set_smooth()
            vis.update_geometry(line_set_smooth)
            need_update_smooth = False
        if need_update_raw:
            update_line_set_raw()
            vis.update_geometry(line_set_raw)
            need_update_raw = False

        if not vis.poll_events():
            break
        vis.update_renderer()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    hands.close()
    print("종료됨.")
