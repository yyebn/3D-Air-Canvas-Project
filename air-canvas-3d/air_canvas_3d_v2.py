"""
Air Canvas 3D - v2 (One Euro Filter 적용)
==========================================
변경점:
  - One Euro filter로 3D 좌표 스무딩
  - x/y/z 각 축 독립 필터 (z축에 더 강한 스무딩)
  - 't' 키로 raw vs smoothed 궤적 비교 토글
  - 's' 키로 현재 궤적 PNG로 저장 (리포트 figure용)
  - 'r' 키 시점 리셋 강화
  - 브러시 스타일: 8색 팔레트 순환(B키) 색상 모드
  - 브러시 스타일은 3D 궤적에만 반영 (웹캠에는 비표시)
  - smoothed 궤적 렌더링을 LineSet -> Cylinder Mesh로 변경
    (macOS에서도 두께 변화가 눈에 보이도록 개선)
  - 세그먼트별 두께 저장: '['/']'는 이후에 그리는 선에만 적용
  - 세그먼트별 색상 저장: 'B'는 이후에 그리는 선에만 적용
  - Backspace undo: 제스처와 무관하게 고정 픽셀 길이만큼 tail 삭제

제스처:
  - 검지만 펼침         → drawing mode
  - 검지+중지 펼침      → pen up

3D 창 키:
  - 'c'                 → 모두 지우기
  - 'r'                 → 시점 완전 리셋
  - 't'                 → raw 궤적 표시 토글 (비교용)
  - 's'                 → 현재 3D 뷰 PNG 저장
  - 'b'                 → 브러시 색상 전환 (빨주노초파남보/검정)
  - '[' / ']'           → 3D 선 두께 감소/증가
  - Backspace           → 고정 픽셀 길이 undo
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

# Brush 스타일
BRUSH_PALETTE = [
    ("Red",    [1.0, 0.0, 0.0]),
    ("Orange", [1.0, 0.5, 0.0]),
    ("Yellow", [1.0, 1.0, 0.0]),
    ("Green",  [0.0, 1.0, 0.0]),
    ("Blue",   [0.0, 0.2, 1.0]),
    ("Navy",   [0.1, 0.1, 0.55]),
    ("Violet", [0.56, 0.0, 1.0]),
    ("Black",  [0.02, 0.02, 0.02]),
]
brush_color_index = 0
LINE_WIDTH_MIN = 1.0
LINE_WIDTH_MAX = 30.0
LINE_WIDTH_STEP = 1.0
line_width = 3.0
LINE_WIDTH_TO_RADIUS = 0.0009    # line_width -> cylinder radius 변환
CYLINDER_RESOLUTION = 10
ERASE_PIXELS_PER_PRESS = 180.0   # Backspace 1회당 지울 누적 픽셀 길이

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
strokes_px = [[]]       # webcam 픽셀 궤적 (undo 길이 계산용)
strokes_ts = [[]]       # 포인트 timestamp
strokes_seg_metrics = [[]]  # 각 segment 메타데이터 [{"speed":.., "width":.., "color":[r,g,b]}, ...]

# smoothed 메인 궤적 (두께 표현용 triangle mesh)
smooth_mesh = None

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


def map_metric_to_rgb(metric):
    """segment에 저장된 색상을 RGB로 반환."""
    return np.array(metric["color"], dtype=float)


def _rotation_from_z_to_vec(vec):
    """z축([0,0,1])을 vec 방향으로 회전시키는 3x3 행렬."""
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    v = vec / max(np.linalg.norm(vec), 1e-12)
    c = np.clip(np.dot(z_axis, v), -1.0, 1.0)

    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        # z축과 반대 방향: x축 기준 180도 회전
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

    axis = np.cross(z_axis, v)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    s = math.sqrt(max(1.0 - c * c, 0.0))
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _line_width_to_radius(width_value):
    return float(width_value) * LINE_WIDTH_TO_RADIUS


def _build_smooth_mesh():
    mesh = o3d.geometry.TriangleMesh()

    for stroke, seg_meta in zip(strokes_smooth, strokes_seg_metrics):
        if len(stroke) < 2:
            continue
        for i in range(1, len(stroke)):
            p0 = np.asarray(stroke[i - 1], dtype=float)
            p1 = np.asarray(stroke[i], dtype=float)
            seg = p1 - p0
            length = np.linalg.norm(seg)
            if length < 1e-8:
                continue

            cyl = o3d.geometry.TriangleMesh.create_cylinder(
                radius=_line_width_to_radius(seg_meta[i - 1]["width"]),
                height=float(length),
                resolution=CYLINDER_RESOLUTION,
                split=1,
            )
            cyl.paint_uniform_color(map_metric_to_rgb(seg_meta[i - 1]).tolist())
            cyl.rotate(_rotation_from_z_to_vec(seg), center=np.zeros(3))
            cyl.translate((p0 + p1) * 0.5)
            mesh += cyl

    if len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()
    return mesh


def refresh_smooth_mesh(vis_obj):
    global smooth_mesh
    new_mesh = _build_smooth_mesh()
    if smooth_mesh is not None:
        vis_obj.remove_geometry(smooth_mesh, reset_bounding_box=False)
    smooth_mesh = new_mesh
    vis_obj.add_geometry(smooth_mesh, reset_bounding_box=False)


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


def apply_line_width(vis_obj):
    # 현재 brush 두께(line_width)는 "이후 segment"에 적용됨.
    # 화면 동기화를 위해 갱신은 유지.
    refresh_smooth_mesh(vis_obj)


def _compact_trailing_empty_strokes():
    """말단의 빈 stroke를 정리하되 최소 1개는 유지."""
    global strokes_smooth, strokes_raw, strokes_px, strokes_ts, strokes_seg_metrics
    while (
        len(strokes_smooth) > 1
        and len(strokes_smooth[-1]) == 0
        and len(strokes_smooth[-2]) == 0
    ):
        strokes_smooth.pop()
        strokes_raw.pop()
        strokes_px.pop()
        strokes_ts.pop()
        strokes_seg_metrics.pop()


def _ensure_nonempty_container():
    """모든 stroke가 비어 있으면 기본 빈 stroke 1개로 초기화."""
    global strokes_smooth, strokes_raw, strokes_px, strokes_ts, strokes_seg_metrics
    if all(len(s) == 0 for s in strokes_smooth):
        strokes_smooth = [[]]
        strokes_raw = [[]]
        strokes_px = [[]]
        strokes_ts = [[]]
        strokes_seg_metrics = [[]]


def cb_undo_backspace(vis_obj):
    """
    Backspace 한 번당 tail에서 일정 픽셀 길이만큼 삭제.
    손가락 제스처와 무관하게 동작.
    """
    remaining = ERASE_PIXELS_PER_PRESS
    removed_any = False

    while remaining > 0:
        idx = -1
        for i in range(len(strokes_smooth) - 1, -1, -1):
            if len(strokes_smooth[i]) > 0:
                idx = i
                break
        if idx < 0:
            break

        ss = strokes_smooth[idx]
        rr = strokes_raw[idx]
        pp = strokes_px[idx]
        tt = strokes_ts[idx]
        mm = strokes_seg_metrics[idx]

        if len(ss) == 1:
            ss.pop()
            rr.pop()
            pp.pop()
            tt.pop()
            mm.clear()
            remaining -= 1.0
            removed_any = True
            continue

        # 마지막 segment 길이(픽셀)
        seg_len = float(np.linalg.norm(np.array(pp[-1]) - np.array(pp[-2])))
        ss.pop()
        rr.pop()
        pp.pop()
        tt.pop()
        if mm:
            mm.pop()
        remaining -= max(seg_len, 1.0)
        removed_any = True

    if removed_any:
        _compact_trailing_empty_strokes()
        _ensure_nonempty_container()
        refresh_smooth_mesh(vis_obj)
        update_line_set_raw()
        vis_obj.update_geometry(line_set_raw)
        print(f"[undo] erased ~{ERASE_PIXELS_PER_PRESS:.0f}px")
    else:
        print("[undo] nothing to erase")
    return False


# ---------- 키 콜백 ----------
def cb_clear(vis_obj):
    global strokes_smooth, strokes_raw, strokes_px, strokes_ts, strokes_seg_metrics
    strokes_smooth = [[]]
    strokes_raw = [[]]
    strokes_px = [[]]
    strokes_ts = [[]]
    strokes_seg_metrics = [[]]
    euro.reset()
    refresh_smooth_mesh(vis_obj)
    update_line_set_raw()
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


def cb_toggle_brush_color(vis_obj):
    global brush_color_index
    brush_color_index = (brush_color_index + 1) % len(BRUSH_PALETTE)
    name, _ = BRUSH_PALETTE[brush_color_index]
    # 기존 segment는 색상 고정이라 그대로 유지. 새로 그리는 선부터 적용.
    print(f"[brush color] {name} (next segments)")
    return False


def cb_line_width_up(vis_obj):
    global line_width
    line_width = min(LINE_WIDTH_MAX, line_width + LINE_WIDTH_STEP)
    apply_line_width(vis_obj)
    print(f"[line width] {line_width:.1f}  (next segments)")
    return False


def cb_line_width_down(vis_obj):
    global line_width
    line_width = max(LINE_WIDTH_MIN, line_width - LINE_WIDTH_STEP)
    apply_line_width(vis_obj)
    print(f"[line width] {line_width:.1f}  (next segments)")
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
vis.register_key_callback(ord('B'), cb_toggle_brush_color)
vis.register_key_callback(ord(']'), cb_line_width_up)
vis.register_key_callback(ord('['), cb_line_width_down)
# GLFW KEY_BACKSPACE = 259, KEY_DELETE = 261
vis.register_key_callback(259, cb_undo_backspace)
vis.register_key_callback(261, cb_undo_backspace)
apply_line_width(vis)


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
print("  3D창 'B'         → 브러시 색상 전환 (8 colors)")
print("  3D창 '[' ']'     → 다음에 그릴 3D 선 두께 조절")
print(f"  Backspace/Delete  → 약 {ERASE_PIXELS_PER_PRESS:.0f}px undo")
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
                cs_p = strokes_px[-1]
                cs_t = strokes_ts[-1]
                cs_m = strokes_seg_metrics[-1]
                if len(cs_s) < MAX_POINTS_PER_STROKE:
                    if len(cs_s) == 0 or np.linalg.norm(tip_smooth - cs_s[-1]) > MIN_POINT_DISTANCE:
                        if len(cs_s) > 0:
                            dt = max(now - cs_t[-1], 1e-6)
                            speed = np.linalg.norm(tip_smooth - cs_s[-1]) / dt
                            _, rgb = BRUSH_PALETTE[brush_color_index]
                            seg_metric = {
                                "speed": float(speed),
                                "width": float(line_width),
                                "color": list(rgb),
                            }
                            cs_m.append(seg_metric)
                        cs_s.append(tip_smooth.copy())
                        cs_r.append(tip_raw.copy())
                        cs_p.append((cx, cy))
                        cs_t.append(now)
                        need_update_smooth = True
                        if show_raw:
                            need_update_raw = True
            elif gesture == "pen_up":
                if len(strokes_smooth[-1]) > 0:
                    strokes_smooth.append([])
                    strokes_raw.append([])
                    strokes_px.append([])
                    strokes_ts.append([])
                    strokes_seg_metrics.append([])
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
            f"Brush width(next): {line_width:.1f}",
            f"Brush color(next): {BRUSH_PALETTE[brush_color_index][0]} (B)",
            f"Undo range: {ERASE_PIXELS_PER_PRESS:.0f}px (Backspace)",
        ]
        for i, t in enumerate(info):
            y = 30 + i * 30
            cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        cv2.imshow("Webcam", frame)

        if need_update_smooth:
            refresh_smooth_mesh(vis)
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
        elif key == ord('b'):
            cb_toggle_brush_color(vis)
        elif key == ord(']'):
            cb_line_width_up(vis)
        elif key == ord('['):
            cb_line_width_down(vis)
        elif key in (8, 127):
            cb_undo_backspace(vis)

finally:
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    hands.close()
    print("종료됨.")
