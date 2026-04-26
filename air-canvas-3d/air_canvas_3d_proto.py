"""
Air Canvas 3D - Prototype
=========================
검지로 허공에 그린 궤적을 3D로 실시간 시각화.

사용법:
    python air_canvas_3d_proto.py

    제스처:
      - 검지만 펼침         → drawing mode (3D 궤적 기록)
      - 검지+중지 펼침      → pen up (그리기 멈춤)

    키보드:
      - 'c'                 → 현재 궤적 모두 지우기
      - 'r'                 → 3D 뷰 리셋
      - 'q' 또는 ESC       → 종료

창:
    - "Webcam"      : 손 스켈레톤 + 현재 모드 표시
    - "3D Trajectory": Open3D 3D 뷰어 (마우스 드래그로 회전, 휠로 줌)

설계 메모:
    - z축은 시각화 시 Z_AMPLIFY배 증폭 → 입체감 강조
    - One Euro filter는 다음 단계에서 추가 (지금은 raw 좌표)
    - 궤적은 LineSet으로 렌더링 (점들을 순서대로 연결)
"""

import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import time
from collections import deque

# ============================================================
# 설정
# ============================================================
CAMERA_INDEX = 1                # 맥북 내장 카메라 (Continuity 회피)
FRAME_W, FRAME_H = 1280, 720
Z_AMPLIFY = 5.0                 # z 시각 증폭 계수
MAX_POINTS_PER_STROKE = 5000    # 한 획 최대 점 수 (메모리 보호)
FINGER_BENT_THRESHOLD = 0.0     # tip이 PIP보다 위(작은 y)면 펴짐

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

# 손가락 펴짐 판단을 위한 landmark 인덱스
TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
PIP_IDS = {
    "thumb": 2,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}


def is_finger_extended(landmarks, finger_name):
    """
    손가락이 펴졌는지 여부.
    tip의 y가 pip의 y보다 작으면 (화면 위쪽이면) 펴진 것으로 판단.
    엄지는 좌우 좌표로 판단해야 정확하지만, 본 프로토타입에선 검지/중지만 봄.
    """
    tip = landmarks[TIP_IDS[finger_name]]
    pip = landmarks[PIP_IDS[finger_name]]
    return tip.y < pip.y


def classify_gesture(landmarks):
    """
    간단한 rule-based 제스처 분류.
    반환:
        'draw'   : 검지만 펴짐
        'pen_up' : 검지+중지 펴짐
        'other'  : 그 외
    """
    idx = is_finger_extended(landmarks, "index")
    mid = is_finger_extended(landmarks, "middle")
    ring = is_finger_extended(landmarks, "ring")
    pinky = is_finger_extended(landmarks, "pinky")

    if idx and not mid and not ring and not pinky:
        return "draw"
    if idx and mid and not ring and not pinky:
        return "pen_up"
    return "other"


# ============================================================
# Open3D 시각화 초기화
# ============================================================
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="3D Trajectory",
                  width=900, height=700,
                  left=100, top=100)

# 좌표축 (작게, 좌하단 구석에 위치)
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
axis.translate([-0.5, -0.5, 0])
vis.add_geometry(axis)

# 궤적 데이터 (여러 stroke 지원)
# 각 stroke는 점 리스트
strokes = [[]]   # 현재 stroke은 strokes[-1]

# Open3D LineSet (전체 궤적 표시용)
line_set = o3d.geometry.LineSet()
vis.add_geometry(line_set)

# 현재 핑거팁 마커 (작은 빨간 구)
finger_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
finger_marker.paint_uniform_color([1.0, 0.2, 0.2])
finger_marker.translate([0, 0, 0])
vis.add_geometry(finger_marker)
prev_marker_pos = np.array([0.0, 0.0, 0.0])

# 카메라 뷰 초기 설정
view_ctl = vis.get_view_control()
view_ctl.set_zoom(0.7)


# 3D 창에서도 'c'/'r' 키 작동하도록 콜백 등록
def _clear_cb(vis_obj):
    global strokes
    strokes = [[]]
    update_line_set()
    vis_obj.update_geometry(line_set)
    print("[clear] all strokes removed")
    return False


def _reset_view_cb(vis_obj):
    vis_obj.get_view_control().set_zoom(0.7)
    print("[reset view]")
    return False


vis.register_key_callback(ord('C'), _clear_cb)
vis.register_key_callback(ord('R'), _reset_view_cb)


def update_line_set():
    """모든 stroke를 하나의 LineSet으로 합쳐서 갱신."""
    all_points = []
    all_lines = []
    all_colors = []
    point_offset = 0

    for stroke in strokes:
        if len(stroke) < 2:
            point_offset += len(stroke)
            continue
        for i, p in enumerate(stroke):
            all_points.append(p)
            if i > 0:
                all_lines.append([point_offset + i - 1, point_offset + i])
                # z를 색상으로도 표현 (z 양수→빨강, 음수→파랑)
                z_norm = np.clip(p[2] / 0.5 + 0.5, 0, 1)
                all_colors.append([z_norm, 0.4, 1.0 - z_norm])
        point_offset += len(stroke)

    if all_points:
        line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
    else:
        line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))

    if all_lines:
        line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines))
        line_set.colors = o3d.utility.Vector3dVector(np.array(all_colors))
    else:
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=int))
        line_set.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))


def landmark_to_3d(lm):
    """
    MediaPipe landmark (x, y, z) → Open3D 좌표계.
    - x: 화면 좌→우, [0, 1] → 좌표계 [-0.5, 0.5]
    - y: 화면 위→아래, [0, 1] → 좌표계 [+0.5, -0.5] (y 반전)
    - z: 손목 기준 상대 depth, Z_AMPLIFY배 증폭
    """
    x = lm.x - 0.5
    y = -(lm.y - 0.5)
    z = lm.z * Z_AMPLIFY
    return np.array([x, y, z])


# ============================================================
# 웹캠 초기화
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"카메라 {CAMERA_INDEX}를 열 수 없음. CAMERA_INDEX 변경 시도.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# ============================================================
# 메인 루프
# ============================================================
prev_time = time.perf_counter()
fps = 0.0
need_geometry_update = False

print("=" * 60)
print("Air Canvas 3D - Prototype")
print("=" * 60)
print("  검지만 펼침      → 그리기")
print("  검지+중지 펼침   → pen up")
print("  'c'              → 지우기")
print("  'r'              → 3D 뷰 리셋")
print("  'q' / ESC        → 종료")
print("=" * 60)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임 읽기 실패")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        # FPS
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        gesture = "no_hand"
        tip_3d = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            gesture = classify_gesture(hand_landmarks.landmark)

            # 검지 끝 3D 좌표
            tip_lm = hand_landmarks.landmark[TIP_IDS["index"]]
            tip_3d = landmark_to_3d(tip_lm)

            # 마커 이동
            delta = tip_3d - prev_marker_pos
            finger_marker.translate(delta)
            prev_marker_pos = tip_3d.copy()
            vis.update_geometry(finger_marker)

            # 검지 끝 화면 픽셀 좌표
            cx, cy = int(tip_lm.x * w), int(tip_lm.y * h)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)

            # 그리기 모드 처리
            if gesture == "draw":
                current_stroke = strokes[-1]
                if len(current_stroke) < MAX_POINTS_PER_STROKE:
                    # 직전 점과 너무 가까우면 skip (중복 방지)
                    if len(current_stroke) == 0 or \
                       np.linalg.norm(tip_3d - current_stroke[-1]) > 0.005:
                        current_stroke.append(tip_3d.copy())
                        need_geometry_update = True
            elif gesture == "pen_up":
                # 새 stroke 시작 (현재 stroke가 비어있지 않을 때만)
                if len(strokes[-1]) > 0:
                    strokes.append([])

        # ---------- HUD ----------
        mode_color = {
            "draw": (0, 255, 0),
            "pen_up": (0, 255, 255),
            "other": (180, 180, 180),
            "no_hand": (0, 0, 255),
        }.get(gesture, (255, 255, 255))

        total_pts = sum(len(s) for s in strokes)
        info = [
            f"FPS: {fps:5.1f}",
            f"Mode: {gesture}",
            f"Strokes: {len(strokes)}  Points: {total_pts}",
        ]
        for i, t in enumerate(info):
            y = 30 + i * 32
            cv2.putText(frame, t, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(frame, t, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

        cv2.imshow("Webcam", frame)

        # ---------- Open3D 갱신 ----------
        if need_geometry_update:
            update_line_set()
            vis.update_geometry(line_set)
            need_geometry_update = False

        if not vis.poll_events():
            # 3D 창 닫혔을 때 종료
            print("3D viewer closed.")
            break
        vis.update_renderer()

        # ---------- 키 입력 ----------
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            strokes = [[]]
            update_line_set()
            vis.update_geometry(line_set)
            print("[clear] all strokes removed")
        elif key == ord('r'):
            view_ctl = vis.get_view_control()
            view_ctl.set_zoom(0.7)
            print("[reset view]")

finally:
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    hands.close()
    print("종료됨.")
