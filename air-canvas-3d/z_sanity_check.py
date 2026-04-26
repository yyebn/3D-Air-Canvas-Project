"""
MediaPipe Hands z-coordinate sanity check
=========================================
목적:
    - MediaPipe가 출력하는 검지 끝(landmark 8)의 (x, y, z) 값을 실시간 확인
    - 손을 앞뒤로 움직였을 때 z가 의미 있게 변하는지 검증
    - 3D Air Canvas 프로젝트의 go/no-go 판단용

사용법:
    python z_sanity_check.py

    - 's' 키: 현재 프레임의 값을 "기준점"으로 기록 (reset)
    - 'q' 또는 ESC 키: 종료

화면 표시:
    - 검지 끝에 초록 점
    - 좌측 상단: 현재 (x, y, z) 값
    - 좌측 상단: 기준점 대비 Δz (앞뒤 움직임 크기)
    - z값은 손목 기준 상대적 depth (단위: landmark의 x 스케일과 유사)

판단 기준:
    손을 카메라 앞뒤로 천천히 움직였을 때 Δz가
    ±0.1 이상 변하면 → z 신호 충분함 → 3D 프로젝트 GO
    거의 안 변하거나 (±0.02 이하) 노이즈만 보이면 → 대안 검토 필요
"""

import cv2
import mediapipe as mp
import time

# ---------- MediaPipe 초기화 ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,          # 한 손만 추적 (속도+정확도)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ---------- 웹캠 초기화 ----------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 카메라 권한을 확인하세요.")

# 해상도 설정 (너무 높으면 느려짐)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------- 상태 변수 ----------
baseline_z = None          # 기준점 z값 ('s' 키로 설정)
z_history = []             # 최근 z값 기록 (노이즈 관찰용)
MAX_HISTORY = 60           # 약 2초 분량 (30fps 기준)

# FPS 측정
prev_time = time.perf_counter()
fps = 0.0

print("=" * 60)
print("MediaPipe Z-Coordinate Sanity Check")
print("=" * 60)
print("조작법:")
print("  's' 키: 현재 z를 기준점으로 저장")
print("  'q' 또는 ESC: 종료")
print()
print("테스트 방법:")
print("  1) 손을 평소 위치에 두고 's' 눌러 기준 설정")
print("  2) 손을 카메라 쪽으로 가까이 (30cm → 15cm)")
print("  3) Δz 값 관찰")
print("  4) 손을 뒤로 멀리 (30cm → 60cm)")
print("  5) Δz 값 다시 관찰")
print()
print("기대 결과: 앞으로 갈수록 z가 음수로, 뒤로 갈수록 양수로")
print("(또는 반대 — 중요한 건 '변화가 있는가')")
print("=" * 60)

while True:
    ok, frame = cap.read()
    if not ok:
        print("프레임 읽기 실패")
        break

    # 거울 모드 (사용자 편의)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # BGR → RGB (MediaPipe 요구사항)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    # FPS 계산
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    # ---------- 손 검출되었을 때 ----------
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # 전체 스켈레톤 그리기
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        # landmark 8 = 검지 끝 (INDEX_FINGER_TIP)
        tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x_norm, y_norm, z_norm = tip.x, tip.y, tip.z

        # 픽셀 좌표 (x, y는 정규화된 0~1 값)
        cx, cy = int(x_norm * w), int(y_norm * h)

        # 검지 끝에 초록 점
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # z 히스토리 업데이트
        z_history.append(z_norm)
        if len(z_history) > MAX_HISTORY:
            z_history.pop(0)

        # ---------- HUD 텍스트 ----------
        lines = [
            f"FPS: {fps:5.1f}",
            f"x: {x_norm:+.3f}  y: {y_norm:+.3f}",
            f"z: {z_norm:+.4f}  (pixel: {cx}, {cy})",
        ]

        if baseline_z is not None:
            dz = z_norm - baseline_z
            lines.append(f"Delta z (vs baseline): {dz:+.4f}")

            # 판단 힌트
            if abs(dz) > 0.10:
                hint = "STRONG signal"
                color = (0, 255, 0)
            elif abs(dz) > 0.03:
                hint = "moderate signal"
                color = (0, 255, 255)
            else:
                hint = "weak/no signal"
                color = (0, 0, 255)
            lines.append(f"Signal: {hint}")
        else:
            color = (255, 255, 255)
            lines.append("press 's' to set baseline")

        # z 노이즈 수준 (최근 기록의 표준편차)
        if len(z_history) >= 10:
            import statistics
            z_std = statistics.stdev(z_history[-30:]) if len(z_history) >= 30 else 0.0
            lines.append(f"z noise (std of last 30): {z_std:.4f}")

        # 텍스트 렌더링
        for i, text in enumerate(lines):
            y = 30 + i * 28
            # 검은 외곽선
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            # 본문
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    else:
        cv2.putText(frame, "No hand detected - show your hand to camera",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("MediaPipe Z Sanity Check", frame)

    # ---------- 키 입력 ----------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' or ESC
        break
    elif key == ord('s'):
        if results.multi_hand_landmarks:
            tip = results.multi_hand_landmarks[0].landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]
            baseline_z = tip.z
            print(f"[baseline set] z = {baseline_z:+.4f}")
        else:
            print("[baseline] 손이 검출되지 않음. 손을 보이게 한 뒤 다시 's'")

# ---------- 정리 ----------
cap.release()
cv2.destroyAllWindows()
hands.close()
print("종료됨.")
