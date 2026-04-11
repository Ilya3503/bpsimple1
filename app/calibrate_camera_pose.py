import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

MARKER_ID = 1
MARKER_LENGTH = 0.20          # ← ОБЯЗАТЕЛЬНО ИЗМЕРЬ реальный размер чёрного квадрата в метрах!
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Подключение RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

profile = pipeline.start(config)

color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()

camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)

detector = cv2.aruco.ArucoDetector(DICTIONARY)

poses = {}

for pos in ["A", "B"]:
    print(f"\n=== ПОЗИЦИЯ {pos} ===")
    input("Поставь камеру в нужное положение и нажми ENTER...")

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    corners, ids, _ = detector.detectMarkers(color_image)

    if ids is None or MARKER_ID not in ids.flatten():
        print(f"Ошибка: Маркер не обнаружен в позиции {pos}")
        pipeline.stop()
        exit(1)

    idx = np.where(ids.flatten() == MARKER_ID)[0][0]
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners[idx:idx+1], MARKER_LENGTH, camera_matrix, dist_coeffs
    )

    rvec, tvec = rvecs[0][0], tvecs[0][0]

    # Визуальная проверка
    cv2.aruco.drawDetectedMarkers(color_image, corners[idx:idx+1])
    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.6)
    cv2.imshow(f"Проверка позиции {pos}", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    poses[pos] = {"rvec": rvec.copy(), "tvec": tvec.copy()}
    print(f"Позиция {pos} сохранена (tvec = {tvec.round(4)})")

pipeline.stop()

# ===================== Расчёт трансформации =====================
def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

T_marker_to_camA = pose_to_matrix(poses["A"]["rvec"], poses["A"]["tvec"])
T_marker_to_camB = pose_to_matrix(poses["B"]["rvec"], poses["B"]["tvec"])

T_camB_to_camA = T_marker_to_camA @ np.linalg.inv(T_marker_to_camB)

print("\nМатрица трансформации T_camB_to_camA:")
print(np.round(T_camB_to_camA, decimals=6))

# Сохранение
np.save("T_camB_to_camA.npy", T_camB_to_camA)
print("\nФайл сохранён: T_camB_to_camA.npy")