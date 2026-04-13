import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

# ===================== НАСТРОЙКИ =====================
MARKER_ID = 1                    # ID твоего ArUco маркера
MARKER_LENGTH = 0.16            # размер маркера в метрах (например 20 см)

# Известная позиция маркера в системе мира (PyBullet / стенд)
# ←←← ИЗМЕНИ ЭТИ ЗНАЧЕНИЯ ПОД СВОЙ СТЕНД ←←←
MARKER_WORLD_POS = np.array([0.0, 0.0, 0.5])   # центр маркера в мире
MARKER_WORLD_ROT = np.eye(3)                    # маркер лежит ровно

OUTPUT_PATH = Path("T_cam_to_world.npy")
# ====================================================

# ===================== ПОДКЛЮЧЕНИЕ КАМЕРЫ =====================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

profile = pipeline.start(config)

# Получаем реальные параметры камеры (intrinsics)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)

print(f"Камера запущена. Разрешение: {intr.width}x{intr.height}")
print(f"Фокусное расстояние: fx={intr.fx:.1f}, fy={intr.fy:.1f}")

# ===================== ArUco =====================
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector = cv2.aruco.ArucoDetector(dictionary)

print("\n=== Калибровка камера → мир ===")
print("Положите ArUco маркер ровно в известное место на столе.")
print("Наведите камеру и нажмите ПРОБЕЛ, когда маркер хорошо виден.\n")

captured = False
while not captured:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    display = color_image.copy()

    corners, ids, _ = detector.detectMarkers(color_image)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

        if MARKER_ID in ids.flatten():
            idx = np.where(ids.flatten() == MARKER_ID)[0][0]
            marker_corners = corners[idx]

            # Оценка позы маркера относительно камеры
            obj_points = np.array([
                [-MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
                [ MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
                [ MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
                [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_points, marker_corners, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH)

    cv2.imshow("Калибровка камера → мир (Пробел = захват, ESC = выход)", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # Пробел
        if success and MARKER_ID in ids.flatten():
            captured = True
            print(f"✓ Захват успешен! tvec = {tvec.flatten().round(4)}")
        else:
            print("Маркер не найден или solvePnP не удался. Попробуйте снова.")
    elif key == 27:  # ESC
        print("Выход.")
        pipeline.stop()
        cv2.destroyAllWindows()
        exit()

pipeline.stop()
cv2.destroyAllWindows()

# ===================== РАСЧЁТ МАТРИЦЫ T_cam_to_world =====================
R_cam_to_marker, _ = cv2.Rodrigues(rvec)

T_cam_to_marker = np.eye(4, dtype=np.float32)
T_cam_to_marker[:3, :3] = R_cam_to_marker
T_cam_to_marker[:3, 3] = tvec.flatten()

T_marker_to_world = np.eye(4, dtype=np.float32)
T_marker_to_world[:3, :3] = MARKER_WORLD_ROT
T_marker_to_world[:3, 3] = MARKER_WORLD_POS

# Итоговая матрица
T_cam_to_world = T_marker_to_world @ np.linalg.inv(T_cam_to_marker)

print("\nМатрица T_cam_to_world успешно вычислена:")
print(np.round(T_cam_to_world, decimals=6))

np.save(OUTPUT_PATH, T_cam_to_world)
print(f"\nФайл сохранён: {OUTPUT_PATH}")