import cv2
import numpy as np
import pyrealsense2 as rs

MARKER_ID = 1
MARKER_LENGTH = 0.16

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# ===================== Подключение камеры =====================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

profile = pipeline.start(config)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

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
    print("Наведите камеру на маркер.")
    print("Когда маркер хорошо виден — нажмите ПРОБЕЛ для захвата.")
    print("ESC — выход\n")

    captured = False
    while not captured:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        corners, ids, _ = detector.detectMarkers(color_image)
        display = color_image.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners)

            if MARKER_ID in ids.flatten():
                idx = np.where(ids.flatten() == MARKER_ID)[0][0]
                marker_corners = corners[idx]

                # === Новая оценка позы через solvePnP ===
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
                    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.6)

        cv2.imshow(f"Live — Позиция {pos} (Пробел = захват, ESC = выход)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:   # Пробел
            if ids is not None and MARKER_ID in ids.flatten() and success:
                captured = True
                poses[pos] = {"rvec": rvec.copy(), "tvec": tvec.copy()}
                print(f"✓ Позиция {pos} захвачена (tvec = {tvec.flatten().round(4)})")
            else:
                print("Маркер не обнаружен или solvePnP не удался. Попробуйте снова.")
        elif key == 27:  # ESC
            print("Выход из программы.")
            pipeline.stop()
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyAllWindows()

pipeline.stop()

# ===================== Расчёт трансформации =====================
def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

T_marker_to_camA = pose_to_matrix(poses["A"]["rvec"], poses["A"]["tvec"])
T_marker_to_camB = pose_to_matrix(poses["B"]["rvec"], poses["B"]["tvec"])

T_camB_to_camA = T_marker_to_camA @ np.linalg.inv(T_marker_to_camB)

print("\nМатрица T_camB_to_camA:")
print(np.round(T_camB_to_camA, decimals=6))

np.save("T_camB_to_camA.npy", T_camB_to_camA)
print("\nФайл сохранён: T_camB_to_camA.npy")