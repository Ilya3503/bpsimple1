import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path

def calibrate_cam_to_world():
    """
    Скрипт для получения матрицы T_camA_to_world с помощью ArUco маркера.
    """
    print("=== Калибровка камеры относительно мира (стенда) ===\n")

    # ================== НАСТРОЙКИ (поменяй под себя) ==================
    ARUCO_DICT = aruco.DICT_6X6_250
    MARKER_SIZE = 0.16       # размер маркера в метрах (например 20 см)
    
    # Известная позиция маркера в системе мира (PyBullet / стенд)
    # Пример: маркер лежит в центре стола
    marker_world_pos = np.array([0.0, 0.0, 0.50])        # [X, Y, Z] в метрах
    marker_world_rot = np.eye(3)                         # если маркер лежит ровно на столе

    # Путь к сохранению матрицы
    output_path = Path("T_cam_to_world.npy")
    # =================================================================

    # Инициализация ArUco
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)        # 0 — обычно встроенная или первая подключённая камера
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    print("Наведи камеру на ArUco маркер и нажми 'c' для захвата.")
    print("Нажми 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Оцениваем позу маркера относительно камеры
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, 
                cameraMatrix=np.array([[600, 0, frame.shape[1]/2],
                                       [0, 600, frame.shape[0]/2],
                                       [0, 0, 1]]),   # примерная матрица камеры (подправь!)
                distCoeffs=np.zeros(5)
            )

            # Берём первый маркер
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            # Преобразуем rvec в матрицу поворота
            R_cam_to_marker, _ = cv2.Rodrigues(rvec)

            # Матрица T_cam_to_marker
            T_cam_to_marker = np.eye(4)
            T_cam_to_marker[:3, :3] = R_cam_to_marker
            T_cam_to_marker[:3, 3] = tvec

            # Матрица T_marker_to_world
            T_marker_to_world = np.eye(4)
            T_marker_to_world[:3, :3] = marker_world_rot
            T_marker_to_world[:3, 3] = marker_world_pos

            # Итоговая матрица
            T_cam_to_world = T_marker_to_world @ np.linalg.inv(T_cam_to_marker)

            print("\nМатрица T_camA_to_world успешно вычислена!")
            print(T_cam_to_world)

            # Сохраняем
            np.save(output_path, T_cam_to_world)
            print(f"\nМатрица сохранена в: {output_path}")
            break

        cv2.imshow("ArUco Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            print("Попытка захвата...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrate_cam_to_world()