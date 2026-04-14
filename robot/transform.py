"""
Модуль трансформации координат.

Цепочка:
    camera frame → world frame → robot base frame

camera frame  — система координат камеры (выход perception pipeline)
world frame   — мировая система координат (PyBullet world)
robot base    — система координат основания робота

Матрицы задаются через .npy файлы, которые создаются скриптами калибровки:
    T_cam_to_world.npy   — создаётся calibration_coordinate_systems.py
    T_cam_to_world задаёт позицию камеры относительно мира

Robot base offset задаётся вручную как позиция основания робота в PyBullet.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional


# ==============================================================================
# ЗАГРУЗКА МАТРИЦ КАЛИБРОВКИ
# ==============================================================================

def _load_matrix(path: Path, name: str) -> np.ndarray:
    """Загружает матрицу 4x4 из .npy файла. При отсутствии — единичная."""
    if path.exists():
        T = np.load(str(path))
        print(f"[transform] Загружена {name}: {path}")
        print(f"[transform] Трансляция: {np.round(T[:3, 3], 4)}")
        return T
    else:
        print(f"[transform] ВНИМАНИЕ: {path} не найден — используется единичная матрица")
        print(f"[transform] Запустите скрипт калибровки для создания {path.name}")
        return np.eye(4, dtype=np.float64)


# Путь к матрице калибровки камера → мир
# Создаётся скриптом app/calibration_coordinate_systems.py
_CALIB_PATH = Path(__file__).resolve().parent.parent / "app" / "T_cam_to_world.npy"

T_CAM_TO_WORLD: np.ndarray = _load_matrix(_CALIB_PATH, "T_cam_to_world")


# ==============================================================================
# ОСНОВНЫЕ ФУНКЦИИ ТРАНСФОРМАЦИИ
# ==============================================================================

def camera_to_world(position_cam: List[float]) -> np.ndarray:
    """
    Переводит точку из camera frame в world frame.

    Использует T_CAM_TO_WORLD из файла калибровки.
    Если файл не загружен — возвращает исходные координаты (заглушка).

    Args:
        position_cam: [x, y, z] в системе координат камеры

    Returns:
        np.ndarray [x, y, z] в мировой системе координат
    """
    p = np.array([*position_cam, 1.0], dtype=np.float64)
    result = T_CAM_TO_WORLD @ p
    return result[:3]


def world_to_robot_base(
    position_world: List[float],
    robot_base_position: List[float],
    robot_base_orientation_euler: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Переводит точку из world frame в robot base frame.

    robot_base_position — [x, y, z] основания робота в PyBullet
    robot_base_orientation_euler — [roll, pitch, yaw] в радианах (обычно [0,0,0])

    Args:
        position_world: [x, y, z] в мировой системе
        robot_base_position: позиция основания робота в PyBullet
        robot_base_orientation_euler: ориентация основания робота

    Returns:
        np.ndarray [x, y, z] в системе координат основания робота
    """
    if robot_base_orientation_euler is None or all(v == 0 for v in robot_base_orientation_euler):
        # Частый случай — робот стоит ровно, только смещение
        result = np.array(position_world) - np.array(robot_base_position)
        return result

    # Общий случай — с поворотом
    T_base = _build_transform(robot_base_position, robot_base_orientation_euler)
    T_base_inv = np.linalg.inv(T_base)
    p = np.array([*position_world, 1.0], dtype=np.float64)
    result = T_base_inv @ p
    return result[:3]


def camera_to_robot_base(
    position_cam: List[float],
    robot_base_position: List[float],
    robot_base_orientation_euler: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Полная цепочка: camera frame → world frame → robot base frame.

    Это главная функция которую использует controller.

    Args:
        position_cam: [x, y, z] из perception pipeline (camera frame)
        robot_base_position: позиция основания робота в PyBullet
        robot_base_orientation_euler: ориентация основания [roll, pitch, yaw]

    Returns:
        np.ndarray [x, y, z] в системе координат основания робота (для IK)
    """
    pos_world = camera_to_world(position_cam)
    pos_base = world_to_robot_base(
        pos_world.tolist(),
        robot_base_position,
        robot_base_orientation_euler,
    )
    return pos_base


def transform_orientation(
    orientation_cam: List[float],
    to_base: bool = True,
) -> List[float]:
    """
    Трансформирует ориентацию (quaternion) из camera frame в world/base frame.

    Для top-down grasp ориентация обычно фиксирована и не требует
    трансформации — gripper всегда смотрит вниз в мировой системе.

    Args:
        orientation_cam: [qx, qy, qz, qw]
        to_base: True — трансформировать, False — вернуть как есть

    Returns:
        [qx, qy, qz, qw] в целевой системе координат
    """
    if not to_base:
        return orientation_cam

    # Извлекаем матрицу вращения из T_CAM_TO_WORLD
    R_cam_to_world = T_CAM_TO_WORLD[:3, :3]

    # Конвертируем quaternion в матрицу
    R_obj = _quaternion_to_rotation_matrix(orientation_cam)

    # Трансформируем ориентацию
    R_world = R_cam_to_world @ R_obj

    # Конвертируем обратно в quaternion
    return _rotation_matrix_to_quaternion(R_world)


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def _build_transform(position: List[float], euler_rpy: List[float]) -> np.ndarray:
    """Строит матрицу трансформации 4x4 из позиции и углов Эйлера (roll, pitch, yaw)."""
    roll, pitch, yaw = euler_rpy

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])

    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def _quaternion_to_rotation_matrix(quat: List[float]) -> np.ndarray:
    """[qx, qy, qz, qw] → матрица вращения 3x3."""
    qx, qy, qz, qw = quat
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    return R


def _rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """Матрица вращения 3x3 → [qx, qy, qz, qw]. Формула Shepperd."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def print_transform_status():
    """Выводит статус загруженных матриц калибровки."""
    is_identity = np.allclose(T_CAM_TO_WORLD, np.eye(4))
    status = "ЗАГЛУШКА (единичная)" if is_identity else "загружена"
    print(f"[transform] T_cam_to_world: {status}")
    if not is_identity:
        t = T_CAM_TO_WORLD[:3, 3]
        print(f"[transform] Камера в мире: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")