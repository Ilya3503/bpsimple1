import numpy as np
from typing import Dict, List


def select_best_cluster(clusters: List[Dict]) -> Dict:
    """
    Выбирает лучший кластер для захвата.

    Стратегия: берём кластер с наибольшим количеством точек —
    он скорее всего является реальным объектом, а не шумом.

    Когда появится ICP — можно переключить на выбор по fitness score.
    """
    if not clusters:
        raise ValueError("Список кластеров пуст")

    best = max(clusters, key=lambda c: c["points_count"])
    return best


def compute_grasp_pose(cluster: Dict, grasp_offset_z: float = 0.05) -> Dict:
    """
    Вычисляет позу захвата (grasp pose) для робота.

    Стратегия: захват сверху (top-down grasp).
    Gripper подходит вертикально сверху над центром объекта.

    Параметры:
        cluster        — кластер из position.json
        grasp_offset_z — высота подхода над объектом (метры)
                         по умолчанию 5см над поверхностью куба

    Возвращает:
        grasp_pose — словарь с position и orientation
                     совместимый с IK решателем и PyBullet
    """
    pos = cluster["pose"]["position"]
    extent = cluster["pose"]["extent"]

    # Высота верхней грани объекта
    object_top_z = pos[2] + extent[2] / 2.0

    # Позиция захвата: над центром объекта
    grasp_position = [
        pos[0],
        pos[1],
        object_top_z + grasp_offset_z,
    ]

    # Ориентация: gripper смотрит вниз (top-down)
    # quaternion [qx, qy, qz, qw]
    # Поворот на 180 градусов вокруг X — захват направлен вниз
    grasp_orientation = [1.0, 0.0, 0.0, 0.0]

    return {
        "position": grasp_position,
        "orientation": grasp_orientation,
        "object_position": pos,
        "object_extent": extent,
        "grasp_offset_z": grasp_offset_z,
    }


def compute_pregrasp_pose(grasp_pose: Dict, pregrasp_offset_z: float = 0.10) -> Dict:
    """
    Вычисляет позу подхода (pre-grasp pose).

    Робот сначала идёт в pre-grasp (безопасная высота),
    затем опускается в grasp.

    Стандартная двухшаговая стратегия:
        1. Переместиться в pre-grasp
        2. Опуститься в grasp
        3. Закрыть gripper
    """
    pre_pos = list(grasp_pose["position"])
    pre_pos[2] += pregrasp_offset_z

    return {
        "position": pre_pos,
        "orientation": grasp_pose["orientation"],
    }