import numpy as np
from typing import Dict, List


import numpy as np
from typing import Dict, List


def select_best_cluster(clusters: List[Dict]) -> Dict:
    """
    Выбирает лучший кластер:
      1. Сначала по ICP fitness (чем выше — тем лучше)
      2. Если fitness нет или равен 0 — по количеству точек
    """
    if not clusters:
        raise ValueError("Список кластеров пуст")

    def score(c: Dict):
        pose = c.get("pose", {})
        fitness = pose.get("fitness")
        # Если fitness реальный — используем его как главный приоритет
        if isinstance(fitness, (int, float)) and fitness > 0:
            return (fitness, c.get("points_count", 0))
        else:
            return (0.0, c.get("points_count", 0))

    best = max(clusters, key=score)

    pose = best.get("pose", {})
    fitness = pose.get("fitness")

    print(f"[controller] Выбран кластер {best['id']} | "
          f"points={best.get('points_count', 0)} | "
          f"method={pose.get('method', 'unknown')} | "
          f"fitness={fitness:.4f if isinstance(fitness, (int,float)) else 'N/A'}")

    return best


def compute_grasp_pose(cluster: Dict, grasp_offset_z: float = 0.05) -> Dict:
    """
    Вычисляет позу захвата сверху.
    Теперь использует реальную ориентацию из ICP/OBB.
    """
    pos = cluster["pose"]["position"]
    extent = cluster["pose"]["extent"]

    # Высота верхней грани объекта
    object_top_z = pos[2] + extent[2] / 2.0

    grasp_position = [
        pos[0],
        pos[1],
        object_top_z + grasp_offset_z,
    ]

    # БЕРЁМ РЕАЛЬНУЮ ОРИЕНТАЦИЮ ИЗ POSE (очень важно!)
    grasp_orientation = cluster["pose"].get("orientation", [1.0, 0.0, 0.0, 0.0])

    return {
        "position": grasp_position,
        "orientation": grasp_orientation,
        "object_position": pos,
        "object_extent": extent,
        "grasp_offset_z": grasp_offset_z,
    }


def compute_pregrasp_pose(grasp_pose: Dict, pregrasp_offset_z: float = 0.12) -> Dict:
    """Просто поднимаемся выше"""
    pre_pos = list(grasp_pose["position"])
    pre_pos[2] += pregrasp_offset_z

    return {
        "position": pre_pos,
        "orientation": grasp_pose["orientation"],
    }