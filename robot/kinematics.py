import numpy as np
from typing import List, Optional, Dict
from pathlib import Path



class RobotKinematics:
    """
    Стабильная версия IK для mycobot_280.
    """

    def __init__(self, urdf_path: Optional[str] = None):
        self.urdf_path = urdf_path
        self.chain = None
        self.num_joints = 6
        self.is_stub = True

        if urdf_path and Path(urdf_path).exists():
            self._load_urdf(urdf_path)

    def _load_urdf(self, urdf_path: str):
        try:
            from ikpy.chain import Chain

            self.chain = Chain.from_urdf_file(
                urdf_path,
                base_elements=["g_base"],
                last_link_vector=[0.0, 0.0, 0.045],
                active_links_mask=None,                    # пусть ikpy сам определит
            )

            self.num_joints = 6
            self.is_stub = False

            print(f"[kinematics] ✅ Mycobot URDF загружен (6 активных джоинтов)")

        except Exception as e:
            print(f"[kinematics] Ошибка загрузки URDF: {e}")
            self.is_stub = True

    def solve_ik(self, target_position: List[float], target_orientation: Optional[List[float]] = None) -> Dict:
        if self.is_stub or self.chain is None:
            return self._stub_ik(target_position)

        try:
            target = np.array(target_position, dtype=float)

            # Правильный вызов для ikpy
            if target_orientation is not None:
                T = self._quaternion_to_matrix(target_orientation, target)
                full_angles = self.chain.inverse_kinematics_frame(
                    T,
                    initial_position=[0.0] * 9   # 9 — безопасная длина для mycobot
                )
            else:
                full_angles = self.chain.inverse_kinematics(
                    target,
                    initial_position=[0.0] * 9
                )

            # Берем только реальные 6 джоинтов (обычно индексы 1-7)
            joint_angles = full_angles[1:7].tolist()

            # Проверка
            fk = self.chain.forward_kinematics(full_angles)
            error = float(np.linalg.norm(fk[:3, 3] - target))

            return {
                "joint_angles": joint_angles,
                "is_stub": False,
                "reachable": error < 0.05,
                "ik_error_m": error,
                "target_position": target_position,
            }

        except Exception as e:
            print(f"[kinematics] IK ошибка: {e}")
            return self._stub_ik(target_position)

    def _stub_ik(self, target_position: List[float]) -> Dict:
        print(f"[kinematics] STUB IK для позиции { [round(x,3) for x in target_position] }")
        return {
            "joint_angles": [0.0] * self.num_joints,
            "is_stub": True,
            "reachable": False,
            "target_position": target_position,
        }

    def _quaternion_to_matrix(self, quat: List[float], position: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat
        T = np.eye(4)
        T[0,0] = 1 - 2*(qy**2 + qz**2)
        T[0,1] = 2*(qx*qy - qz*qw)
        T[0,2] = 2*(qx*qz + qy*qw)
        T[1,0] = 2*(qx*qy + qz*qw)
        T[1,1] = 1 - 2*(qx**2 + qz**2)
        T[1,2] = 2*(qy*qz - qx*qw)
        T[2,0] = 2*(qx*qz - qy*qw)
        T[2,1] = 2*(qy*qz + qx*qw)
        T[2,2] = 1 - 2*(qx**2 + qy**2)
        T[:3,3] = position
        return T