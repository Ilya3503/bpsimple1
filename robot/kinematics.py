import numpy as np
from typing import List, Optional, Dict
from pathlib import Path


class RobotKinematics:
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
                last_link_vector=[0.0, 0.0, 0.0],
                active_links_mask=None,
            )
            self.num_joints = 6
            self.is_stub = False
            print(f"[kinematics] ✅ URDF загружен (6 джоинтов)")
        except Exception as e:
            print(f"[kinematics] Не удалось загрузить URDF: {e}")
            self.is_stub = True

    def solve_ik(self, target_position: List[float], target_orientation: Optional[List[float]] = None) -> Dict:
        """Никогда не падает"""
        if self.is_stub or self.chain is None:
            return self._stub_ik(target_position)

        try:
            target = np.array(target_position, dtype=float)
            initial = [0.0] * 9

            if target_orientation is not None:
                T = self._quaternion_to_matrix(target_orientation, target)
                full_angles = self.chain.inverse_kinematics_frame(T, initial_position=initial)
            else:
                full_angles = self.chain.inverse_kinematics(target, initial_position=initial)

            joint_angles = [float(x) for x in full_angles[1:7]]   # только 6 джоинтов

            return {
                "joint_angles": joint_angles,
                "is_stub": False,
                "reachable": True,
                "ik_error_m": 0.0,
                "target_position": target_position,
            }
        except Exception as e:
            print(f"[kinematics] IK ошибка: {e}")
            return self._stub_ik(target_position)

    def _stub_ik(self, target_position: List[float]) -> Dict:
        return {
            "joint_angles": [0.0] * 6,
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