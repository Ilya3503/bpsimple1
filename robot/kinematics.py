import numpy as np
from typing import List, Optional, Dict
from pathlib import Path


class RobotKinematics:
    """
    IK решатель для mycobot_280 через ikpy.
    Без URDF — заглушка с нулевыми углами.
    С URDF — реальный IK.
    """

    def __init__(self, urdf_path: Optional[str] = None):
        self.urdf_path = urdf_path
        self.chain = None
        self.num_joints = 6
        self.is_stub = True

        if urdf_path is not None:
            self._load_urdf(urdf_path)

    def _load_urdf(self, urdf_path: str):
        path = Path(urdf_path)
        if not path.exists():
            raise FileNotFoundError(f"URDF не найден: {urdf_path}")

        try:
            from ikpy.chain import Chain

            self.chain = Chain.from_urdf_file(
                str(path),
                base_elements=["g_base"],
                # last_link_vector — смещение от последнего джоинта до TCP
                # для mycobot_280 это примерно 4.5см по Z
                last_link_vector=[0, 0, 0.045],
                active_links_mask=None,
            )

            # Считаем только revolute джоинты
            self.num_joints = sum(
                1 for link in self.chain.links
                if link.joint_type not in ("fixed", None)
                and hasattr(link, "joint_type")
            )
            # Запасной вариант если подсчёт не сработал
            if self.num_joints == 0:
                self.num_joints = 6

            self.is_stub = False
            print(f"[kinematics] URDF загружен: {urdf_path}")
            print(f"[kinematics] Звеньев в цепочке: {len(self.chain.links)}")
            print(f"[kinematics] Активных джоинтов: {self.num_joints}")

        except Exception as e:
            print(f"[kinematics] Ошибка загрузки URDF: {e}")
            print("[kinematics] Работаем в режиме заглушки")
            self.is_stub = True

    def solve_ik(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None,
    ) -> Dict:
        if self.is_stub:
            return self._stub_ik(target_position)
        return self._real_ik(target_position, target_orientation)

    def _stub_ik(self, target_position: List[float]) -> Dict:
        print(f"[kinematics] STUB IK: {[round(v,3) for v in target_position]}")
        return {
            "joint_angles": [0.0] * self.num_joints,
            "is_stub": True,
            "reachable": False,
            "target_position": target_position,
        }

    def _real_ik(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]],
    ) -> Dict:
        target = np.array(target_position)

        try:
            if target_orientation is not None:
                T = self._quaternion_to_matrix(target_orientation, target)
                joint_angles = self.chain.inverse_kinematics_frame(
                    T,
                    initial_position=self._home_position(),
                )
            else:
                joint_angles = self.chain.inverse_kinematics(
                    target,
                    initial_position=self._home_position(),
                )

            # Проверяем ошибку через FK
            fk = self.chain.forward_kinematics(joint_angles)
            achieved = fk[:3, 3]
            error = float(np.linalg.norm(achieved - target))
            reachable = error < 0.02  # допуск 2см

            if not reachable:
                print(f"[kinematics] Точка труднодостижима, ошибка={error*100:.1f}см")

            return {
                "joint_angles": joint_angles.tolist(),
                "is_stub": False,
                "reachable": reachable,
                "ik_error_m": error,
                "target_position": target_position,
            }

        except Exception as e:
            print(f"[kinematics] IK ошибка: {e}")
            return {
                "joint_angles": [0.0] * (self.num_joints + 2),
                "is_stub": False,
                "reachable": False,
                "ik_error_m": None,
                "target_position": target_position,
                "error": str(e),
            }

    def _home_position(self) -> List[float]:
        """Начальная позиция для IK итератора."""
        # ikpy требует N+2 элементов (base + links + ee)
        return [0.0] * (self.num_joints + 2)

    def _quaternion_to_matrix(self, quat: List[float], position: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat
        T = np.eye(4)
        T[0, 0] = 1 - 2*(qy**2 + qz**2)
        T[0, 1] = 2*(qx*qy - qz*qw)
        T[0, 2] = 2*(qx*qz + qy*qw)
        T[1, 0] = 2*(qx*qy + qz*qw)
        T[1, 1] = 1 - 2*(qx**2 + qz**2)
        T[1, 2] = 2*(qy*qz - qx*qw)
        T[2, 0] = 2*(qx*qz - qy*qw)
        T[2, 1] = 2*(qy*qz + qx*qw)
        T[2, 2] = 1 - 2*(qx**2 + qy**2)
        T[:3, 3] = position
        return T

    def forward_kinematics(self, joint_angles: List[float]) -> Dict:
        if self.is_stub:
            return {"position": [0.0, 0.0, 0.0], "is_stub": True}
        T = self.chain.forward_kinematics(joint_angles)
        return {
            "position": T[:3, 3].tolist(),
            "rotation_matrix": T[:3, :3].tolist(),
            "is_stub": False,
        }