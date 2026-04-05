import numpy as np
from typing import List, Optional, Dict
from pathlib import Path


class RobotKinematics:
    """
    IK решатель для 6-осевого манипулятора.

    Использует ikpy для вычисления обратной кинематики.

    Сейчас работает в режиме заглушки — возвращает
    нулевые углы джоинтов, пока не загружен URDF.

    Как подключить реального робота:
        1. Получить URDF файл от напарника
        2. Передать путь в конструктор: RobotKinematics("robot.urdf")
        3. Всё остальное работает без изменений
    """

    def __init__(self, urdf_path: Optional[str] = None):
        self.urdf_path = urdf_path
        self.chain = None
        self.num_joints = 6
        self.is_stub = True

        if urdf_path is not None:
            self._load_urdf(urdf_path)

    def _load_urdf(self, urdf_path: str):
        """
        Загружает кинематическую цепочку из URDF через ikpy.
        Вызывается автоматически если передан urdf_path.
        """
        path = Path(urdf_path)
        if not path.exists():
            raise FileNotFoundError(f"URDF файл не найден: {urdf_path}")

        try:
            from ikpy.chain import Chain
            self.chain = Chain.from_urdf_file(
                str(path),
                active_links_mask=self._build_active_mask(),
            )
            self.num_joints = len(
                [l for l in self.chain.links if l.joint_type != "fixed"]
            )
            self.is_stub = False
            print(f"[kinematics] URDF загружен: {urdf_path}")
            print(f"[kinematics] Джоинтов: {self.num_joints}")

        except ImportError:
            raise ImportError("ikpy не установлен. Запустите: pip install ikpy")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки URDF: {e}")

    def _build_active_mask(self) -> List[bool]:
        """
        Маска активных звеньев для ikpy.
        Первое и последнее звено обычно фиксированные (base и ee).
        Заполняется корректно после загрузки URDF.
        """
        # Заглушка — будет уточнена когда напарник принесёт URDF
        return None

    def solve_ik(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None,
    ) -> Dict:
        """
        Решает обратную кинематику для целевой позиции.

        Параметры:
            target_position    — [x, y, z] в метрах (world frame)
            target_orientation — [qx, qy, qz, qw] (опционально)

        Возвращает:
            {
                "joint_angles": [...],   # углы в радианах
                "is_stub": bool,         # True если реальный IK не решался
                "reachable": bool,       # False если точка недостижима
            }
        """
        if self.is_stub:
            return self._stub_ik(target_position)

        return self._real_ik(target_position, target_orientation)

    def _stub_ik(self, target_position: List[float]) -> Dict:
        """
        Заглушка IK — возвращает нулевые углы.
        Используется пока нет URDF.

        Робот в симуляции не будет двигаться правильно,
        но вся остальная цепочка кода будет работать.
        """
        print(f"[kinematics] STUB IK для позиции {target_position}")
        print(f"[kinematics] Загрузите URDF для реального IK")

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
        """
        Реальный IK через ikpy.
        Активируется автоматически после загрузки URDF.
        """
        import ikpy.utils.math as ikpy_math

        target = np.array(target_position)

        # Строим матрицу цели для ikpy
        if target_orientation is not None:
            # Конвертируем quaternion в матрицу вращения
            T = self._quaternion_to_matrix(target_orientation, target)
        else:
            T = None

        try:
            if T is not None:
                joint_angles = self.chain.inverse_kinematics_frame(
                    T,
                    initial_position=self._home_position(),
                )
            else:
                joint_angles = self.chain.inverse_kinematics(
                    target,
                    initial_position=self._home_position(),
                )

            # Проверяем достижимость: считаем FK и смотрим ошибку
            fk = self.chain.forward_kinematics(joint_angles)
            achieved = fk[:3, 3]
            error = float(np.linalg.norm(achieved - target))
            reachable = error < 0.01  # допуск 1см

            return {
                "joint_angles": joint_angles.tolist(),
                "is_stub": False,
                "reachable": reachable,
                "ik_error_m": error,
                "target_position": target_position,
            }

        except Exception as e:
            print(f"[kinematics] IK не решился: {e}")
            return {
                "joint_angles": self._home_position(),
                "is_stub": False,
                "reachable": False,
                "ik_error_m": None,
                "target_position": target_position,
                "error": str(e),
            }

    def _home_position(self) -> List[float]:
        """Начальная позиция для итеративного IK решателя."""
        return [0.0] * (self.num_joints + 2)  # +2 для base и ee звеньев ikpy

    def _quaternion_to_matrix(
        self,
        quat: List[float],
        position: np.ndarray,
    ) -> np.ndarray:
        """
        Строит матрицу 4x4 из quaternion [qx, qy, qz, qw] и позиции.
        """
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
        """
        Прямая кинематика — позиция end-effector для заданных углов.
        Используется для проверки и визуализации.
        """
        if self.is_stub:
            return {
                "position": [0.0, 0.0, 0.0],
                "is_stub": True,
            }

        T = self.chain.forward_kinematics(joint_angles)
        return {
            "position": T[:3, 3].tolist(),
            "rotation_matrix": T[:3, :3].tolist(),
            "is_stub": False,
        }