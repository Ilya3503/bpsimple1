import time
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path


class SimulationBridge:
    """
    Интерфейс между robot_controller и PyBullet симулятором.

    Отвечает за:
        - запуск/остановку симуляции
        - загрузку сцены (стол, объект, робот)
        - отправку команд джоинтам
        - визуализацию движения

    Как подключить реального робота:
        1. Напарник предоставляет URDF файл
        2. Передать путь: SimulationBridge(robot_urdf="path/to/robot.urdf")
        3. Сцену напарник описывает в simulation/scene.py

    Сейчас работает с заглушкой сцены — простой куб вместо робота.
    """

    def __init__(
        self,
        robot_urdf: Optional[str] = None,
        use_gui: bool = True,
        gravity: float = -9.81,
    ):
        self.robot_urdf = robot_urdf
        self.use_gui = use_gui
        self.gravity = gravity

        self.physics_client = None
        self.robot_id = None
        self.table_id = None
        self.object_id = None
        self.num_joints = 0
        self.is_stub = robot_urdf is None

        self._connect()

    def _connect(self):
        """Подключается к PyBullet и настраивает симулятор."""
        try:
            import pybullet as p
            import pybullet_data
            self.p = p
            self.pybullet_data = pybullet_data.getDataPath()
        except ImportError:
            raise ImportError("pybullet не установлен. Запустите: pip install pybullet")

        mode = self.p.GUI if self.use_gui else self.p.DIRECT
        self.physics_client = self.p.connect(mode)

        self.p.setAdditionalSearchPath(self.pybullet_data)
        self.p.setGravity(0, 0, self.gravity)
        self.p.setRealTimeSimulation(0)

        print(f"[simulation] PyBullet подключён (GUI={self.use_gui})")

    def load_scene(self, object_position: Optional[List[float]] = None):
        """
        Загружает сцену: плоскость, стол, объект, робот.

        object_position — позиция куба из perception pipeline [x, y, z].
        Если не задана — куб ставится в центр сцены.

        Сцену (стол, освещение, камеру) напарник может расширить
        в simulation/scene.py не трогая этот файл.
        """
        p = self.p

        # --- Плоскость ---
        self.plane_id = p.loadURDF("plane.urdf")

        # --- Стол (заглушка — простой прямоугольник) ---
        # Напарник заменит на нормальный URDF стола
        table_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.02],
        )
        table_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.02],
            rgbaColor=[0.8, 0.7, 0.6, 1],
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0, 0, 0.65],  # высота стола ~65см
        )

        # --- Объект (куб — заглушка) ---
        pos = object_position if object_position else [0.0, 0.0, 0.72]
        cube_size = 0.05  # 5см куб по умолчанию
        cube_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3,
        )
        cube_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3,
            rgbaColor=[0.2, 0.5, 0.9, 1],
        )
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_col,
            baseVisualShapeIndex=cube_vis,
            basePosition=pos,
        )

        # --- Робот ---
        if self.robot_urdf and Path(self.robot_urdf).exists():
            self._load_robot()
        else:
            print("[simulation] URDF робота не задан — робот не загружен")
            print("[simulation] Передайте robot_urdf= когда напарник принесёт файл")

        print(f"[simulation] Сцена загружена. Объект в позиции: {pos}")

    def _load_robot(self):
        """Загружает URDF робота в симулятор."""
        p = self.p

        # Робот стоит рядом со столом
        robot_base_position = [-0.5, 0.0, 0.0]
        robot_base_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robot_id = p.loadURDF(
            self.robot_urdf,
            basePosition=robot_base_position,
            baseOrientation=robot_base_orientation,
            useFixedBase=True,
        )

        # Считаем количество джоинтов
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"[simulation] Робот загружен: {self.robot_urdf}")
        print(f"[simulation] Джоинтов: {self.num_joints}")

        # Устанавливаем демпфирование для стабильной симуляции
        for i in range(self.num_joints):
            p.changeDynamics(
                self.robot_id, i,
                linearDamping=0.04,
                angularDamping=0.04,
            )

    def move_to_joint_angles(
        self,
        joint_angles: List[float],
        speed: float = 0.5,
        wait: bool = True,
    ):
        """
        Отправляет команду движения роботу по джоинтам.

        Параметры:
            joint_angles — углы в радианах (из IK решателя)
            speed        — скорость движения (рад/с)
            wait         — ждать завершения движения

        Если робот не загружен (нет URDF) — логирует и пропускает.
        """
        if self.robot_id is None:
            print(f"[simulation] STUB move_to_joint_angles: {joint_angles}")
            return

        p = self.p
        active_joints = self._get_active_joints()

        for i, joint_idx in enumerate(active_joints):
            if i >= len(joint_angles):
                break
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                maxVelocity=speed,
            )

        if wait:
            self._wait_for_motion(active_joints, joint_angles)

    def _get_active_joints(self) -> List[int]:
        """Возвращает индексы не-фиксированных джоинтов."""
        p = self.p
        active = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_type = info[2]
            if joint_type != p.JOINT_FIXED:
                active.append(i)
        return active

    def _wait_for_motion(
        self,
        joint_indices: List[int],
        target_angles: List[float],
        timeout: float = 5.0,
        tolerance: float = 0.01,
    ):
        """Ждёт пока робот достигнет целевых углов."""
        p = self.p
        start = time.time()

        while time.time() - start < timeout:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

            current = [
                p.getJointState(self.robot_id, j)[0]
                for j in joint_indices
            ]
            errors = [
                abs(current[i] - target_angles[i])
                for i in range(min(len(current), len(target_angles)))
            ]

            if all(e < tolerance for e in errors):
                break

    def step(self, steps: int = 1):
        """Шаг симуляции вручную."""
        for _ in range(steps):
            self.p.stepSimulation()

    def run_seconds(self, seconds: float):
        """Запускает симуляцию на заданное количество секунд."""
        steps = int(seconds * 240)
        for _ in range(steps):
            self.p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def get_object_position(self) -> Optional[List[float]]:
        """Возвращает текущую позицию объекта в симуляторе."""
        if self.object_id is None:
            return None
        pos, _ = self.p.getBasePositionAndOrientation(self.object_id)
        return list(pos)

    def disconnect(self):
        """Отключается от PyBullet."""
        if self.physics_client is not None:
            self.p.disconnect()
            print("[simulation] PyBullet отключён")