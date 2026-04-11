import pybullet as p
import pybullet_data
from pathlib import Path
import numpy as np
from typing import Optional, List


class SimulationBridge:
    def __init__(self, robot_urdf: Optional[str] = None, use_gui: bool = True):
        self.use_gui = use_gui
        self.robot_urdf = robot_urdf

        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240.0)

        self.robot_id = None
        self.num_joints = 0
        self.gripper_link_index = None

        self._load_scene()
        if robot_urdf and Path(robot_urdf).exists():
            self._load_robot(robot_urdf)
        else:
            print("[bridge] ⚠️ URDF робота не задан — будет только сцена")

        p.resetDebugVisualizerCamera(2.2, 45, -30, [0.2, 0, 0.6])

    def _load_scene(self):
        """Загружает плоскость + стол"""
        p.loadURDF("plane.urdf")

        # Стол
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.5, 0.02])
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.5, 0.02],
                                        rgbaColor=[0.65, 0.55, 0.4, 1.0])
        p.createMultiBody(0, table_col, table_vis, [0, 0, 0.5])

    def _load_robot(self, urdf_path: str):
        """Загружает mycobot"""
        robot_base_pos = [-0.4, 0.0, 0.0]  # слева от стола

        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=robot_base_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        self.num_joints = p.getNumJoints(self.robot_id)
        self.gripper_link_index = self.num_joints - 1

        print(f"[bridge] ✅ Робот загружен ({self.num_joints} джоинтов)")

        # Сброс в home-позу
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, 0.0)

    def load_object(self, position: List[float], orientation: List[float], extent: List[float]):
        """Создаёт объект (куб) из perception"""
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[e / 2 for e in extent])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[e / 2 for e in extent],
                                  rgbaColor=[0.0, 0.7, 1.0, 1.0])
        obj_id = p.createMultiBody(0.5, col, vis, position, orientation)
        print(f"[bridge] Объект создан в позиции {position}")
        return obj_id

    def move_to_joint_angles(self, joint_angles: List[float]):
        """Движение робота"""
        if self.robot_id is None:
            print("[bridge] Нет робота — пропускаем движение")
            return

        for i, angle in enumerate(joint_angles):
            if i >= self.num_joints:
                break
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle, force=500)

        for _ in range(100):  # плавное движение
            p.stepSimulation()

    def disconnect(self):
        if self.client is not None:
            p.disconnect(self.client)
            print("[bridge] PyBullet отключён")