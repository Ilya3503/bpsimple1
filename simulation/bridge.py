import pybullet as p
import pybullet_data
import time
from pathlib import Path
from typing import Optional, List


class SimulationBridge:
    def __init__(self, robot_urdf: Optional[str] = None, use_gui: bool = True):
        self.use_gui = use_gui
        self.robot_urdf = robot_urdf

        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240.0)

        self.robot_id = None
        self.num_joints = 0
        self.gripper_link_index = None

        self._load_scene()
        if robot_urdf and Path(robot_urdf).exists():
            self._load_robot(robot_urdf)

        p.resetDebugVisualizerCamera(1.8, 50, -40, [0.0, 0.0, 0.6])

    def _load_scene(self):
        p.loadURDF("plane.urdf")

        # Стол
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.7, 0.6, 0.025])
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.7, 0.6, 0.025],
                                        rgbaColor=[0.7, 0.6, 0.5, 1.0])
        p.createMultiBody(0, table_col, table_vis, [0.0, 0.0, 0.50])

        print("[bridge] Сцена (плоскость + стол) загружена")

    def _load_robot(self, urdf_path: str):
        """Загрузка mycobot_280"""
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[-0.35, 0.0, 0.52],   # робот стоит слева от стола
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        self.num_joints = p.getNumJoints(self.robot_id)
        self.gripper_link_index = self.num_joints - 1

        print(f"[bridge] ✅ Mycobot загружен ({self.num_joints} джоинтов)")

        # Сброс в home
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, 0.0)

    def load_object(self, position: List[float], extent: List[float], orientation: Optional[List[float]] = None):
        """Создаём объект из perception"""
        if orientation is None:
            orientation = [0, 0, 0, 1]

        half = [max(e/2, 0.005) for e in extent]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.1, 0.6, 1.0, 1.0])

        obj_id = p.createMultiBody(0.2, col, vis, position, orientation)
        print(f"[bridge] Объект создан: pos={[round(x,3) for x in position]}")
        return obj_id

    def move_to_joint_angles(self, joint_angles: List[float], speed: float = 0.6):
        if self.robot_id is None:
            return

        for i, angle in enumerate(joint_angles):
            if i >= self.num_joints:
                break
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle, maxVelocity=speed, force=800)

        for _ in range(120):
            p.stepSimulation()

    def disconnect(self):
        """Безопасное отключение PyBullet даже если окно уже закрыто."""
        if hasattr(self, 'client') and self.client is not None:
            try:
                p.disconnect(self.client)
                print("[bridge] PyBullet отключён корректно")
            except Exception as e:
                print(f"[bridge] Warning: PyBullet уже закрыт или ошибка при disconnect: {e}")
            finally:
                self.client = None
                self.robot_id = None