import pybullet as p
import pybullet_data
import time
from pathlib import Path
from typing import Optional, List


class SimulationBridge:
    def run_seconds(self, seconds: float):
        """Прогоняет симуляцию заданное количество секунд."""
        steps = int(seconds * 240)
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
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

    def get_active_joints(self) -> List[int]:
        """Возвращает только активные (revolute/prismatic) джоинты"""
        active = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED:   # 4 = fixed
                active.append(i)
        return active

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
            basePosition=[-0.15, 0.0, 0.52],
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
        """Безопасное движение только по активным джоинтам"""
        if self.robot_id is None:
            print("[bridge] Нет робота — пропускаем движение")
            return

        active_joints = self.get_active_joints()

        if len(joint_angles) != len(active_joints):
            print(
                f"[bridge] WARNING: Получено {len(joint_angles)} углов, а активных джоинтов {len(active_joints)}. Используем первые {len(active_joints)}")
            joint_angles = joint_angles[:len(active_joints)]

        for idx, joint_idx in enumerate(active_joints):
            angle = joint_angles[idx]
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=angle,
                maxVelocity=speed,
                force=800
            )

        # Плавное выполнение
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