import pybullet as p
import pybullet_data
import time
from pathlib import Path
from typing import Optional, List


class SimulationBridge:
    """
    PyBullet интерфейс для mycobot_280.
    Сцена: плоскость + стол + объект (из perception) + робот.
    """

    def __init__(self, robot_urdf: Optional[str] = None, use_gui: bool = True):
        self.use_gui = use_gui
        self.robot_urdf = robot_urdf
        self.robot_id = None
        self.num_joints = 0
        self.object_ids = []

        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240.0)

        self._load_scene()

        if robot_urdf and Path(robot_urdf).exists():
            self._load_robot(robot_urdf)
        else:
            print("[bridge] URDF не задан — только сцена без робота")

        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0.1, 0.0, 0.55],
        )

    def _load_scene(self):
        p.loadURDF("plane.urdf")
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.5, 0.025])
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.5, 0.025],
                                        rgbaColor=[0.65, 0.55, 0.4, 1.0])
        # Верхняя грань стола на Z=0.52
        p.createMultiBody(0, table_col, table_vis, [0.1, 0.0, 0.495])
        print("[bridge] Сцена загружена")

    def _load_robot(self, urdf_path: str):
        # Основание робота на поверхности стола Z=0.52
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[-0.35, 0.0, 0.52],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        self.num_joints = p.getNumJoints(self.robot_id)
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, 0.0)
            p.changeDynamics(self.robot_id, i, linearDamping=0.04, angularDamping=0.04)
        print(f"[bridge] Робот загружен: {self.num_joints} джоинтов")
        self._print_joints()

    def _print_joints(self):
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            t = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}.get(info[2], "?")
            print(f"  [{i}] {info[1].decode()} ({t})")

    def load_object(self, position: List[float], extent: List[float],
                    orientation: Optional[List[float]] = None) -> int:
        if orientation is None:
            orientation = [0, 0, 0, 1]
        half = [max(e / 2.0, 0.005) for e in extent]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.1, 0.6, 1.0, 1.0])
        obj_id = p.createMultiBody(0.1, col, vis, position, orientation)
        self.object_ids.append(obj_id)
        print(f"[bridge] Объект: pos={[round(v,3) for v in position]}")
        return obj_id

    def get_active_joints(self) -> List[int]:
        return [i for i in range(self.num_joints)
                if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]

    def move_to_joint_angles(self, joint_angles: List[float],
                              speed: float = 0.5, wait: bool = True):
        if self.robot_id is None:
            print(f"[bridge] Нет робота — пропуск: {joint_angles}")
            return
        active = self.get_active_joints()
        for i, j in enumerate(active):
            if i >= len(joint_angles):
                break
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=joint_angles[i],
                                    maxVelocity=speed, force=500)
        if wait:
            self._wait_motion(active, joint_angles)

    def _wait_motion(self, joints: List[int], targets: List[float],
                     timeout: float = 5.0, tol: float = 0.02):
        start = time.time()
        while time.time() - start < timeout:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            cur = [p.getJointState(self.robot_id, j)[0] for j in joints]
            if all(abs(cur[i] - targets[i]) < tol
                   for i in range(min(len(cur), len(targets)))):
                break

    def run_seconds(self, seconds: float):
        for _ in range(int(seconds * 240)):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def disconnect(self):
        if self.client is not None:
            p.disconnect(self.client)
            print("[bridge] PyBullet отключён")