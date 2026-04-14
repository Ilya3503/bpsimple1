import json
from pathlib import Path
from typing import Dict, List, Optional

from robot.grasp import select_best_cluster, compute_grasp_pose, compute_pregrasp_pose
from robot.kinematics import RobotKinematics
from robot.transform import camera_to_world, world_to_robot_base, transform_orientation, print_transform_status
from simulation.bridge import SimulationBridge


# Позиция основания робота в PyBullet.
# ВАЖНО: должна совпадать с basePosition в SimulationBridge._load_robot()
ROBOT_BASE_POSITION = [-0.15, 0.0, 0.52]
ROBOT_BASE_ORIENTATION = [0.0, 0.0, 0.0]  # roll, pitch, yaw в радианах


class RobotController:
    """
    Главный контроллер: perception → coordinate transform → grasp → IK → PyBullet.

    Цепочка координат:
        position.json (camera frame)
            → camera_to_world()  [T_cam_to_world из app/T_cam_to_world.npy]
            → world frame        [PyBullet world — объект спавнится сюда]
            → world_to_robot_base()  [вычитаем позицию базы робота]
            → robot base frame   [IK считается в этой системе]
    """

    def __init__(
        self,
        robot_urdf: Optional[str] = "simulation/models/mycobot_280/mycobot_280_m5.urdf",
        use_gui: bool = True,
        grasp_offset_z: float = 0.05,
        pregrasp_offset_z: float = 0.12,
    ):
        self.grasp_offset_z = grasp_offset_z
        self.pregrasp_offset_z = pregrasp_offset_z

        print_transform_status()

        self.kinematics = RobotKinematics(urdf_path=robot_urdf)
        self.sim = SimulationBridge(robot_urdf=robot_urdf, use_gui=use_gui)

    def execute_from_json(self, json_path: str) -> Dict:
        """
        Полный цикл захвата из position.json.

        Координаты проходят три системы:
            camera → world → robot base
        """
        print(f"\n{'='*55}")
        print(f"[controller] Старт: {json_path}")

        # --- 1. Читаем perception ---
        perception = self._load_json(json_path)
        if perception.get("status") != "ok":
            raise ValueError(f"Статус perception: {perception.get('status')}")

        clusters = perception.get("clusters", [])
        if not clusters:
            raise ValueError("Нет кластеров в position.json")

        print(f"[controller] Кластеров: {len(clusters)}")

        # --- 2. Выбираем кластер ---
        best = select_best_cluster(clusters)
        pose_cam = best["pose"]

        pos_camera = pose_cam["position"]          # координаты в camera frame
        ori_camera = pose_cam.get("orientation", [0, 0, 0, 1])
        extent = pose_cam["extent"]

        print(f"[controller] Camera frame:     {[round(v,3) for v in pos_camera]}")

        # --- 3. Camera frame → World frame ---
        pos_world = camera_to_world(pos_camera).tolist()
        # Ориентацию трансформируем через матрицу вращения камеры
        ori_world = transform_orientation(ori_camera, to_base=True)

        print(f"[controller] World frame:      {[round(v,3) for v in pos_world]}")

        # --- 4. Спавним объект в world frame в PyBullet ---
        # Объект появляется там где он реально находится на столе
        self.sim.load_object(
            position=pos_world,
            extent=extent,
            orientation=ori_world,
        )

        # --- 5. World frame → Robot base frame для IK ---
        pos_base = world_to_robot_base(
            pos_world,
            ROBOT_BASE_POSITION,
            ROBOT_BASE_ORIENTATION,
        ).tolist()

        print(f"[controller] Robot base frame: {[round(v,3) for v in pos_base]}")

        # --- 6. Grasp pose в robot base frame ---
        # Передаём кластер с координатами в base frame
        cluster_base = {
            "id": best["id"],
            "points_count": best["points_count"],
            "pose": {
                "position": pos_base,
                "orientation": [1.0, 0.0, 0.0, 0.0],  # top-down захват всегда
                "extent": extent,
                "method": pose_cam.get("method", "unknown"),
                "fitness": pose_cam.get("fitness"),
            }
        }

        grasp = compute_grasp_pose(cluster_base, self.grasp_offset_z)
        pregrasp = compute_pregrasp_pose(grasp, self.pregrasp_offset_z)

        print(f"[controller] Pre-grasp (base): {[round(v,3) for v in pregrasp['position']]}")
        print(f"[controller] Grasp (base):     {[round(v,3) for v in grasp['position']]}")

        # --- 7. IK ---
        ik_pre = self.kinematics.solve_ik(pregrasp["position"], pregrasp["orientation"])
        ik_grasp = self.kinematics.solve_ik(grasp["position"], grasp["orientation"])

        print(f"[controller] IK pre-grasp: reachable={ik_pre.get('reachable', False)}")
        print(f"[controller] IK grasp:     reachable={ik_grasp.get('reachable', False)}")

        # --- 8. Движение ---
        self._execute_grasp_sequence(
            ik_pre.get("joint_angles", [0.0] * 6),
            ik_grasp.get("joint_angles", [0.0] * 6),
        )

        result = {
            "status": "ok",
            "cluster_id": best["id"],
            "position_camera": pos_camera,
            "position_world": pos_world,
            "position_base": pos_base,
            "grasp_position": grasp["position"],
            "pregrasp_position": pregrasp["position"],
            "ik_pregrasp": ik_pre,
            "ik_grasp": ik_grasp,
        }

        print(f"[controller] Готово")
        print(f"{'='*55}\n")
        return result

    def _execute_grasp_sequence(
        self,
        pregrasp_angles: List[float],
        grasp_angles: List[float],
        hold_seconds: float = 10.0,
    ):
        sim = self.sim

        print("[controller] → Pre-grasp")
        try:
            sim.move_to_joint_angles(pregrasp_angles, speed=0.5)
        except Exception as e:
            print(f"[controller] Ошибка pre-grasp: {e}")
        sim.run_seconds(2.0)

        print("[controller] → Grasp")
        try:
            sim.move_to_joint_angles(grasp_angles, speed=0.3)
        except Exception as e:
            print(f"[controller] Ошибка grasp: {e}")
        sim.run_seconds(2.0)

        print("[controller] → Захват")
        sim.run_seconds(1.0)

        print("[controller] → Home")
        home = [0.0] * max(sim.num_joints, 6)
        try:
            sim.move_to_joint_angles(home, speed=0.5)
        except Exception as e:
            print(f"[controller] Ошибка home: {e}")
        sim.run_seconds(2.0)

        print(f"[controller] Держим сцену {hold_seconds}с...")
        sim.run_seconds(hold_seconds)

    def _load_json(self, path: str) -> Dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def shutdown(self):
        try:
            self.sim.disconnect()
        except Exception as e:
            print(f"[controller] shutdown warning: {e}")