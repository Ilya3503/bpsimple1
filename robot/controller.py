import json
from pathlib import Path
from typing import Dict, List, Optional

from robot.grasp import select_best_cluster, compute_grasp_pose, compute_pregrasp_pose
from robot.kinematics import RobotKinematics
from simulation.bridge import SimulationBridge


class RobotController:
    """
    Главный контроллер робота.

    Связывает perception pipeline и симулятор:
        position.json → grasp pose → IK → joint angles → PyBullet

    Использование:
        controller = RobotController()
        controller.execute_from_json("results/position.json")

    Когда напарник принесёт URDF:
        controller = RobotController(
            robot_urdf="simulation/models/robot/robot.urdf"
        )
    """

    def __init__(
        self,
        robot_urdf: Optional[str] = None,
        use_gui: bool = True,
        grasp_offset_z: float = 0.05,
        pregrasp_offset_z: float = 0.10,
    ):
        self.grasp_offset_z = grasp_offset_z
        self.pregrasp_offset_z = pregrasp_offset_z

        # IK решатель
        self.kinematics = RobotKinematics(urdf_path=robot_urdf)

        # Симулятор
        self.sim = SimulationBridge(
            robot_urdf=robot_urdf,
            use_gui=use_gui,
        )

    def execute_from_json(self, json_path: str) -> Dict:
        """
        Полный цикл: читает position.json и выполняет захват.

        Шаги:
            1. Читаем position.json
            2. Выбираем лучший кластер
            3. Вычисляем grasp pose
            4. Решаем IK
            5. Загружаем сцену в симулятор
            6. Двигаем робота

        Возвращает словарь с результатом выполнения.
        """
        print(f"\n{'='*50}")
        print(f"[controller] Старт выполнения из: {json_path}")

        # --- 1. Читаем perception результат ---
        perception = self._load_json(json_path)

        if perception.get("status") != "ok":
            raise ValueError(f"Perception pipeline вернул статус: {perception.get('status')}")

        clusters = perception.get("clusters", [])
        if not clusters:
            raise ValueError("Нет кластеров в position.json")

        print(f"[controller] Кластеров в сцене: {len(clusters)}")

        # --- 2. Выбираем лучший кластер ---
        best = select_best_cluster(clusters)
        print(f"[controller] Выбран кластер {best['id']} "
              f"({best['points_count']} точек) "
              f"pos={best['pose']['position']}")

        # --- 3. Вычисляем grasp pose ---
        grasp = compute_grasp_pose(best, self.grasp_offset_z)
        pregrasp = compute_pregrasp_pose(grasp, self.pregrasp_offset_z)

        print(f"[controller] Pre-grasp позиция: {pregrasp['position']}")
        print(f"[controller] Grasp позиция:     {grasp['position']}")

        # --- 4. Решаем IK ---
        ik_pregrasp = self.kinematics.solve_ik(
            pregrasp["position"],
            pregrasp["orientation"],
        )
        ik_grasp = self.kinematics.solve_ik(
            grasp["position"],
            grasp["orientation"],
        )

        print(f"[controller] IK pre-grasp: stub={ik_pregrasp['is_stub']} "
              f"reachable={ik_pregrasp['reachable']}")
        print(f"[controller] IK grasp:     stub={ik_grasp['is_stub']} "
              f"reachable={ik_grasp['reachable']}")

        # --- 5. Загружаем сцену ---
        object_pos = best["pose"]["position"]
        self.sim.load_scene(object_position=object_pos)

        # --- 6. Движение робота ---
        self._execute_grasp_sequence(
            ik_pregrasp["joint_angles"],
            ik_grasp["joint_angles"],
        )

        result = {
            "status": "ok",
            "cluster_id": best["id"],
            "object_position": object_pos,
            "grasp_position": grasp["position"],
            "pregrasp_position": pregrasp["position"],
            "ik_pregrasp": ik_pregrasp,
            "ik_grasp": ik_grasp,
        }

        print(f"[controller] Выполнение завершено")
        print(f"{'='*50}\n")
        return result

    def _execute_grasp_sequence(
        self,
        pregrasp_angles: List[float],
        grasp_angles: List[float],
    ):
        """
        Выполняет последовательность движений для захвата:
            1. Home позиция
            2. Pre-grasp (над объектом)
            3. Grasp (к объекту)
            4. Пауза (имитация захвата)
            5. Возврат в home

        Если URDF не загружен — симуляция всё равно запускается,
        просто робот не будет виден в сцене.
        """
        sim = self.sim
        print("[controller] Шаг 1: pre-grasp позиция")
        sim.move_to_joint_angles(pregrasp_angles, speed=0.5)
        sim.run_seconds(1.0)

        print("[controller] Шаг 2: движение к объекту")
        sim.move_to_joint_angles(grasp_angles, speed=0.3)
        sim.run_seconds(1.0)

        print("[controller] Шаг 3: захват (пауза)")
        sim.run_seconds(0.5)

        print("[controller] Шаг 4: возврат в home")
        home = [0.0] * max(sim.num_joints, 6)
        sim.move_to_joint_angles(home, speed=0.5)
        sim.run_seconds(1.0)

    def _load_json(self, path: str) -> Dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def shutdown(self):
        """Завершает работу симулятора."""
        self.sim.disconnect()