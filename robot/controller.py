import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from robot.grasp import select_best_cluster, compute_grasp_pose, compute_pregrasp_pose
from robot.kinematics import RobotKinematics
from simulation.bridge import SimulationBridge


class RobotController:
    """
    Главный контроллер: perception → grasp → IK → PyBullet.

    Использование без URDF (только сцена):
        controller = RobotController()
        controller.execute_from_json("results/position.json")

    С URDF mycobot:
        controller = RobotController(
            robot_urdf="simulation/models/mycobot_280/mycobot_280_m5.urdf"
        )
    """

    def __init__(
        self,
        robot_urdf: "simulation/models/mycobot_280/mycobot_280_m5.urdf",
        use_gui: bool = True,
        grasp_offset_z: float = 0.05,
        pregrasp_offset_z: float = 0.12,
    ):
        self.grasp_offset_z = grasp_offset_z
        self.pregrasp_offset_z = pregrasp_offset_z

        self.kinematics = RobotKinematics(urdf_path=robot_urdf)
        self.sim = SimulationBridge(robot_urdf=robot_urdf, use_gui=use_gui)

    def execute_from_json(self, json_path: str) -> Dict:
        """
        Полный цикл захвата:
            1. Читаем position.json
            2. Выбираем лучший кластер
            3. Загружаем объект в сцену
            4. Вычисляем grasp pose
            5. Решаем IK
            6. Двигаем робота
        """
        print(f"\n{'='*50}")
        print(f"[controller] Старт: {json_path}")

        # ====================== ЗАГЛУШКА ДЛЯ ТЕСТИРОВАНИЯ ======================

        object_pos = [0.18, 0.05, 0.25]         # X=18см, Y=5см, Z=25см — удобно для myCobot 280
        print(f"[controller] ЗАГЛУШКА ВКЛЮЧЕНА → Принудительная позиция кубика: {object_pos}")
        print(f"[controller] Реальные координаты из position.json игнорируются")

        # Загружаем объект в сцену
        self.sim.load_object(
            position=object_pos,
            extent=[0.05, 0.05, 0.05],      # размер кубика 5см
            orientation=[0, 0, 0, 1],
        )

        # Вычисляем позы захвата
        grasp = compute_grasp_pose(None, self.grasp_offset_z, object_pos=object_pos)
        pregrasp = compute_pregrasp_pose(grasp, self.pregrasp_offset_z)

        print(f"[controller] Pre-grasp позиция: {[round(v,3) for v in pregrasp['position']]}")
        print(f"[controller] Grasp позиция:     {[round(v,3) for v in grasp['position']]}")

        # Решаем Inverse Kinematics
        ik_pre = self.kinematics.solve_ik(pregrasp["position"], pregrasp["orientation"])
        ik_grasp = self.kinematics.solve_ik(grasp["position"], grasp["orientation"])

        print(f"[controller] IK pre-grasp reachable = {ik_pre.get('reachable', False)}")
        print(f"[controller] IK grasp reachable     = {ik_grasp.get('reachable', False)}")

        # Выполняем движение
        self._execute_grasp_sequence(
            ik_pre.get("joint_angles", [0.0] * 6),
            ik_grasp.get("joint_angles", [0.0] * 6)
        )

        result = {
            "status": "ok",
            "mode": "stub",
            "object_position": object_pos,
            "grasp_position": grasp["position"],
            "pregrasp_position": pregrasp["position"],
            "ik_pregrasp": ik_pre,
            "ik_grasp": ik_grasp,
        }

        print(f"[controller] Симуляция завершена")
        print(f"{'='*60}\n")
        return result


        # ====================== СТАРЫЙ КОД ======================

        # perception = self._load_json(json_path)

        # if perception.get("status") != "ok":
        #     raise ValueError(f"Статус perception: {perception.get('status')}")

        # clusters = perception.get("clusters", [])
        # if not clusters:
        #     raise ValueError("Нет кластеров в position.json")

        # print(f"[controller] Кластеров: {len(clusters)}")

        # # --- Выбираем кластер ---
        # best = select_best_cluster(clusters)
        # pose = best["pose"]
        # print(f"[controller] Кластер {best['id']}: "
        #       f"{best['points_count']} точек | "
        #       f"метод={pose['method']} | "
        #       f"pos={[round(v,3) for v in pose['position']]}")

        # # --- Загружаем объект в сцену ---
        # self.sim.load_object(
        #     position=pose["position"],
        #     extent=pose["extent"],
        #     orientation=pose["orientation"],
        # )

        # # --- Grasp pose ---
        # grasp = compute_grasp_pose(best, self.grasp_offset_z)
        # pregrasp = compute_pregrasp_pose(grasp, self.pregrasp_offset_z)

        # print(f"[controller] Pre-grasp: {[round(v,3) for v in pregrasp['position']]}")
        # print(f"[controller] Grasp:     {[round(v,3) for v in grasp['position']]}")

        # # --- IK ---
        # ik_pre = self.kinematics.solve_ik(pregrasp["position"], pregrasp["orientation"])
        # ik_grasp = self.kinematics.solve_ik(grasp["position"], grasp["orientation"])

        # print(f"[controller] IK pre-grasp: stub={ik_pre['is_stub']} reachable={ik_pre['reachable']}")
        # print(f"[controller] IK grasp:     stub={ik_grasp['is_stub']} reachable={ik_grasp['reachable']}")

        # # --- Движение ---
        # self._execute_grasp_sequence(ik_pre["joint_angles"], ik_grasp["joint_angles"])

        # result = {
        #     "status": "ok",
        #     "cluster_id": best["id"],
        #     "object_position": pose["position"],
        #     "grasp_position": grasp["position"],
        #     "pregrasp_position": pregrasp["position"],
        #     "ik_pregrasp": ik_pre,
        #     "ik_grasp": ik_grasp,
        # }

        # print(f"[controller] Готово")
        # print(f"{'='*50}\n")
        # return result

    def _execute_grasp_sequence(
        self,
        pregrasp_angles: List[float],
        grasp_angles: List[float],
    ):
        sim = self.sim

        print("[controller] → Pre-grasp")
        sim.move_to_joint_angles(pregrasp_angles, speed=0.5)
        sim.run_seconds(5)

        print("[controller] → Grasp")
        sim.move_to_joint_angles(grasp_angles, speed=0.3)
        sim.run_seconds(5)

        print("[controller] → Захват (пауза)")
        sim.run_seconds(5)

        print("[controller] → Возврат в home")
        home = [0.0] * max(sim.num_joints, 6)
        sim.move_to_joint_angles(home, speed=0.5)
        sim.run_seconds(5)

        # Держим окно открытым для демонстрации
        print("[controller] Сцена открыта. Нажми Enter для закрытия...")
        input()

    def _load_json(self, path: str) -> Dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def shutdown(self):
        """Завершает работу симулятора безопасно."""
        try:
            self.sim.disconnect()
        except Exception as e:
            print(f"[controller] Warning during shutdown: {e}")