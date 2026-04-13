# scripts/test_urdf.py
import pybullet as p
import pybullet_data

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

try:
    robot = p.loadURDF(
        "simulation/models/mycobot_280/mycobot_280_m5.urdf",
        basePosition=[0, 0, 0.52],
        useFixedBase=True,
    )
    n = p.getNumJoints(robot)
    print(f"Робот загружен, джоинтов: {n}")
    for i in range(n):
        info = p.getJointInfo(robot, i)
        print(f"  [{i}] {info[1].decode()}")
    import time
    time.sleep(10)
except Exception as e:
    print(f"ОШИБКА: {e}")
finally:
    p.disconnect()