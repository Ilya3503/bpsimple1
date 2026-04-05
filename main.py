from fastapi import FastAPI, Query, HTTPException
from app.camera import capture_pointcloud
from app.processing import process_pointcloud
from robot.controller import RobotController
from app.merge import merge_point_cloud_files, get_two_latest_files

app = FastAPI(
    title="Point Cloud Perception API",
    description="Perception pipeline: depth camera → clustering → pose estimation",
)


@app.get("/capture", tags=["Камера"])
def capture():
    """Захват облака точек с RealSense камеры."""
    filepath = capture_pointcloud()
    return {"status": "ok", "file": filepath}


@app.post("/process_pointcloud", tags=["Обработка"], summary="Обработка облака точек")
def process_endpoint(
        # --- Источник ---
        use_latest: bool = Query(True, description="Взять последний файл из папки"),
        folder: str = Query("data", description="Папка с файлами"),
        input_file: str = Query(None, description="Путь к файлу (если use_latest=False)"),
        results_dir: str = Query("results", description="Папка для результатов"),

        # --- ROI (метры) ---
        roi_x_min: float = Query(-0.5, description="ROI X мин (м)"),
        roi_x_max: float = Query(0.5, description="ROI X макс (м)"),
        roi_y_min: float = Query(-0.25, description="ROI Y мин (м)"),
        roi_y_max: float = Query(0.25, description="ROI Y макс (м)"),
        roi_z_min: float = Query(0.50, description="ROI Z мин (м)"),
        roi_z_max: float = Query(0.75, description="ROI Z макс (м)"),

        # --- Предобработка ---
        voxel_size: float = Query(0.02, description="Размер вокселя (м), рекомендуется 0.005"),
        nb_neighbors: int = Query(20, description="Соседи для статистической фильтрации"),
        std_ratio: float = Query(2.0, description="Коэф. стд. отклонения для фильтрации"),

        # --- RANSAC ---
        remove_table: bool = Query(True, description="Удалять плоскость стола"),
        plane_distance_threshold: float = Query(0.01, description="Допуск RANSAC (м)"),
        plane_num_iterations: int = Query(1000, description="Итерации RANSAC"),

        # --- DBSCAN ---
        eps: float = Query(0.03, description="Радиус соседства DBSCAN (м)"),
        min_points: int = Query(30, description="Мин. точек в кластере"),
        max_points: int = Query(None, description="Макс. точек в кластере (опционально)"),
        min_extent: float = Query(0.02, description="Мин. размер кластера (м), отсекает мусор"),
        max_extent: float = Query(0.30, description="Макс. размер кластера (м), отсекает фон"),

        # --- ICP ---
        cad_file: str = Query(None, description="Путь к CAD-модели .ply (опционально)"),
):
    try:
        result = process_pointcloud(
            input_file=input_file,
            results_dir=results_dir,
            use_latest=use_latest,
            folder=folder,
            roi_x_min=roi_x_min, roi_x_max=roi_x_max,
            roi_y_min=roi_y_min, roi_y_max=roi_y_max,
            roi_z_min=roi_z_min, roi_z_max=roi_z_max,
            voxel_size=voxel_size,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
            remove_table=remove_table,
            plane_distance_threshold=plane_distance_threshold,
            plane_num_iterations=plane_num_iterations,
            eps=eps,
            min_points=min_points,
            max_points=max_points,
            min_extent=min_extent,
            max_extent=max_extent,
            cad_file=cad_file,
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {e}")







@app.post("/execute", tags=["Робот"], summary="Выполнить захват объекта")
def execute_endpoint(
    json_path: str = Query("results/position.json", description="Путь к position.json"),
    robot_urdf: str = Query(None, description="Путь к URDF робота (опционально)"),
    use_gui: bool = Query(True, description="Показывать окно PyBullet"),
    grasp_offset_z: float = Query(0.05, description="Высота захвата над объектом (м)"),
):
    """
    Запускает полный цикл:
    position.json → grasp pose → IK → PyBullet симуляция

    Пока нет URDF — симулятор запускается со сценой без робота.
    Куб и стол будут видны в окне PyBullet.
    """
    try:
        controller = RobotController(
            robot_urdf=robot_urdf,
            use_gui=use_gui,
            grasp_offset_z=grasp_offset_z,
        )
        result = controller.execute_from_json(json_path)
        controller.shutdown()
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения: {e}")











@app.post("/merge", tags=["Съёмка"], summary="Объединить два облака точек")
def merge_endpoint(

    # --- Источник файлов ---
    use_latest: bool = Query(
        True,
        description="Взять два последних файла из папки автоматически"
    ),
    folder: str = Query(
        "data",
        description="Папка для поиска файлов (если use_latest=True)"
    ),

    # --- Явное указание файлов ---
    file_a: str = Query(
        None,
        description="Путь к файлу A (позиция камеры 1). "
                    "Используется если use_latest=False"
    ),
    file_b: str = Query(
        None,
        description="Путь к файлу B (позиция камеры 2). "
                    "Используется если use_latest=False"
    ),

    # --- Параметры ---
    output_dir: str = Query(
        "data",
        description="Папка для сохранения merged файла"
    ),
    voxel_size: float = Query(
        0.005,
        description="Вокселизация после merge (м). 0 — отключить"
    ),
):
    """
    Объединяет два облака точек из разных позиций камеры.

    Два режима выбора файлов:

    **Автоматически (use_latest=True):**
    Берёт два последних .ply файла из папки по timestamp.
    Удобно если только что сделал два /capture подряд.

    **Вручную (use_latest=False):**
    Передаёшь явные пути file_a и file_b.
    Удобно если хочешь объединить конкретные файлы.

    **Трансформация между позициями камеры:**
    Сейчас используется заглушка (единичная матрица).
    Объекты могут задвоиться — это ожидаемо.
    После калибровки напарник заполняет DEFAULT_T_B_TO_A
    в файле app/merge.py реальными значениями.

    **Результат:**
    Сохраняет merged_TIMESTAMP.ply в output_dir.
    Путь к файлу можно сразу передать в /process_pointcloud.
    """
    try:
        # Определяем файлы
        if use_latest:
            files = get_two_latest_files(folder)
            resolved_a = str(files[0])
            resolved_b = str(files[1])
        else:
            if file_a is None or file_b is None:
                raise HTTPException(
                    status_code=400,
                    detail="При use_latest=False нужно передать file_a и file_b"
                )
            resolved_a = file_a
            resolved_b = file_b

        merged_path, is_stub = merge_point_cloud_files(
            file_a=resolved_a,
            file_b=resolved_b,
            output_dir=output_dir,
            T_b_to_a=None,   # None = использует заглушку из merge.py
            voxel_size=voxel_size,
        )

        return {
            "status": "ok",
            "file_a": resolved_a,
            "file_b": resolved_b,
            "merged_file": merged_path,
            "transform_is_stub": is_stub,
            "note": (
                "Трансформация — заглушка (единичная матрица). "
                "Заполни DEFAULT_T_B_TO_A в app/merge.py после калибровки."
            ) if is_stub else "Трансформация применена.",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка merge: {e}")