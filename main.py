from fastapi import FastAPI, Query, HTTPException
from app.camera import capture_pointcloud
from app.processing import process_pointcloud

app = FastAPI()


@app.get("/capture")
def capture():

    filepath = capture_pointcloud()

    return {
        "status": "ok",
        "file": filepath
    }


@app.post("/process_pointcloud", tags=["Обработка облака точек"], summary="Обработка облака точек с параметрами")
def process_endpoint(
        use_latest: bool = Query(True, description="Брать последний файл или нет"),
        folder: str = Query("data", description="Папка для поиска файлов, если use_latest=True"),
        input_file: str = Query(None, description="Путь к файлу, если use_latest=False"),
        results_dir: str = Query("results", description="Папка для сохранения результатов"),

        voxel_size: float = Query(0.0, description="Размер вокселя для даунсэмплинга"),
        nb_neighbors: int = Query(20, description="Количество соседей для фильтрации шума"),
        std_ratio: float = Query(2.0, description="Коэффициент стандартного отклонения при фильтрации"),
        distance_threshold: float = Query(70.0, description="Порог расстояния при удалении плоскости"),
        ransac_n: int = Query(3, description="Количество точек для RANSAC"),
        num_iterations: int = Query(1000, description="Количество итераций RANSAC"),
        min_bound_x: float = Query(-231, description="Мин. X для обрезки"),
        min_bound_y: float = Query(-190, description="Мин. Y для обрезки"),
        min_bound_z: float = Query(474, description="Мин. Z для обрезки"),
        max_bound_x: float = Query(264, description="Макс. X для обрезки"),
        max_bound_y: float = Query(190, description="Макс. Y для обрезки"),
        max_bound_z: float = Query(670, description="Макс. Z для обрезки"),

        eps: float = Query(30.0, description="Параметр eps для DBSCAN"),
        min_points: int = Query(20, description="Минимальное количество точек для кластера"),
        max_points: int = Query(None, description="Максимальное количество точек для кластера"),

        cad_file: str = Query(None, description="Путь к CAD-модели для ICP (опционально)"),
):
    try:
        min_bound = (min_bound_x, min_bound_y, min_bound_z)
        max_bound = (max_bound_x, max_bound_y, max_bound_z)

        result = process_pointcloud(
            input_file=input_file,
            results_dir=results_dir,
            use_latest=use_latest,
            folder=folder,
            voxel_size=voxel_size,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            min_bound=min_bound,
            max_bound=max_bound,
            eps=eps,
            min_points=min_points,
            max_points=max_points,
            cad_file=cad_file
        )
        return result

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки облака точек: {exc}")