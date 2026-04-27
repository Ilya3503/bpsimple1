"""
scripts/experiment_runner.py
============================
Автоматический перебор параметров perception pipeline.

Запуск:
    python3 scripts/experiment_runner.py --input data/merged_*.ply
    python3 scripts/experiment_runner.py --input data/merged_*.ply --cad cad_models/Cube_30х30х30.ply
    python3 scripts/experiment_runner.py --input data/merged_*.ply --mode quick
    python3 scripts/experiment_runner.py --input data/merged_*.ply --mode full --cad cad_models/Cube_30х30х30.ply

Режимы:
    quick  — ~30 комбинаций, ~2 минуты, для быстрой проверки
    full   — ~200 комбинаций, ~15 минут, для полного исследования
    custom — параметры задаются в секции CUSTOM GRID ниже

Выходные файлы (в results/experiments/):
    results_TIMESTAMP.csv       — все прогоны, одна строка = один эксперимент
    results_TIMESTAMP.md        — Markdown таблица для отчёта
    best_TIMESTAMP.json         — лучший результат по ICP fitness
    summary_TIMESTAMP.txt       — человекочитаемый итог

Интерпретация результатов:
    n_clusters = 0          — параметры слишком агрессивные, объекты не найдены
    n_clusters > 10         — параметры слишком мягкие, шум считается объектами
    icp_fitness > 0.5       — хорошее выравнивание ICP
    icp_fitness < 0.2       — плохое выравнивание, нужно менять параметры
    points_after_ransac < 50 — слишком мало точек, pipeline деградирует
"""

import sys
import csv
import json
import time
import argparse
import itertools
from pathlib import Path
from datetime import datetime
from typing import Optional

# Добавляем корень проекта в путь
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.processing import (
    load_point_cloud,
    clean_point_cloud,
    crop_roi,
    voxel_downsample,
    remove_noise,
    remove_plane,
    cluster_dbscan,
    get_cluster_info,
    load_cad_model,
    estimate_pose_for_cluster,
)


# ==============================================================================
# СЕТКИ ПАРАМЕТРОВ
# ==============================================================================

# Базовые значения (фиксированы во всех экспериментах если не варьируются)
BASELINE = {
    "roi_x_min": -0.5, "roi_x_max": 0.5,
    "roi_y_min": -0.25, "roi_y_max": 0.25,
    "roi_z_min": 0.50, "roi_z_max": 0.75,
    "nb_neighbors": 20,
    "std_ratio": 2.0,
    "remove_table": True,
    "plane_distance_threshold": 0.01,
    "plane_num_iterations": 1000,
    "min_extent": 0.02,
    "max_extent": 0.30,
    "icp_voxel_size": 0.003,
}

# Быстрый режим — ключевые параметры, ~30 комбинаций
GRID_QUICK = {
    "voxel_size":   [0.005, 0.01, 0.02],
    "eps":          [0.02, 0.03, 0.05],
    "min_points":   [20, 50],
    # остальные из BASELINE
}

# Полный режим — систематическое исследование, ~200 комбинаций
GRID_FULL = {
    "voxel_size":   [0.003, 0.005, 0.008, 0.01, 0.015, 0.02],
    "eps":          [0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
    "min_points":   [10, 20, 30, 50],
    "plane_distance_threshold": [0.005, 0.01, 0.02],
    # остальные из BASELINE
}

# Кастомный режим — редактируй под свои нужды
GRID_CUSTOM = {
    "voxel_size":   [0.005, 0.01],
    "eps":          [0.02, 0.03],
    "min_points":   [20, 30],
}


# ==============================================================================
# ОДИН ЭКСПЕРИМЕНТ — ПРЯМОЙ ВЫЗОВ ФУНКЦИЙ
# ==============================================================================

def run_single_experiment(
    input_file: str,
    params: dict,
    cad_file: Optional[str] = None,
    suppress_print: bool = True,
) -> dict:
    """
    Прогоняет один эксперимент напрямую через функции processing.py.
    Возвращает словарь с входными параметрами и выходными метриками.

    Преимущество перед HTTP: перехватываем промежуточные данные
    (сколько точек на каждом шаге) что через API недоступно.
    """

    result = {
        # Входные параметры
        "input_file": Path(input_file).name,
        "voxel_size": params.get("voxel_size", BASELINE["voxel_size"] if "voxel_size" in BASELINE else 0.01),
        "eps": params.get("eps", 0.03),
        "min_points": params.get("min_points", 30),
        "min_extent": params.get("min_extent", BASELINE["min_extent"]),
        "max_extent": params.get("max_extent", BASELINE["max_extent"]),
        "plane_distance_threshold": params.get("plane_distance_threshold", BASELINE["plane_distance_threshold"]),
        "nb_neighbors": params.get("nb_neighbors", BASELINE["nb_neighbors"]),
        "std_ratio": params.get("std_ratio", BASELINE["std_ratio"]),
        "icp_voxel_size": params.get("icp_voxel_size", BASELINE["icp_voxel_size"]),
        "cad_file": Path(cad_file).name if cad_file else "none",

        # Выходные метрики — заполняются ниже
        "points_raw": 0,
        "points_after_clean": 0,
        "points_after_roi": 0,
        "points_after_voxel": 0,
        "points_after_noise": 0,
        "points_after_ransac": 0,
        "n_clusters": 0,
        "best_cluster_points": 0,
        "best_cluster_extent_x": 0.0,
        "best_cluster_extent_y": 0.0,
        "best_cluster_extent_z": 0.0,
        "max_icp_fitness": None,
        "best_icp_rmse": None,
        "best_cluster_method": "none",
        "processing_time_sec": 0.0,
        "status": "ok",
        "error": "",
    }

    t_start = time.time()

    try:
        # Параметры
        voxel_size = params.get("voxel_size", 0.01)
        nb_neighbors = params.get("nb_neighbors", BASELINE["nb_neighbors"])
        std_ratio = params.get("std_ratio", BASELINE["std_ratio"])
        eps = params.get("eps", 0.03)
        min_points = params.get("min_points", 30)
        min_extent = params.get("min_extent", BASELINE["min_extent"])
        max_extent = params.get("max_extent", BASELINE["max_extent"])
        plane_dt = params.get("plane_distance_threshold", BASELINE["plane_distance_threshold"])
        icp_voxel = params.get("icp_voxel_size", BASELINE["icp_voxel_size"])

        roi = {
            "x_min": params.get("roi_x_min", BASELINE["roi_x_min"]),
            "x_max": params.get("roi_x_max", BASELINE["roi_x_max"]),
            "y_min": params.get("roi_y_min", BASELINE["roi_y_min"]),
            "y_max": params.get("roi_y_max", BASELINE["roi_y_max"]),
            "z_min": params.get("roi_z_min", BASELINE["roi_z_min"]),
            "z_max": params.get("roi_z_max", BASELINE["roi_z_max"]),
        }

        # ── Шаг 1: Загрузка ──
        pcd = load_point_cloud(input_file)
        result["points_raw"] = len(pcd.points)

        # ── Шаг 2: Очистка NaN ──
        pcd = clean_point_cloud(pcd)
        result["points_after_clean"] = len(pcd.points)

        # ── Шаг 3: ROI crop ──
        pcd = crop_roi(pcd, **roi)
        result["points_after_roi"] = len(pcd.points)
        if len(pcd.points) == 0:
            result["status"] = "empty_after_roi"
            result["processing_time_sec"] = round(time.time() - t_start, 3)
            return result

        # ── Шаг 4: Voxel downsample ──
        pcd = voxel_downsample(pcd, voxel_size)
        result["points_after_voxel"] = len(pcd.points)

        # ── Шаг 5: Noise removal ──
        pcd = remove_noise(pcd, nb_neighbors, std_ratio)
        result["points_after_noise"] = len(pcd.points)
        if len(pcd.points) == 0:
            result["status"] = "empty_after_noise"
            result["processing_time_sec"] = round(time.time() - t_start, 3)
            return result

        # ── Шаг 6: RANSAC plane ──
        pcd, _ = remove_plane(pcd, distance_threshold=plane_dt,
                               ransac_n=3, num_iterations=1000)
        result["points_after_ransac"] = len(pcd.points)

        # ── Шаг 7: DBSCAN ──
        clusters = cluster_dbscan(pcd, eps=eps, min_points=min_points,
                                  min_extent=min_extent, max_extent=max_extent)
        result["n_clusters"] = len(clusters)

        if not clusters:
            result["status"] = "no_clusters"
            result["processing_time_sec"] = round(time.time() - t_start, 3)
            return result

        # ── Шаг 8: Pose estimation ──
        cad_model = load_cad_model(cad_file) if cad_file else None

        best_fitness = -1.0
        best_cluster = None
        best_pose = None

        for cluster in clusters:
            pose = estimate_pose_for_cluster(cluster, cad_model, icp_voxel)
            fitness = pose.get("fitness") or 0.0
            if fitness > best_fitness:
                best_fitness = fitness
                best_cluster = cluster
                best_pose = pose

        # Метрики лучшего кластера
        info = get_cluster_info(best_cluster, 0)
        result["best_cluster_points"] = info["points_count"]
        result["best_cluster_extent_x"] = round(info["extent"][0], 4)
        result["best_cluster_extent_y"] = round(info["extent"][1], 4)
        result["best_cluster_extent_z"] = round(info["extent"][2], 4)
        result["best_cluster_method"] = best_pose.get("method", "unknown")

        if best_fitness > 0:
            result["max_icp_fitness"] = round(best_fitness, 4)
            result["best_icp_rmse"] = round(best_pose.get("inlier_rmse") or 0.0, 6)
        else:
            result["max_icp_fitness"] = None
            result["best_icp_rmse"] = None

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:120]

    result["processing_time_sec"] = round(time.time() - t_start, 3)
    return result


# ==============================================================================
# GRID SEARCH
# ==============================================================================

def build_combinations(grid: dict) -> list:
    """Строит список всех комбинаций параметров из сетки."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        params = dict(BASELINE)  # начинаем с baseline
        params.update(dict(zip(keys, combo)))  # перекрываем варьируемыми
        combos.append(params)
    return combos


# ==============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==============================================================================

# Порядок колонок в CSV/MD
COLUMNS = [
    "voxel_size", "eps", "min_points", "plane_distance_threshold",
    "nb_neighbors", "std_ratio", "min_extent", "max_extent", "icp_voxel_size",
    "cad_file",
    "points_raw", "points_after_roi", "points_after_voxel",
    "points_after_noise", "points_after_ransac",
    "n_clusters", "best_cluster_points",
    "best_cluster_extent_x", "best_cluster_extent_y", "best_cluster_extent_z",
    "max_icp_fitness", "best_icp_rmse", "best_cluster_method",
    "processing_time_sec", "status", "error",
]

def save_csv(results: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[runner] CSV сохранён: {path}")


def save_markdown(results: list, path: Path, input_file: str, mode: str):
    """Генерирует Markdown таблицу для вставки в отчёт."""
    lines = []
    lines.append(f"# Результаты эксперимента")
    lines.append(f"")
    lines.append(f"**Входной файл:** `{Path(input_file).name}`  ")
    lines.append(f"**Режим:** {mode}  ")
    lines.append(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Всего прогонов:** {len(results)}  ")
    lines.append(f"")

    # Статистика
    ok = [r for r in results if r["status"] == "ok"]
    no_cl = [r for r in results if r["status"] == "no_clusters"]
    err = [r for r in results if r["status"] == "error"]
    lines.append(f"**Успешных:** {len(ok)} | **Без кластеров:** {len(no_cl)} | **Ошибок:** {len(err)}")
    lines.append(f"")

    # Лучший результат
    with_fitness = [r for r in ok if r.get("max_icp_fitness") is not None]
    if with_fitness:
        best = max(with_fitness, key=lambda r: r["max_icp_fitness"])
        lines.append(f"## Лучший результат по ICP fitness")
        lines.append(f"")
        lines.append(f"| Параметр | Значение |")
        lines.append(f"|---|---|")
        lines.append(f"| ICP fitness | **{best['max_icp_fitness']:.4f}** |")
        lines.append(f"| ICP RMSE | {best['best_icp_rmse']} |")
        lines.append(f"| voxel_size | {best['voxel_size']} |")
        lines.append(f"| eps | {best['eps']} |")
        lines.append(f"| min_points | {best['min_points']} |")
        lines.append(f"| n_clusters | {best['n_clusters']} |")
        lines.append(f"| points_after_ransac | {best['points_after_ransac']} |")
        lines.append(f"")

    # Таблица всех прогонов (только ключевые колонки для читаемости)
    lines.append(f"## Все прогоны")
    lines.append(f"")
    md_cols = [
        "voxel_size", "eps", "min_points",
        "points_after_voxel", "points_after_ransac",
        "n_clusters", "best_cluster_points",
        "max_icp_fitness", "processing_time_sec", "status",
    ]
    lines.append("| " + " | ".join(md_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(md_cols)) + " |")

    for r in results:
        row = []
        for col in md_cols:
            val = r.get(col, "")
            if val is None:
                val = "—"
            elif isinstance(val, float):
                val = f"{val:.4f}" if col == "max_icp_fitness" else f"{val:.3f}"
            row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[runner] Markdown сохранён: {path}")


def save_summary(results: list, path: Path, elapsed_total: float):
    """Человекочитаемый итог эксперимента."""
    ok = [r for r in results if r["status"] == "ok"]
    with_fitness = [r for r in ok if r.get("max_icp_fitness") is not None]

    lines = []
    lines.append("=" * 60)
    lines.append("ИТОГИ ЭКСПЕРИМЕНТА")
    lines.append("=" * 60)
    lines.append(f"Всего прогонов:      {len(results)}")
    lines.append(f"Успешных:            {len(ok)}")
    lines.append(f"Без кластеров:       {sum(1 for r in results if r['status']=='no_clusters')}")
    lines.append(f"Ошибок:              {sum(1 for r in results if r['status']=='error')}")
    lines.append(f"Общее время:         {elapsed_total:.1f}с ({elapsed_total/60:.1f} мин)")
    lines.append(f"Среднее на прогон:   {elapsed_total/len(results):.2f}с")
    lines.append("")

    if ok:
        avg_cl = sum(r["n_clusters"] for r in ok) / len(ok)
        lines.append(f"Среднее кластеров:   {avg_cl:.1f}")
        max_cl = max(r["n_clusters"] for r in ok)
        lines.append(f"Макс кластеров:      {max_cl}")

    if with_fitness:
        best = max(with_fitness, key=lambda r: r["max_icp_fitness"])
        lines.append("")
        lines.append("ЛУЧШИЙ ICP FITNESS:")
        lines.append(f"  fitness={best['max_icp_fitness']:.4f}  rmse={best['best_icp_rmse']}")
        lines.append(f"  voxel_size={best['voxel_size']}  eps={best['eps']}  min_points={best['min_points']}")
        lines.append(f"  n_clusters={best['n_clusters']}  points_after_ransac={best['points_after_ransac']}")
        lines.append("")
        lines.append("Диапазон fitness:")
        fitnesses = [r["max_icp_fitness"] for r in with_fitness]
        lines.append(f"  min={min(fitnesses):.4f}  max={max(fitnesses):.4f}  "
                     f"mean={sum(fitnesses)/len(fitnesses):.4f}")
    else:
        lines.append("\nICP не запускался (CAD модель не задана)")
        lines.append("Запустите с --cad cad_models/YOUR_MODEL.ply для ICP метрик")

    lines.append("=" * 60)
    text = "\n".join(lines)
    print(text)
    path.write_text(text, encoding="utf-8")
    print(f"[runner] Summary сохранён: {path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Автоматический перебор параметров perception pipeline"
    )
    parser.add_argument(
        "--input", required=True,
        help="Путь к PLY файлу (например: data/merged_2026-04-07_14-34-50.ply)"
    )
    parser.add_argument(
        "--cad", default=None,
        help="Путь к CAD PLY файлу для ICP (например: cad_models/Cube_30х30х30.ply)"
    )
    parser.add_argument(
        "--mode", default="quick", choices=["quick", "full", "custom"],
        help="quick (~30 комбинаций) | full (~200) | custom (GRID_CUSTOM в скрипте)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Папка для результатов (по умолчанию: results/experiments/)"
    )
    args = parser.parse_args()

    # Проверяем входной файл
    input_path = Path(args.input)
    if not input_path.exists():
        # Пробуем найти самый свежий merged файл
        data_dir = ROOT / "data"
        merged_files = sorted(data_dir.glob("merged_*.ply"), key=lambda f: f.stat().st_mtime)
        if merged_files:
            input_path = merged_files[-1]
            print(f"[runner] Файл не найден, используем последний merged: {input_path.name}")
        else:
            # Берём последний любой PLY
            all_ply = sorted(data_dir.glob("*.ply"), key=lambda f: f.stat().st_mtime)
            if not all_ply:
                print(f"[runner] ОШИБКА: не найден ни один PLY файл в {data_dir}")
                sys.exit(1)
            input_path = all_ply[-1]
            print(f"[runner] Используем последний PLY: {input_path.name}")

    # Выходная папка
    out_dir = Path(args.out) if args.out else ROOT / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Выбираем сетку
    if args.mode == "quick":
        grid = GRID_QUICK
    elif args.mode == "full":
        grid = GRID_FULL
    else:
        grid = GRID_CUSTOM

    combos = build_combinations(grid)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT RUNNER")
    print(f"{'='*60}")
    print(f"Входной файл: {input_path.name}")
    print(f"CAD модель:   {Path(args.cad).name if args.cad else 'не задана (OBB fallback)'}")
    print(f"Режим:        {args.mode}")
    print(f"Комбинаций:   {len(combos)}")
    print(f"Результаты:   {out_dir}")
    print(f"{'='*60}\n")

    results = []
    t_total = time.time()

    for i, params in enumerate(combos):
        pct = (i + 1) / len(combos) * 100
        elapsed = time.time() - t_total
        eta = (elapsed / (i + 1)) * (len(combos) - i - 1) if i > 0 else 0

        print(f"[{i+1:3d}/{len(combos)}] {pct:.0f}% | "
              f"voxel={params.get('voxel_size', '?')} "
              f"eps={params.get('eps', '?')} "
              f"min_pts={params.get('min_points', '?')} | "
              f"ETA: {eta:.0f}s", end="  ")

        r = run_single_experiment(
            input_file=str(input_path),
            params=params,
            cad_file=args.cad,
            suppress_print=True,
        )
        results.append(r)

        # Краткий статус в строку
        if r["status"] == "ok":
            fitness_str = f"fitness={r['max_icp_fitness']:.3f}" if r["max_icp_fitness"] else "no_icp"
            print(f"✓ clusters={r['n_clusters']} pts_ransac={r['points_after_ransac']} {fitness_str}")
        else:
            print(f"✗ {r['status']}: {r['error'][:50]}")

    elapsed_total = time.time() - t_total

    # Сохраняем
    csv_path = out_dir / f"results_{ts}.csv"
    md_path = out_dir / f"results_{ts}.md"
    summary_path = out_dir / f"summary_{ts}.txt"

    save_csv(results, csv_path)
    save_markdown(results, md_path, str(input_path), args.mode)
    save_summary(results, summary_path, elapsed_total)

    # Лучший результат отдельно
    with_fitness = [r for r in results if r.get("max_icp_fitness") is not None]
    if with_fitness:
        best = max(with_fitness, key=lambda r: r["max_icp_fitness"])
        best_path = out_dir / f"best_{ts}.json"
        best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[runner] Лучший результат: {best_path}")

    print(f"\n[runner] Готово. Файлы в: {out_dir}")
    print(f"[runner] Открыть таблицу: {md_path.name}")


if __name__ == "__main__":
    main()