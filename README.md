# **Robotic Perception & Manipulation System**

Простая система машинного зрения и манипуляции: от съёмки RealSense → обработка облака точек → захват робота → загрузка детали в станок ЧПУ.

### Основные возможности

- Захват облака точек с RealSense D415
- Обработка: фильтрация, удаление стола, кластеризация объектов
- Определение позы объектов
- Симуляция в PyBullet: 6-осевой робот + стол + **станок ЧПУ**
- Полноценный **Pick-and-Place** — робот берёт куб и кладёт его в ЧПУ
- FastAPI веб-интерфейс (Swagger)

---

### Структура проекта

```
/
├── main.py                 # FastAPI сервер
├── app/
│   ├── camera.py           # Съёмка с RealSense
│   ├── processing.py       # Обработка облака точек
│   ├── merge.py            # Объединение нескольких ракурсов
│   └── utils.py
├── robot/
│   ├── controller.py       # Главная логика (grasp + IK)
│   ├── grasp.py            # Планирование захвата
│   └── kinematics.py       # IK (ikpy)
├── simulation/
│   └── bridge.py           # PyBullet + сцена (стол + ЧПУ)
├── data/                   # .ply файлы
├── results/                # position.json и результаты
└── requirements.txt
```

---

### Как запустить

#### 1. Установка
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

#### 2. Запуск сервера
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --workers=1
```


### Основные эндпоинты

| Метод | Эндпоинт                    | Что делает                                            |
|-------|-----------------------------|-------------------------------------------------------|
| GET   | `/capture`                  | Сделать один снимок                                   |
| POST  | `/merge`                    | Объединить 2 последних снимка                         |
| POST  | `/process_pointcloud`       | Обработать облако → `position.json`                   |
| POST  | `/execute`                  | **Полный цикл работы робота**: захват + перенос в ЧПУ |