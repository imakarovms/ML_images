# ML_images — структура проекта

## Дерево папок

```text
.
├── core/                 # Базовые модули (конфиг, хеши, ORB, DSU, экспорт)
├── pipelines/            # Основные пайплайны группировки
├── training/             # Подготовка данных, обучение и инференс эмбеддингов
├── tools/                # Вспомогательные утилиты
├── embeddings/           # Готовые эмбеддинги и пути
├── models/               # Обученные модели
├── training_data/        # Датасеты для обучения/валидации
├── images/               # Входные изображения (ожидается по умолчанию)
├── main.py               # Совместимый wrapper для pipelines.main
├── main_nn.py            # Совместимый wrapper для pipelines.main_nn
├── nn_candidates.py      # Совместимый wrapper для pipelines.nn_candidates
├── infer_embeddings.py   # Совместимый wrapper для training.infer_embeddings
├── train_embedding.py    # Совместимый wrapper для training.train_embedding
└── prepare_training_data.py # Совместимый wrapper для training.prepare_training_data
```

## Что где

- `core/config.py` — все параметры запуска и пороги.
- `pipelines/main.py` — классический пайплайн (hash + ORB).
- `pipelines/main_nn.py` — пайплайн с нейросеточными кандидатами + ORB в 2 прохода.
- `pipelines/nn_candidates.py` — генерация candidate pairs из эмбеддингов.

## Как запускать

Из корня репозитория:

1. Сгенерировать NN-кандидатов:

```bash
python3 nn_candidates.py
```

2. Запустить NN-пайплайн группировки:

```bash
python3 main_nn.py
```

3. Альтернатива (классический пайплайн без NN-кандидатов):

```bash
python3 main.py
```

## Примечания

- Старые команды (`python3 main_nn.py`, `python3 main.py` и т.д.) сохранены через wrapper-файлы в корне.
- Основной рабочий код теперь разложен по папкам `core/`, `pipelines/`, `training/`, `tools/`.
