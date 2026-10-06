# embeddings-quality

Для извлечения DINOv3-эмбеддингов из исходных изображений и bbox доступен
пакет [dinov3-embedder](https://github.com/Wasilkas/dinov3-embedder). Его NPZ и CSV
метаданных совместимы с API и CLI этого пакета.

Python-пакет для оценки разделимости классов в **исходном пространстве
эмбеддингов SSL-модели**. Основной сценарий — bbox-level эмбеддинги SimCLR
для дефектов поверхности металла, выгруженные из Qdrant в NumPy.
Размерность признаков сохраняется во всех проверках.

## Установка

Требуется Python 3.10+ и установленный Git. Создайте и активируйте окружение:

```bash
python -m venv .venv
source .venv/bin/activate
```

В Windows PowerShell для активации используйте `.venv\Scripts\Activate.ps1`.
Если окружение уже создано, достаточно активировать его.

Установите пакет напрямую из GitHub вместе с зависимостями для графиков:

```bash
python -m pip install "embeddings-quality[plots] @ git+https://github.com/Wasilkas/embeddings-quality.git@main"
```

Без графиков уберите `[plots]`; для разработки, включая Ruff и mypy,
используйте `[dev,plots]`.

При настроенном SSH-доступе к GitHub можно использовать:

```bash
python -m pip install "embeddings-quality[plots] @ git+ssh://git@github.com/Wasilkas/embeddings-quality.git@main"
```

`@main` выбирает ветку; для воспроизводимой установки замените `main` на тег
или полный хеш коммита.

Для разработки из локальной копии, находясь в папке `embeddings-quality`:

```bash
python -m pip install -e '.[plots]'
```

Без графиков достаточно `python -m pip install -e .`.
Пакет принимает готовые эмбеддинги и метки: запуск SSL-модели не требуется.

## Python API

```python
from embeddings_quality import AuditConfig, evaluate_embeddings

report = evaluate_embeddings(
    embeddings,  # np.ndarray, (N, D), например (10000, 512)
    labels,  # (N,), int или str, один класс на объект
    sample_ids=point_ids,  # необязательно: уникальные ID Qdrant
    groups=image_ids,  # необязательно: image_id/coil_id/batch_id
    metadata={  # необязательно: массивы (N,)
        "bbox_area": bbox_areas,
        "padding_fraction": padding_fractions,
    },
    config=AuditConfig(ks=(5, 10, 20, 50), cv_folds=5),
)

print(report.classes)
print(report.pairwise_auc)
print(report.summary["linear_probe"])
report.save("reports/simclr", plots=True)
suspects = report.review_queue(limit=100)
```

Минимальный вызов — `evaluate_embeddings(embeddings, labels)`.
Порядок строк у меток, ID, групп и метаданных должен совпадать с порядком
эмбеддингов. Входные массивы не изменяются. Метки и ID в отчёте представлены
строками; число и строка, представляющие одно значение, считаются одной меткой.

`groups` обозначает источник зависимости между объектами. Несколько bbox
одного изображения имеют одинаковый `image_id`. Во всех probes группа целиком
попадает либо в train, либо в validation. По умолчанию объекты той же группы
также исключаются из соседей для purity/margin. Silhouette описывает геометрию
всего датасета и включает объекты той же группы.
Для обычной геометрии соседей вместе с групповым CV используйте
`AuditConfig(exclude_same_group=False)`. Без `groups` CV предполагает независимость объектов.

## Массивы из Qdrant

Если массивы уже сформированы, передайте их прямо в API. Пример преобразования
списка точек, полученного вашим кодом выгрузки:

```python
import numpy as np

# points — все выгруженные точки с vectors и payload; при scroll нужно
# собрать все страницы. Имена payload-полей замените своими.
embeddings = np.asarray([point.vector for point in points], dtype=np.float32)
labels = np.asarray([point.payload["class"] for point in points], dtype=str)
point_ids = np.asarray([point.id for point in points], dtype=str)
image_ids = np.asarray([point.payload["image_id"] for point in points], dtype=str)

# Для named vector замените point.vector на point.vector["simclr"].
# Вектор должен быть dense-массивом длины D.
np.savez_compressed(
    "simclr.npz",
    embeddings=embeddings,
    labels=labels,
    sample_ids=point_ids,
    groups=image_ids,
)
```

Метрики сравнивают эмбеддинги с имеющимися метками. Ошибочная разметка влияет
на оценки; без дополнительных экспериментов/ручной проверки нельзя отделить
шум меток от недостатков representation.

## CLI

```bash
embeddings-quality simclr.npz --output reports/simclr --plots
python -m embeddings_quality simclr.npz --output reports/simclr
```

Обязательные ключи NPZ: `embeddings`, `labels`. Необязательные: `sample_ids`,
`groups`. Строки сохраняйте с `dtype=str`: загрузка pickle/object массивов отключена.

Метаданные можно передать CSV. Столбец `sample_id` обязателен, уникален и должен
содержать все ID из NPZ. Строки соединяются по ID; дополнительные ID в CSV допускаются.

```csv
sample_id,bbox_area,padding_fraction,camera
101,2400,0.32,line_A
102,3200,0.18,line_B
```

```bash
embeddings-quality simclr.npz --output reports/simclr \
  --metadata bbox_metadata.csv --categorical-metadata camera --plots
```

Числовые nuisance-столбцы проверяются через RandomForestRegressor (OOF R²,
MAE и baseline, предсказывающий среднее train). Строковые категории — через
линейную классификацию. Для числовых ID камер/линий используйте
`--categorical-metadata` либо передавайте строки в Python API.
Столбцы с пропусками или постоянным значением получают статус `skipped`.

Настройки CLI: `--ks`, `--metric`, `--normalize`/`--no-normalize`,
`--cv-folds`, `--margin-k`, `--probe-k`, `--linear-c`, `--max-iter`,
`--seed`, `--working-memory-mb`, `--include-same-group-neighbors`, `--log-level`.

## Логи и прогресс анализа

Логирование использует [Loguru](https://loguru.readthedocs.io/), который входит
в зависимости пакета. CLI по умолчанию пишет INFO-логи в stderr: загрузка данных,
подготовка входов, геометрия, OOF probes, пары классов, nuisance-столбцы
и сохранение отчёта.
В логах видны текущий fold/пара классов и время выполнения этапа. Во время
обработки геометрии прогресс по объектам выводится примерно раз в 5 секунд
и при завершении. DEBUG также показывает расчёт отдельных блоков расстояний
и сохранение CSV-файлов.

```bash
embeddings-quality simclr.npz --output reports/simclr --log-level DEBUG
embeddings-quality simclr.npz --output reports/simclr --log-level WARNING
embeddings-quality simclr.npz --output reports/simclr 2>analysis.log
```

`WARNING` отключает сообщения о прогрессе. Итоговые сообщения CLI остаются
в stdout; notices также выводятся в stderr.

В Python API логи по умолчанию отключены. Включите их до начала анализа.
Если Loguru ещё не настроен вашим приложением, настройте вывод в консоль:

```python
import sys

from loguru import logger

from embeddings_quality import evaluate_embeddings

logger.remove()  # Заменяем стандартный sink Loguru своим.
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} {level} {message}",
    diagnose=False,
)
logger.enable("embeddings_quality")

# Далее обычный вызов evaluate_embeddings(...).
```

Если Loguru уже настроен вашим приложением или ноутбуком, достаточно
`logger.enable("embeddings_quality")`; существующий sink должен допускать INFO.
Python API не добавляет и не удаляет sinks. Отключение логов библиотеки:
`logger.disable("embeddings_quality")`. Для записи в файл после настройки
Loguru можно добавить `logger.add("analysis.log", level="INFO", diagnose=False)`.
CLI настраивает sinks для консольного запуска; в приложении с собственной
конфигурацией Loguru используйте Python API.
Отдельный вызов scikit-learn (расчёт блока расстояний или обучение модели)
может занять долгое время: внутри него обновлений прогресса нет, но в логах
виден текущий этап; для блоков расстояний используйте DEBUG.

## Метрики

| Проверка | Что показывает |
|---|---|
| kNN purity @ 5/10/20/50 | Доля соседей с той же меткой, без самого объекта |
| Purity baseline и lift | Преимущество над частотой класса среди допустимых соседей |
| Silhouette по объектам и классам | Компактность классов и расстояние до других классов |
| d_positive/d_negative/margin | Ближе ли объект к своему классу, чем к чужому |
| Multiclass linear probe | Линейная разделимость: OOF balanced accuracy, macro-F1, confusion |
| Multiclass kNN probe | Локальная предсказуемость классов на отложенных объектах |
| Попарный linear probe | Какие пары можно разделить: ROC-AUC и balanced accuracy |
| Nuisance probes | Предсказуемы ли bbox size, scale, padding, camera и другие метаданные |

По умолчанию — cosine distance и L2-нормализация каждого вектора. Она
сохраняет D координат и косинусные расстояния. Линейные probes обучаются
на тех же нормализованных векторах. Для исходных координат вместе с нормой:
`AuditConfig(metric="euclidean", normalize=False)`. Feature-wise scaling не применяется.
При `metric="cosine", normalize=False` геометрия остаётся cosine, а линейный
probe использует ненормализованные векторы.

Доступны расстояния `cosine`, `euclidean` и `manhattan`. Manhattan (L1) — сумма
абсолютных разностей координат: `sum(abs(x_j - y_j))`. Выбранное расстояние
используется в геометрии (purity, silhouette, margins) и kNN probe.
Для `euclidean` и `manhattan` L2-нормализация по умолчанию отключена;
её можно включить через `normalize=True` или CLI-флаг `--normalize`.

```python
report = evaluate_embeddings(embeddings, labels, config=AuditConfig(metric="manhattan"))
```

```bash
embeddings-quality simclr.npz --output reports/simclr --metric manhattan
```

Для объекта i:

```text
purity_i(k) = число соседей своего класса / k
baseline_i = число объектов своего класса среди допустимых соседей /
             число всех допустимых соседей
lift_i(k) = purity_i(k) / baseline_i

d_positive_i = median расстояний до min(margin_k, available) ближайших
               допустимых объектов своего класса
d_negative_i = расстояние до ближайшего допустимого объекта другого класса
margin_i = d_negative_i - d_positive_i
```

Если k больше числа допустимых соседей, purity отсутствует: k автоматически
не уменьшается. Если baseline=0, lift отсутствует. Без объектов своего класса
d_positive/margin отсутствуют. Фактическое число положительных соседей сохраняется
в `positive_neighbors_used`. Дубликаты остаются отдельными объектами; tie-break
при одинаковом расстоянии — порядок входных строк. В неоднозначных областях
смена порядка может изменить purity. Сам объект всегда исключён из соседей.

Silhouette точный для всех объектов. Для singleton-класса значение 0 соответствует
соглашению scikit-learn. Если каждый объект — отдельный класс, silhouette не определён.
Global mean взвешен числом объектов; macro mean даёт классам одинаковый вес.
В `classes.csv` есть mean, p10/p50/p90 и число валидных оценок каждой метрики.

LogisticRegression использует `class_weight="balanced"`, фиксированный C и
out-of-fold predictions. Hyperparameter tuning не проводится. CV: StratifiedKFold
либо StratifiedGroupKFold. Число folds уменьшается до доступного числа объектов/групп
каждого класса. Фактические train/test проверяются на наличие всех классов;
невозможное разбиение пропускается. Попарный probe использует только два класса.
Матрицы содержат **среднее метрик отдельных folds**. Fold standard deviation
сохранён в JSON; это не доверительный интервал.

## Интерпретация

Начните с OOF balanced accuracy/macro-F1 линейного probe и dummy baseline.
Высокие результаты на групповой validation поддерживают вывод о линейной
разделимости при данном способе разбиения. Попарный ROC-AUC покажет трудные пары.
Около 0.5 соответствует случайному ранжированию; около 1 — сильному ранжированию
отложенных объектов.

Низкая линейная оценка не доказывает отсутствие нелинейной разделимости.
Сопоставьте её с kNN probe, purity/lift и margin. Высокий purity крупного класса
сам по себе слабый результат: lift около 1 означает уровень частотного baseline.

Низкий silhouette может сочетаться с хорошей предсказуемостью: класс может
состоять из нескольких подтипов. Отрицательный margin выделяет объекты, у которых
чужой класс ближе медианы ближайших представителей своего. `review_queue.csv`
ранжирует объекты для ручного просмотра.

Высокий R² для padding/bbox size означает наличие этой информации в representation.
Чтобы установить, определяет ли она качество классификации, нужны сравнения
preprocessing на одних объектах, контроль по метаданным или специальный holdout.

Универсальный порог «эмбеддинги хорошие» не задаётся. Сравнивайте модели и
preprocessing на одинаковых объектах, метках, группах и настройках. Аудит
frozen features не подтверждает отсутствие утечки при обучении самой SSL-модели.

## Артефакты

- `summary.json`: параметры, размеры, общие метрики, folds, nuisance probes,
  причины пропусков и notices; неопределённые значения — null.
- `samples.csv`: ID/метки, геометрия и OOF prediction/probability исходной метки.
- `classes.csv`: размеры/частоты классов, средние, квантили, число валидных оценок.
- `pairwise_auc.csv`, `pairwise_balanced_accuracy.csv`: симметричные матрицы;
  диагональ и пропущенные пары пустые.
- `linear_confusion.csv`: confusion matrix по OOF predictions.
- `review_queue.csv`: все объекты по возрастанию margin, затем purity при минимальном k;
  неопределённые margin располагаются в конце.
- `REPORT.md`: основные результаты и пояснения.
- `pairwise_separability.svg`: heatmap при `plots=True`/`--plots`.

Повторный `save` в ту же директорию перезаписывает файлы отчёта.

## Вычисления

Расстояния считаются точно блоками, полная N×N матрица не хранится.
`working_memory_mb` ограничивает блок расстояний, а не всю RAM процесса.
Время расстояний — O(N²D), сортировка добавляет O(N² log N).
Для больших экспортов используйте фиксированную репрезентативную выборку с нужными
классами/группами, отражая её состав при сравнении. Попарных probes — C(C−1)/2.
GPU и Qdrant-соединение не требуются.

## Проверка и демонстрация

Ruff выполняет lint и форматирование `src`, тестов и примеров. Mypy проверяет
`src` с обязательными аннотациями функций и типами NumPy-массивов;
`pandas-stubs` входит в dev-зависимости. Конфигурация обоих инструментов — в
`pyproject.toml`. Метаданные отчётов остаются словарями с динамической структурой.

```bash
python -m pip install -e '.[dev,plots]'
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest -q
python examples/synthetic_audit.py --output reports/demo
python -m build
```

Для применения форматирования используйте `python -m ruff format .`.
Пакет содержит `py.typed`, поэтому его аннотации доступны пользователям mypy.

Демонстрация сравнивает разделённые синтетические классы и те же признаки со
случайно переставленными метками. Для вывода о вашей SSL-модели нужен запуск
на её эмбеддингах.

Основа требований — последний ответ в
[исходном чате](https://chatgpt.com/share/6abf4204-e048-83ed-8558-af4ccb34e2f0).
Технические определения:
[silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html),
[групповой CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[блочные расстояния](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html).
