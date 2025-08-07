# Инструкции по настройке и запуску проекта

## Быстрый старт

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd materials_modeling
```

### 2. Установка зависимостей

#### Локальная установка
```bash
# Создание виртуального окружения
python -m venv venv

# Активация виртуального окружения
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

#### Через Docker
```bash
# Сборка образа
docker build -t nacl-prediction .

# Запуск контейнера
docker run -it --rm -v $(pwd):/app nacl-prediction
```

### 3. Настройка проекта
```bash
# Создание необходимых директорий
mkdir -p data models logs results

# Инициализация проекта
python run.py setup
```

## Использование

### Основные команды

#### Обучение моделей
```bash
# Обучение CGCNN
python run.py train --method cgcnn --epochs 100

# Обучение MEGNet
python run.py train --method megnet --epochs 100

# Обучение SchNet
python run.py train --method schnet --epochs 100

# Обучение MPNN
python run.py train --method mpnn --epochs 100
```

#### Предсказания
```bash
# Предсказание с помощью CGCNN
python run.py predict --method cgcnn

# Предсказание с помощью MEGNet
python run.py predict --method megnet

# Предсказание с помощью SchNet
python run.py predict --method schnet

# Предсказание с помощью MPNN
python run.py predict --method mpnn
```

#### Сравнение моделей
```bash
# Сравнение всех обученных моделей
python run.py compare
```

### Jupyter Notebooks

#### Запуск Jupyter
```bash
# Запуск Jupyter Notebook
jupyter notebook notebooks/

# Или через Docker
docker-compose up
```

#### Последовательность выполнения ноутбуков
1. `01_data_exploration.ipynb` - Исследование данных NaCl
2. `02_train_cgcnn.ipynb` - Обучение CGCNN
3. `03_train_megnet.ipynb` - Обучение MEGNet
4. `04_train_schnet.ipynb` - Обучение SchNet
5. `05_compare_models.ipynb` - Сравнение всех моделей

## Структура проекта

```
materials_modeling/
├── data_loader/           # Загрузка и предобработка данных
│   ├── download.py       # Загрузка данных из баз
│   └── preprocess.py     # Преобразование в графы
├── models/               # Реализации моделей
│   ├── cgcnn/           # Crystal Graph CNN
│   ├── megnet/          # MatErials Graph Network
│   ├── schnet/          # Schrödinger Network
│   └── mpnn/            # Message Passing NN
├── notebooks/            # Jupyter ноутбуки
├── tests/               # Unit-тесты
├── data/                # Данные (создается автоматически)
├── models/              # Сохраненные модели
├── logs/                # Логи обучения
├── results/             # Результаты сравнения
├── run.py               # Главный CLI-скрипт
├── requirements.txt     # Зависимости
├── Dockerfile           # Docker-образ
├── docker-compose.yml   # Docker Compose
└── README.md            # Документация
```

## Конфигурация

### Параметры обучения
- `--epochs`: Количество эпох обучения (по умолчанию: 100)
- `--batch-size`: Размер батча (по умолчанию: 32)
- `--lr`: Скорость обучения (по умолчанию: 0.001)
- `--data-path`: Путь к данным (по умолчанию: "data/")

### Параметры моделей
- `hidden_channels`: Количество скрытых каналов (по умолчанию: 64)
- `num_layers`: Количество слоев (по умолчанию: 3)
- `dropout`: Коэффициент dropout (по умолчанию: 0.2)

## Тестирование

### Запуск тестов
```bash
# Запуск всех тестов
python -m pytest tests/ -v

# Запуск конкретного теста
python -m pytest tests/test_pipeline.py::TestDataPipeline -v

# Запуск с покрытием
python -m pytest tests/ --cov=. --cov-report=html
```

### Проверка качества кода
```bash
# Проверка стиля кода
black .
flake8 .
isort .

# Проверка типов
mypy .
```

## Мониторинг и логирование

### Логи обучения
Логи сохраняются в директории `logs/`:
- `run.log` - Основные логи выполнения
- `models/*/training_history.json` - История обучения моделей

### Метрики
- **MSE**: Среднеквадратичная ошибка
- **MAE**: Средняя абсолютная ошибка
- **RMSE**: Корень из среднеквадратичной ошибки
- **R²**: Коэффициент детерминации

## Устранение неполадок

### Частые проблемы

#### 1. Ошибки установки PyTorch Geometric
```bash
# Установка PyTorch Geometric
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

#### 2. Проблемы с CUDA
```bash
# Проверка доступности CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Принудительное использование CPU
export CUDA_VISIBLE_DEVICES=""
```

#### 3. Недостаточно памяти
```bash
# Уменьшение размера батча
python run.py train --method cgcnn --batch-size 16
```

#### 4. Ошибки загрузки данных
```bash
# Пересоздание данных
rm -rf data/
python run.py setup
```

### Получение помощи
```bash
# Справка по командам
python run.py --help
python run.py train --help
python run.py predict --help
```

## Производительность

### Оптимизация для GPU
- Увеличьте `batch_size` для лучшего использования GPU
- Используйте `torch.compile()` для ускорения (PyTorch 2.0+)
- Включите автоматическую смешанную точность

### Оптимизация для CPU
- Уменьшите `batch_size`
- Используйте меньшее количество слоев
- Отключите CUDA: `export CUDA_VISIBLE_DEVICES=""`

## Расширение проекта

### Добавление новых моделей
1. Создайте папку в `models/`
2. Реализуйте `model.py`, `train.py`, `predict.py`
3. Добавьте модель в `run.py`
4. Создайте тесты в `tests/`

### Добавление новых данных
1. Расширьте `data_loader/download.py`
2. Обновите `data_loader/preprocess.py`
3. Добавьте валидацию в `tests/`

### Добавление новых метрик
1. Создайте функции в соответствующих модулях
2. Добавьте визуализацию в ноутбуки
3. Обновите документацию

## Лицензия

MIT License - см. файл LICENSE для подробностей.

