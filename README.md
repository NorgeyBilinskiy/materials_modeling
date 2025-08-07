# NaCl Formation Energy Prediction with Graph Neural Networks

## Цель проекта

Данный проект демонстрирует применение различных графовых нейронных сетей для предсказания энергии формирования кристаллической структуры NaCl (поваренная соль). Проект сравнивает эффективность четырех современных подходов: CGCNN, MEGNet, SchNet и MPNN.

## Теоретическое обоснование

### Энергия формирования NaCl
Энергия формирования NaCl составляет примерно **-3.6 эВ/атом** и является важной термодинамической характеристикой, определяющей стабильность кристаллической структуры.

### Используемые модели

1. **CGCNN (Crystal Graph Convolutional Neural Network)**
   - Специально разработан для кристаллических структур
   - Использует графовые свертки для обработки связей между атомами
   - Учитывает периодические граничные условия

2. **MEGNet (MatErials Graph Network)**
   - Универсальная архитектура для материалов
   - Сочетает графовые нейронные сети с глобальными состояниями
   - Эффективна для предсказания различных свойств материалов

3. **SchNet (Schrödinger Network)**
   - Основана на квантово-механических принципах
   - Использует фильтры для моделирования взаимодействий
   - Хорошо работает с молекулярными и кристаллическими системами

4. **MPNN (Message Passing Neural Network)**
   - Общий фреймворк для графовых нейронных сетей
   - Передает сообщения между узлами графа
   - Гибкая архитектура для различных типов графов

## Структура проекта

```
project_root/
├── Dockerfile                 # Контейнеризация проекта
├── docker-compose.yml         # Оркестрация сервисов
├── run.py                     # Главный CLI-скрипт
├── requirements.txt           # Зависимости Python
├── data_loader/              # Загрузка и предобработка данных
├── models/                   # Реализации моделей
├── notebooks/                # Jupyter ноутбуки для анализа
└── tests/                    # Unit-тесты
```

## Быстрый старт

### Через Docker (рекомендуется)

```bash
# Клонирование репозитория
git clone <repository-url>
cd materials_modeling

# Запуск через Docker Compose
docker-compose up --build

# Или запуск конкретной модели
python run.py --method cgcnn
```

### Локальная установка

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск обучения
python run.py --method cgcnn
```

## Использование

### CLI-интерфейс

```bash
# Обучение модели
python run.py --method cgcnn --train

# Предсказание
python run.py --method cgcnn --predict

# Сравнение всех моделей
python run.py --compare
```

### Доступные методы
- `cgcnn` - Crystal Graph Convolutional Neural Network
- `megnet` - MatErials Graph Network  
- `schnet` - Schrödinger Network
- `mpnn` - Message Passing Neural Network

## Базы данных

Проект использует следующие источники данных:
- **Materials Project** (https://materialsproject.org/) - основная база данных кристаллических структур
- **OQMD** (Open Quantum Materials Database) - дополнительный источник данных

## Результаты

Ожидаемые результаты для NaCl:
- **Референсное значение**: ~-3.6 эВ/атом
- **CGCNN**: Ожидается близкое к референсному значение
- **MEGNet**: Стабильные предсказания с хорошей точностью
- **SchNet**: Физически обоснованные результаты
- **MPNN**: Базовые предсказания для сравнения

## Требования

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- DGL (Deep Graph Library)
- Jupyter Notebook
- Docker (опционально)

## Лицензия

MIT License

## Авторы

Проект создан для демонстрации применения графовых нейронных сетей в материаловедении.
