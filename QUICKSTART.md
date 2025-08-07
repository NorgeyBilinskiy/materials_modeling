# Быстрый старт проекта

## 🚀 Минимальные требования

- Python 3.10+
- 4GB RAM
- 2GB свободного места на диске

## ⚡ Экспресс-запуск

### 1. Клонирование и установка
```bash
git clone <repository-url>
cd materials_modeling
pip install -r requirements.txt
```

### 2. Тестирование установки
```bash
python test_project.py
```

### 3. Быстрый запуск
```bash
# Настройка проекта
python run.py setup

# Обучение CGCNN (самая быстрая модель)
python run.py train --method cgcnn --epochs 50

# Предсказание
python run.py predict --method cgcnn
```

## 📊 Ожидаемые результаты

После выполнения вы должны увидеть:
```
CGCNN predicted formation energy: -3.45 eV/atom
Reference value: -3.6 eV/atom
Absolute error: 0.15 eV/atom
```

## 🔧 Альтернативные способы запуска

### Через Docker
```bash
docker build -t nacl-prediction .
docker run -it --rm nacl-prediction python run.py setup
docker run -it --rm nacl-prediction python run.py train --method cgcnn
```

### Через Jupyter
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🐛 Устранение проблем

### Ошибка импорта PyTorch Geometric
```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Недостаточно памяти
```bash
python run.py train --method cgcnn --batch-size 16 --epochs 30
```

### Проблемы с CUDA
```bash
export CUDA_VISIBLE_DEVICES=""
python run.py train --method cgcnn
```

## 📈 Следующие шаги

1. **Изучите ноутбуки** в папке `notebooks/`
2. **Попробуйте другие модели**: MEGNet, SchNet, MPNN
3. **Сравните результаты**: `python run.py compare`
4. **Изучите документацию**: `README.md`, `SETUP.md`

## 🎯 Что вы получите

- ✅ Рабочий пайплайн предсказания энергии формирования NaCl
- ✅ 4 различные графовые нейронные сети
- ✅ Полную документацию и примеры
- ✅ Готовность к расширению на другие материалы

## 📞 Поддержка

При возникновении проблем:
1. Проверьте `test_project.py`
2. Изучите логи в папке `logs/`
3. Обратитесь к `SETUP.md` для подробных инструкций

