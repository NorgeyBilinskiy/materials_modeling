#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работоспособности проекта.
Запускает основные компоненты и проверяет их функциональность.
"""

import os
import sys
import tempfile
import shutil


def test_imports():
    """Тестирование импортов основных модулей."""
    print("Тестирование импортов...")
    
    try:
        # Тест импорта основных модулей
        from src.data_loader import download_nacl_data, create_sample_nacl_data
        from src.data_loader import preprocess_data, create_graph_features
        from src.models.cgcnn import create_cgcnn_model
        from src.models import create_megnet_model
        from src.models.schnet.model import create_schnet_model
        from src.models.mpnn.model import create_mpnn_model
        
        print("✓ Все импорты успешны")
        return True
    except ImportError as e:
        print(f"✗ Ошибка импорта: {e}")
        return False

def test_data_creation():
    """Тестирование создания данных."""
    print("\nТестирование создания данных...")
    
    try:
        # Создание временной директории
        test_dir = tempfile.mkdtemp()
        
        # Тест создания данных
        from src.data_loader import create_sample_nacl_data, create_training_dataset
        create_sample_nacl_data(test_dir)
        create_training_dataset(test_dir)
        
        # Проверка создания файлов
        required_files = [
            "nacl.cif",
            "nacl_info.json", 
            "training_data.json",
            "training_data.csv"
        ]
        
        for file in required_files:
            if os.path.exists(os.path.join(test_dir, file)):
                print(f"✓ Файл {file} создан")
            else:
                print(f"✗ Файл {file} не найден")
                return False
        
        # Очистка
        shutil.rmtree(test_dir)
        print("✓ Создание данных успешно")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка создания данных: {e}")
        return False

def test_model_creation():
    """Тестирование создания моделей."""
    print("\nТестирование создания моделей...")
    
    try:
        # Тест создания всех моделей
        models = {
            'CGCNN': create_cgcnn_model,
            'MEGNet': create_megnet_model,
            'SchNet': create_schnet_model,
            'MPNN': create_mpnn_model
        }
        
        for name, model_func in models.items():
            model = model_func()
            print(f"✓ Модель {name} создана успешно")
            
            # Проверка атрибутов модели
            if hasattr(model, 'forward'):
                print(f"  - Метод forward() доступен")
            else:
                print(f"  ✗ Метод forward() отсутствует")
                return False
        
        print("✓ Все модели созданы успешно")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка создания моделей: {e}")
        return False

def test_graph_features():
    """Тестирование создания графовых признаков."""
    print("\nТестирование создания графовых признаков...")
    
    try:
        from pymatgen.core import Structure, Lattice
        from src.data_loader import create_graph_features
        
        # Создание тестовой структуры
        lattice = Lattice.cubic(5.64)
        structure = Structure(
            lattice=lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        
        # Создание графовых признаков
        node_features, edge_index, edge_attr = create_graph_features(structure)
        
        # Проверка типов и размеров
        import torch
        assert isinstance(node_features, torch.Tensor), "node_features должен быть torch.Tensor"
        assert isinstance(edge_index, torch.Tensor), "edge_index должен быть torch.Tensor"
        assert isinstance(edge_attr, torch.Tensor), "edge_attr должен быть torch.Tensor"
        
        print(f"✓ Графовые признаки созданы:")
        print(f"  - node_features: {node_features.shape}")
        print(f"  - edge_index: {edge_index.shape}")
        print(f"  - edge_attr: {edge_attr.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка создания графовых признаков: {e}")
        return False

def test_cli_interface():
    """Тестирование CLI интерфейса."""
    print("\nТестирование CLI интерфейса...")
    
    try:
        # Тест импорта CLI
        from run import cli
        
        # Проверка доступности команд
        commands = ['train', 'predict', 'compare', 'setup', 'info']
        
        for cmd in commands:
            if hasattr(cli, cmd):
                print(f"✓ Команда {cmd} доступна")
            else:
                print(f"✗ Команда {cmd} отсутствует")
                return False
        
        print("✓ CLI интерфейс работает корректно")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка CLI интерфейса: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ПРОЕКТА NaCl FORMATION ENERGY PREDICTION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_creation,
        test_model_creation,
        test_graph_features,
        test_cli_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ: {passed}/{total} тестов пройдено")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Проект готов к использованию.")
        print("\nСледующие шаги:")
        print("1. python run.py setup")
        print("2. python run.py train --method cgcnn")
        print("3. python run.py predict --method cgcnn")
        print("4. python run.py compare")
    else:
        print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ. Проверьте установку зависимостей.")
        print("\nРекомендации:")
        print("1. Убедитесь, что все зависимости установлены: pip install -r requirements.txt")
        print("2. Проверьте версии PyTorch и PyTorch Geometric")
        print("3. Убедитесь, что вы находитесь в корневой директории проекта")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

