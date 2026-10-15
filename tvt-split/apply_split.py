#!/usr/bin/env python3
"""
Система фізичного копіювання та структурування датасету (Dataset Split Executor)
Зчитує конфігураційний файл JSON та розподіляє епізоди по папках train/val/test.
"""

import os
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm  # Імпорт бібліотеки для прогрес-бару

# Налаштування логування
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("DatasetExecutor")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()


class DatasetSplitApplier:
    def __init__(self, config_path: Path, source_dir: Path, target_dir: Path, dry_run: bool = False):
        self.config_path = Path(config_path).resolve()
        self.source_dir = Path(source_dir).resolve() if source_dir else None
        self.target_dir = Path(target_dir).resolve()
        self.dry_run = dry_run

    def load_config(self) -> Dict[str, Any]:
        """
        Зчитує JSON конфігурацію розподілу.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Конфігураційний файл не знайдено за вказаним шляхом: {self.config_path}")
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    def execute(self):
        """
        Виконує фізичне копіювання файлів згідно з конфігурацією з відображенням прогрес-бару.
        """
        # 1. Завантаження конфігурації
        try:
            config = self.load_config()
        except Exception as e:
            logger.error(f"Помилка завантаження конфігурації: {e}")
            return

        # 2. Визначення джерела даних
        src_root = self.source_dir
        if not src_root:
            metadata_src = config.get("metadata", {}).get("dataset_root_dir")
            if metadata_src:
                src_root = Path(metadata_src).resolve()
            else:
                logger.error("Не вказано джерело даних. Будь ласка, задайте --source_dir.")
                return

        logger.info(f"Джерело даних (Source): {src_root}")
        logger.info(f"Цільова папка (Target): {self.target_dir}")
        if self.dry_run:
            logger.info("Режим DRY-RUN активований. Фізичні зміни не будуть внесені.")

        if not src_root.exists():
            logger.error(f"Папка джерела даних не існує: {src_root}")
            return

        # 3. Підрахунок загальної кількості епізодів для прогрес-бару
        splits_data = config.get("splits", {})
        total_episodes = sum(len(ep_list) for ep_list in splits_data.values())
        copied_count = 0

        # 4. Обробка вибірок із використанням прогрес-бару tqdm
        logger.info("Початок копіювання епізодів...")
        
        with tqdm(total=total_episodes, desc="Загальний прогрес", unit="еп") as pbar:
            for split_name, episodes in splits_data.items():
                # Оновлюємо опис прогрес-бару поточним сплітом (TRAIN/VAL/TEST)
                pbar.set_description(f"Копіювання {split_name.upper()}")
                
                for ep_key in episodes:
                    copied_count += 1
                    
                    # Визначення шляхів
                    src_ep_dir = src_root / ep_key
                    dst_ep_dir = self.target_dir / split_name / ep_key
                    
                    if not src_ep_dir.exists():
                        tqdm.write(f"[ПРОПУСК] Папку джерела не знайдено: {src_ep_dir}")
                        pbar.update(1)
                        continue
                        
                    try:
                        if self.dry_run:
                            tqdm.write(f"[DRY-RUN] Буде скопійовано: {ep_key} -> {split_name}")
                        else:
                            if dst_ep_dir.exists():
                                # Захист від повторного копіювання (можна розкоментувати рядок нижче за потреби)
                                # tqdm.write(f"[ІСНУЄ] Пропущено: {ep_key}")
                                pass
                            else:
                                dst_ep_dir.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copytree(src_ep_dir, dst_ep_dir)
                    except Exception as e:
                        tqdm.write(f"[ПОМИЛКА] Не вдалося скопіювати {ep_key}: {e}")
                    
                    pbar.update(1)

        logger.info("Процес розподілу файлів завершено.")


def main():
    parser = argparse.ArgumentParser(description="Фізичне копіювання епізодів на основі JSON конфігурації")
    parser.add_argument("--config", type=str, default="dataset_splits_config.json", help="Шлях до файлу конфігурації розподілу JSON")
    parser.add_argument("--source_dir", type=str, default=None, help="Шлях до оригінального датасету (якщо відрізняється від метаданих у JSON)")
    parser.add_argument("--target_dir", type=str, default="dataset_v13", help="Шлях до створення нового розподіленого датасету")
    parser.add_argument("--dry_run", action="store_true", help="Запустити у тестовому режимі без копіювання файлів")
    
    args = parser.parse_args()
    
    applier = DatasetSplitApplier(
        config_path=Path(args.config),
        source_dir=Path(args.source_dir) if args.source_dir else None,
        target_dir=Path(args.target_dir),
        dry_run=args.dry_run
    )
    
    applier.execute()

if __name__ == "__main__":
    main()