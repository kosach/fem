import tkinter as tk
import sys
import os

# Додаємо src до шляху, щоб знайти fem_app, якщо запускаємо main.py напряму
# (Poetry зазвичай робить це автоматично при запуску через `poetry run`)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Перевірка імпортів основних залежностей
try:
    import numpy
    import matplotlib
    import numba
except ImportError as e:
     # Проста перевірка без tkinter, якщо сам tkinter ще не імпортовано
     print(f"ПОМИЛКА: Необхідну бібліотеку не знайдено: {e.name}.\nБудь ласка, встановіть залежності: poetry install", file=sys.stderr)
     sys.exit(1)

# Імпортуємо основний клас програми з нашого пакету в src/
try:
    from fem_app.ui.main_window import FEMApp
except ModuleNotFoundError as e:
     print(f"ПОМИЛКА: Не вдалося імпортувати FEMApp з src/fem_app.\nПеревірте структуру проекту та pyproject.toml.\n{e}", file=sys.stderr)
     sys.exit(1)


if __name__ == "__main__":
    # Цей блок виконується, коли файл запускається як скрипт
    print("Запуск програми МСЕ Аналізу...")
    app = FEMApp()
    app.mainloop()