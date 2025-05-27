import tkinter as tk
from tkinter import ttk
import numpy as np

def create_array_viewer_window(parent, matrix, title="Перегляд масиву"):
    """
    Створює Toplevel вікно з ttk.Treeview для показу numpy масиву,
    додаючи нумерацію рядків.
    parent - батьківське вікно Tkinter.
    matrix - numpy масив для відображення.
    title - заголовок вікна.
    """
    top = tk.Toplevel(parent)
    top.title(title)
    top.geometry("700x500") # Можна збільшити за потреби

    if matrix is None or matrix.size == 0:
        ttk.Label(top, text="Масив порожній або не існує.").pack(padx=20, pady=20)
        return

    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis] # Перетворюємо 1D у 2D (N x 1)

    rows, cols = matrix.shape

    max_cols_display = 50
    max_rows_display = 1000 # Обмеження для уникнення зависань
    display_cols = min(cols, max_cols_display)
    display_rows = min(rows, max_rows_display)

    tree_frame = ttk.Frame(top)
    tree_frame.pack(expand=True, fill='both', padx=5, pady=5)

    # Визначаємо колонки: '#' + колонки даних
    column_ids = ["#0"] + [str(i) for i in range(display_cols)]
    tree = ttk.Treeview(tree_frame, columns=column_ids[1:], show='tree headings', height=15)

    # --- Зміна 1: Налаштування колонки #0 для номерів рядків ---
    tree.heading("#0", text="#") # Заголовок для колонки номерів рядків
    tree.column("#0", width=50, stretch=tk.NO, anchor='center') # Фіксована ширина, центрування

    # Налаштування колонок даних (нумерація з 1)
    for col_idx_str in column_ids[1:]: # Пропускаємо '#0'
        col_num = int(col_idx_str)
        tree.heading(col_idx_str, text=str(col_num + 1))
        tree.column(col_idx_str, width=80, anchor='e') # Вирівнювання праворуч

    # Скролбари
    y_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=y_scrollbar.set)
    x_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
    tree.configure(xscrollcommand=x_scrollbar.set)

    # Додавання даних
    for row in range(display_rows):
        # Форматуємо значення для колонок даних
        values = [
            f"{matrix[row, col]:.6g}" if isinstance(matrix[row, col], (float, np.floating)) else str(matrix[row, col])
            for col in range(display_cols)
        ]
        # --- Зміна 2: Додаємо параметр 'text' для колонки #0 ---
        tree.insert('', 'end', text=str(row + 1), values=values) # text = номер рядка (з 1)

    # Розміщення віджетів
    y_scrollbar.pack(side='right', fill='y')
    x_scrollbar.pack(side='bottom', fill='x')
    tree.pack(expand=True, fill='both')

    # Інформація про обмеження показу
    info_text = ""
    if rows > max_rows_display: info_text += f"Показано перші {max_rows_display} з {rows} рядків. "
    if cols > max_cols_display: info_text += f"Показано перші {max_cols_display} з {cols} стовпців."
    if info_text: ttk.Label(top, text=info_text, foreground="blue").pack(pady=(5,0), anchor='w', padx=5)