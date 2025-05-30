# fem_app/ui/array_viewer.py
import tkinter as tk
from tkinter import ttk
import numpy as np

def create_array_viewer_window(parent, matrix, title="Перегляд масиву", column_headers=None): # Додано column_headers
    """
    Створює Toplevel вікно з ttk.Treeview для показу numpy масиву,
    додаючи нумерацію рядків та можливість кастомних заголовків.
    parent - батьківське вікно Tkinter.
    matrix - numpy масив (або список списків/кортежів) для відображення.
    title - заголовок вікна.
    column_headers - список рядків для заголовків стовпців даних.
    """
    top = tk.Toplevel(parent)
    top.title(title)
    top.geometry("700x500")

    # Перетворюємо на NumPy array, якщо це список списків (для зручності)
    if isinstance(matrix, list):
        try:
            matrix = np.array(matrix, dtype=object) # dtype=object для змішаних типів
        except Exception as e:
            ttk.Label(top, text=f"Помилка перетворення даних: {e}").pack(padx=20, pady=20)
            return


    if matrix is None or matrix.size == 0:
        ttk.Label(top, text="Масив порожній або не існує.").pack(padx=20, pady=20)
        return

    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis] # Перетворюємо 1D у 2D (N x 1)

    rows, cols = matrix.shape

    max_cols_display = 50 # Обмеження для уникнення занадто широких таблиць
    max_rows_display = 1000 # Обмеження для уникнення зависань
    
    display_cols = min(cols, max_cols_display)
    display_rows = min(rows, max_rows_display)

    tree_frame = ttk.Frame(top)
    tree_frame.pack(expand=True, fill='both', padx=5, pady=5)

    # Створюємо ідентифікатори для стовпців Treeview (крім #0)
    tree_column_ids = [f"col{i}" for i in range(display_cols)]
    tree = ttk.Treeview(tree_frame, columns=tree_column_ids, show='headings', height=15) # show='headings' прибирає перший порожній стовпець

    # --- Налаштування стовпців ---
    # Стовпець для номерів рядків (якщо потрібен окремий, а не через tree.insert з text)
    # tree.heading("#0", text="#") # Якщо show='tree headings', то #0 не використовується так
    # tree.column("#0", width=50, stretch=tk.NO, anchor='center')

    # Налаштування стовпців даних
    if column_headers and len(column_headers) == display_cols:
        for idx, col_id_str in enumerate(tree_column_ids):
            tree.heading(col_id_str, text=column_headers[idx])
            tree.column(col_id_str, width=120, anchor='w', stretch=tk.YES) # Збільшено ширину, вирівнювання по лівому краю
    else: # Стандартна поведінка, якщо заголовки не передані
        for idx, col_id_str in enumerate(tree_column_ids):
            tree.heading(col_id_str, text=str(idx + 1)) # Нумерація стовпців з 1
            tree.column(col_id_str, width=100, anchor='e', stretch=tk.YES)

    # Скролбари
    y_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=y_scrollbar.set)
    x_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
    tree.configure(xscrollcommand=x_scrollbar.set)

    # Додавання даних
    for row_idx in range(display_rows):
        values_to_insert = []
        for col_idx in range(display_cols):
            value = matrix[row_idx, col_idx]
            if isinstance(value, (float, np.floating)):
                 values_to_insert.append(f"{value:.6e}") # Використовуємо експоненційний формат для кращої читабельності
            else:
                 values_to_insert.append(str(value))
        # Перший аргумент для tree.insert - parent ('' для кореневих елементів)
        # 'end' - вставляти в кінець
        # text - значення для першого стовпця (якщо show='tree headings', то цей стовпець "#0" може не відображатися явно, якщо не налаштований)
        # values - кортеж/список значень для стовпців, визначених у 'columns'
        tree.insert('', 'end', values=values_to_insert, iid=str(row_idx)) # iid для унікальності рядків

    # Розміщення віджетів
    y_scrollbar.pack(side='right', fill='y')
    x_scrollbar.pack(side='bottom', fill='x')
    tree.pack(expand=True, fill='both')

    # Інформація про обмеження показу
    info_text_parts = []
    if rows > max_rows_display: info_text_parts.append(f"Показано перші {max_rows_display} з {rows} рядків.")
    if cols > max_cols_display: info_text_parts.append(f"Показано перші {max_cols_display} з {cols} стовпців.")
    if info_text_parts:
        ttk.Label(top, text=" ".join(info_text_parts), foreground="blue").pack(pady=(5,0), anchor='w', padx=5)