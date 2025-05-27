# Допоміжні функції (наприклад, парсинг рядків ГУ)
import numpy as np

def parse_zu_string(zu_str):
    """Парсить рядок ZU (номери через кому) в numpy масив."""
    if not zu_str:
        return np.array([], dtype=int)
    try:
        nodes = np.array([int(n.strip()) for n in zu_str.split(',') if n.strip()], dtype=int)
        if np.any(nodes <= 0):
             raise ValueError("Номери вузлів мають бути позитивними.")
        return nodes
    except ValueError:
        raise ValueError("Неправильний формат рядка ZU. Використовуйте цілі числа через кому.")

# Можна додати parse_zp_string аналогічно