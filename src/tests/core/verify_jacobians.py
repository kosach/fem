# tests/core/verify_jacobians.py
import numpy as np
from numpy.testing import assert_allclose
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fem_app.core import mesh, derivatives


ELEMENT_ID_TO_TEST = 1 

BASE_MOCK_PATH = os.path.join(current_dir, "..", "mocks")
ETA_JACOBIAN_PATHS = {
    0: os.path.join(BASE_MOCK_PATH, "yakobians_0_0.csv"),
    1: os.path.join(BASE_MOCK_PATH, "yakobians_0_1.csv"),
    2: os.path.join(BASE_MOCK_PATH, "yakobians_0_2.csv"),
}

TOL = 1e-9

def load_jacobian_from_csv(file_path):
    """Зчитує матрицю 3x3 з CSV файлу з розділювачем ';'."""
    try:
        matrix = np.loadtxt(file_path, delimiter=';', dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"Матриця у файлі {file_path} має неправильну розмірність {matrix.shape}, очікується (3,3)")
        return matrix
    except Exception as e:
        print(f"Помилка завантаження CSV файлу {file_path}: {e}")
        raise

def get_parent_element_data():
    """
    Створює дані для стандартного батьківського елемента (куб 2x2x2, центрований в (0,0,0)).
    Координати вузлів AKT збігаються з локальними координатами вузлів.
    """
    # Локальні координати вузлів з derivatives.py (які використовуються в compute_DFIABG)
    # Це забезпечить, що X=alpha, Y=beta, Z=gamma
    coords_local_parent = np.array([
        [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1],  # 1-4 (скориговані згідно вашого derivatives.py)
        [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1],    # 5-8
        [0, 1, -1], [1, 0, -1], [0, -1, -1], [-1, 0, -1],  # 9-12
        [-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0],    # 13-16
        [0, 1, 1], [1, 0, 1], [0, -1, 1], [-1, 0, 1]     # 17-20
    ], dtype=np.float64)

    AKT_parent = coords_local_parent.T # (3, 20)
    
    # Для одного елемента NT просто нумерує вузли від 1 до 20
    NT_parent = np.arange(1, 21, dtype=np.int32).reshape((20, 1))
    nqp_parent = AKT_parent.shape[1]
    nel_parent = NT_parent.shape[1]
    
    return {"AKT": AKT_parent, "NT": NT_parent, "nqp": nqp_parent, "nel": nel_parent}

def get_dfiabg_data():
    """Обчислює DFIABG."""
    a, b, g = derivatives.alpha_flat_g, derivatives.beta_flat_g, derivatives.gamma_flat_g
    return derivatives.compute_DFIABG(a, b, g)

def verify_jacobian_for_gp(element_data, dfiabg_values, gauss_point_index, expected_jacobian_path):
    """Перевіряє Якобіан для заданої точки Гаусса."""
    print(f"--- Перевірка Якобіана для точки Гаусса GP {gauss_point_index} ---")
    AKT = element_data["AKT"]
    NT = element_data["NT"]

    # Розраховуємо Якобіан
    J_calculated = derivatives.compute_jacobian(
        AKT, NT, ELEMENT_ID_TO_TEST, dfiabg_values, gauss_point_index
    )

    # Завантажуємо еталонний Якобіан
    if not os.path.exists(expected_jacobian_path):
        print(f"ПОПЕРЕДЖЕННЯ: Еталонний файл Якобіана не знайдено: {expected_jacobian_path}")
        print(f"Розрахований Якобіан для GP {gauss_point_index}:\n{J_calculated}\n")
        return False

    J_expected = load_jacobian_from_csv(expected_jacobian_path)

    print(f"Розрахований Якобіан для GP {gauss_point_index} (Елемент {ELEMENT_ID_TO_TEST}):")
    print(J_calculated)
    print(f"Еталонний Якобіан для GP {gauss_point_index} (з {os.path.basename(expected_jacobian_path)}):")
    print(J_expected)

    # Порівнюємо
    try:
        assert_allclose(J_calculated, J_expected, atol=TOL)
        print(f"РЕЗУЛЬТАТ: Якобіан для GP {gauss_point_index} СПІВПАДАЄ з еталоном.\n")
        return True
    except AssertionError as e:
        print(f"ПОМИЛКА: Якобіан для GP {gauss_point_index} НЕ співпадає з еталоном!")
        print(e)
        # Додатково виведемо різницю
        diff_matrix = J_calculated - J_expected
        print(f"Різниця (Розрахований - Еталонний):\n{diff_matrix}\n")
        return False

if __name__ == "__main__":
    element_data_to_use = get_parent_element_data()
    

    print("Завантаження DFIABG...")
    dfiabg_values = get_dfiabg_data()
    print("DFIABG завантажено.\n")

    all_passed = True
    for gp_idx, file_path in ETA_JACOBIAN_PATHS.items():
        if not verify_jacobian_for_gp(element_data_to_use, dfiabg_values, gp_idx, file_path):
            all_passed = False
    
    if all_passed:
        print("--- УСПІХ: Всі перевірені Якобіани співпадають з еталонами! ---")
    else:
        print("--- ПОМИЛКА: Є розбіжності в Якобіанах. Дивіться деталі вище. ---")