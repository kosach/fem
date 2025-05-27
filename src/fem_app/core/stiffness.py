import numpy as np
from numba import njit

# --- Ваги Гаусса для 3x3x3 --- <--- ЦЕЙ БЛОК МАЄ БУТИ
w1D = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
gauss_weights_3D = np.zeros(27) # Визначення змінної
idx = 0
for wk in w1D:
    for wj in w1D:
        for wi in w1D:
            gauss_weights_3D[idx] = wi * wj * wk
            idx += 1

def calculate_lambda_mu(E, nu):
    """Обчислює параметри Ламе."""
    # Додаємо перевірку на nu != 0.5 (нестисливий матеріал) та nu != -1
    if abs(1 - 2 * nu) < 1e-9 or abs(1 + nu) < 1e-9:
         # Обробка граничних випадків - може потребувати іншої моделі або регуляризації
         print(f"Warning: Poisson's ratio nu={nu} is close to limit values.")
         # Повертаємо NaN або великі числа, або виняток
         return np.nan, np.nan # Або розрахувати інакше
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_ = E / (2 * (1 + nu))
    return lambda_, mu_


@njit 
def compute_element_stiffness_MGE(DFIXYZ_elem, DJ_dets, lambda_, mu_, gauss_weights):
    """
    Обчислює ПОВНУ матрицю жорсткості MGE(60, 60) БЕЗ використання симетрії
    для подальшої перевірки.
    """
    MGE = np.zeros((60, 60))
    if np.isnan(lambda_) or np.isnan(mu_):
        return MGE # Повертаємо нулі, якщо матеріал некоректний

    # Цикл по точках Гаусса
    for k in range(27): # Gauss point index (0..26)
        detJ = DJ_dets[k]
        weight = gauss_weights[k]
        if abs(detJ) < 1e-9 or np.isnan(detJ) or np.isnan(DFIXYZ_elem[k, :, :]).any():
            continue # Пропускаємо точку, якщо є проблеми

        factor = detJ * weight

        for i_node in range(20): # Локальний номер вузла i (0..19)
            dNi_dx = DFIXYZ_elem[k, i_node, 0]
            dNi_dy = DFIXYZ_elem[k, i_node, 1]
            dNi_dz = DFIXYZ_elem[k, i_node, 2]

            for j_node in range(20): # Локальний номер вузла j (0..19) - ПОВНИЙ ЦИКЛ
                dNj_dx = DFIXYZ_elem[k, j_node, 0]
                dNj_dy = DFIXYZ_elem[k, j_node, 1]
                dNj_dz = DFIXYZ_elem[k, j_node, 2]

                # Обчислюємо компоненти підматриці 3x3 k_ij
                c1 = lambda_ + 2 * mu_
                c2 = lambda_
                c3 = mu_

                k_xx = (c1 * dNi_dx * dNj_dx + c3 * dNi_dy * dNj_dy + c3 * dNi_dz * dNj_dz) * factor
                k_yy = (c3 * dNi_dx * dNj_dx + c1 * dNi_dy * dNj_dy + c3 * dNi_dz * dNj_dz) * factor
                k_zz = (c3 * dNi_dx * dNj_dx + c3 * dNi_dy * dNj_dy + c1 * dNi_dz * dNj_dz) * factor

                k_xy = (c2 * dNi_dx * dNj_dy + c3 * dNi_dy * dNj_dx) * factor
                k_yx = (c3 * dNi_dx * dNj_dy + c2 * dNi_dy * dNj_dx) * factor

                k_xz = (c2 * dNi_dx * dNj_dz + c3 * dNi_dz * dNj_dx) * factor
                k_zx = (c3 * dNi_dx * dNj_dz + c2 * dNi_dz * dNj_dx) * factor

                k_yz = (c2 * dNi_dy * dNj_dz + c3 * dNi_dz * dNj_dy) * factor
                k_zy = (c3 * dNi_dy * dNj_dz + c2 * dNi_dz * dNj_dy) * factor

                # Визначаємо індекси в матриці MGE(60,60)
                # Структура: [x1..x20, y1..y20, z1..z20]
                idx_xi = i_node;       idx_yi = i_node + 20; idx_zi = i_node + 40
                idx_xj = j_node;       idx_yj = j_node + 20; idx_zj = j_node + 40

                # Додаємо внесок від точки Гаусса k до відповідних елементів MGE
                # Кожен елемент обчислюється незалежно
                MGE[idx_xi, idx_xj] += k_xx
                MGE[idx_xi, idx_yj] += k_xy
                MGE[idx_xi, idx_zj] += k_xz

                MGE[idx_yi, idx_xj] += k_yx
                MGE[idx_yi, idx_yj] += k_yy
                MGE[idx_yi, idx_zj] += k_yz

                MGE[idx_zi, idx_xj] += k_zx
                MGE[idx_zi, idx_yj] += k_zy
                MGE[idx_zi, idx_zj] += k_zz

                # БЛОК 'if i_node != j_node:' ВИДАЛЕНО

    return MGE

# @njit # Поки що без Numba
def assemble_global_stiffness(MG, MGE, elem_nodes_global, nqp):
    """Додає MGE до глобальної MG (повної)."""
    print(" ЗАГЛУШКА: assemble_global_stiffness")
    full_size = 3 * nqp
    for r_loc in range(20):
        g_r = elem_nodes_global[r_loc]
        if g_r < 0: continue
        for c_loc in range(20):
            g_c = elem_nodes_global[c_loc]
            if g_c < 0: continue
            for dof_r in range(3):
                for dof_c in range(3):
                    mg_r_idx = 3 * g_r + dof_r
                    mg_c_idx = 3 * g_c + dof_c
                    # Індексація MGE: [x1..x20, y1..y20, z1..z20]
                    mge_r_idx = r_loc + 20 * dof_r
                    mge_c_idx = c_loc + 20 * dof_c
                    if 0 <= mg_r_idx < full_size and 0 <= mg_c_idx < full_size and \
                       0 <= mge_r_idx < 60 and 0 <= mge_c_idx < 60:
                         MG[mg_r_idx, mg_c_idx] += MGE[mge_r_idx, mge_c_idx]
    # Ця логіка виглядає більш правильною для збирання, АЛЕ ПОТРЕБУЄ КОРЕКТНОЇ MGE

def calculate_half_bandwidth(NT, nqp):
     """Обчислює півширину стрічки ng за формулою (22)."""
     if NT is None or NT.shape[1] == 0 or nqp == 0: return 3 # Мінімальна ширина
     max_diff = 0
     nel = NT.shape[1]
     for j in range(nel):
          nodes = NT[:, j]
          valid_nodes = nodes[nodes > 0] # Враховуємо тільки валідні (>0) номери
          if valid_nodes.size > 0:
               min_node = np.min(valid_nodes)
               max_node = np.max(valid_nodes)
               max_diff = max(max_diff, max_node - min_node)
     # ng = 3 * (різниця_макс_мін_ГЛОБАЛЬНИХ_DOF + 1)
     # Максимальна різниця глобальних номерів DOF = 3 * max_node - (3 * min_node + 0) = 3 * (max_node - min_node)
     # Або треба враховувати конкретні DOF: max(3*Nmax+2) - min(3*Nmin+0) + 1 ???
     # Формула (22) в методичці: ng = 3 * (max(Nmax_e) - min(Nmin_e) + 1)
     # де Nmax_e, Nmin_e - макс/мін ГЛОБАЛЬНІ номери ВУЗЛІВ елемента e
     # Отже, розрахунок max_diff вище правильний.
     ng = 3 * (max_diff + 1)
     # Обмеження: ng не може бути більше за повний розмір системи
     ng = min(ng, 3 * nqp)
     return ng if ng > 0 else 3