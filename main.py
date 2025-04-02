import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numba import njit
import sys # Додано для np.set_printoptions

# Налаштування виводу numpy для повних масивів
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

# -------------------------------------------------------------
# 1️⃣ Параметри сітки
n_x, n_y, n_z = 1, 1, 1  # Кількість елементів у кожному напрямку
a_x, a_y, a_z = 1.0, 1.0, 1.0  # Розміри тіла

# -------------------------------------------------------------
# 2️⃣ Генерація вузлів (без Numba через node_map)
# @njit # Прибираємо @njit через node_map
def is_corner_or_edge(ix, iy, iz):
    """Перевіряє, чи точка - це кут або середина ребра."""
    c_odd = (ix % 2) + (iy % 2) + (iz % 2)
    return c_odd <= 1  # максимум одна координата == 1 (не грань!)

# @njit # Прибираємо @njit через node_map
def generate_mesh(n_x, n_y, n_z, a_x, a_y, a_z):
    """Генерує координати вузлів у форматі (3, nqp)"""
    Nx, Ny, Nz = 2 * n_x + 1, 2 * n_y + 1, 2 * n_z + 1
    nodes = np.zeros((3, Nx * Ny * Nz), dtype=np.float64)
    node_map = {}
    iNode = 0

    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                # Використовуємо is_corner_or_edge без @njit
                if (ix % 2 + iy % 2 + iz % 2) <= 1:
                    x = ix * (a_x / (Nx - 1))
                    y = iy * (a_y / (Ny - 1))
                    z = iz * (a_z / (Nz - 1))
                    node_map[(ix, iy, iz)] = iNode
                    nodes[:, iNode] = np.array([x, y, z])
                    iNode += 1

    return nodes[:, :iNode], node_map

# -------------------------------------------------------------
# 3️⃣ Генерація таблиці `NT` (без Numba через node_map)
# @njit # Прибираємо @njit через node_map
def generate_connectivity(n_x, n_y, n_z, node_map):
    """Генерує таблицю NT для 20-вузлових елементів із правильним порядком вузлів"""
    local_offsets_20 = [
        (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),   # кути низ (1-4)
        (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),   # кути верх (5-8)
        (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0),   # ребра низ (9-12)
        (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1),   # вертикалі (13-16)
        (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2)    # ребра верх (17-20)
    ]

    nel = n_x * n_y * n_z
    NT = np.zeros((20, nel), dtype=np.int32)

    elem_id = 0
    for ez in range(n_z):
        for ey in range(n_y):
            for ex in range(n_x):
                for i_loc, (lx, ly, lz) in enumerate(local_offsets_20):
                    gx, gy, gz = 2 * ex + lx, 2 * ey + ly, 2 * ez + lz
                    NT[i_loc, elem_id] = node_map[(gx, gy, gz)] + 1 # +1 до Python-індексу
                elem_id += 1

    return NT

# Генеруємо сітку
AKT, node_map = generate_mesh(n_x, n_y, n_z, a_x, a_y, a_z)
nqp = AKT.shape[1]

# Генеруємо таблицю `NT`
NT = generate_connectivity(n_x, n_y, n_z, node_map)
nel = NT.shape[1]

# -------------------------------------------------------------
# 4️⃣ Візуалізація сітки (за бажанням)
# ... (код візуалізації залишається без змін, закоментований) ...

# -------------------------------------------------------------
# Обчислення координат точок Гаусса
gp = np.sqrt(0.6)
g_pts_1d = np.array([-gp, 0.0, gp])
alpha_g, beta_g, gamma_g = np.meshgrid(g_pts_1d, g_pts_1d, g_pts_1d, indexing='ij')
alpha_flat_g = alpha_g.flatten()
beta_flat_g = beta_g.flatten()
gamma_flat_g = gamma_g.flatten()
# -------------------------------------------------------------

@njit # Залишаємо Numba тут
def compute_DFIABG(alpha_flat, beta_flat, gamma_flat):
    """
    Обчислює масив DFIABG(27, 3, 20)
    """
    coords = np.array([
        [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1], # 1-4
        [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1], # 5-8
        [ 0,-1,-1],[ 1, 0,-1],[ 0, 1,-1],[-1, 0,-1], # 9-12
        [-1,-1, 0],[ 1,-1, 0],[ 1, 1, 0],[-1, 1, 0], # 13-16
        [ 0,-1, 1],[ 1, 0, 1],[ 0, 1, 1],[-1, 0, 1]  # 17-20
    ], dtype=np.float64)

    a_i = coords[:, 0]
    b_i = coords[:, 1]
    g_i = coords[:, 2]

    DFIABG = np.zeros((27, 3, 20))

    for j in range(27):
        a = alpha_flat[j]
        b = beta_flat[j]
        g = gamma_flat[j]

        for i in range(20):
            ai = a_i[i]
            bi = b_i[i]
            gi = g_i[i]

            if i < 8:
                term1 = (1 + a * ai)
                term2 = (1 + b * bi)
                term3 = (1 + g * gi)
                sum_terms = a * ai + b * bi + g * gi
                dphi_da = 0.125 * ai * term2 * term3 * (2 * a * ai + sum_terms - 1)
                dphi_db = 0.125 * bi * term1 * term3 * (2 * b * bi + sum_terms - 1)
                dphi_dg = 0.125 * gi * term1 * term2 * (2 * g * gi + sum_terms - 1)
            else:
                if abs(ai) < 1e-9:
                   term1 = (1 - a * a)
                   term2 = (1 + b * bi)
                   term3 = (1 + g * gi)
                   dphi_da = 0.25 * (-2 * a) * term2 * term3
                   dphi_db = 0.25 * term1 * bi * term3
                   dphi_dg = 0.25 * term1 * term2 * gi
                elif abs(bi) < 1e-9:
                   term1 = (1 + a * ai)
                   term2 = (1 - b * b)
                   term3 = (1 + g * gi)
                   dphi_da = 0.25 * ai * term2 * term3
                   dphi_db = 0.25 * term1 * (-2 * b) * term3
                   dphi_dg = 0.25 * term1 * term2 * gi
                else: # abs(gi) < 1e-9
                   term1 = (1 + a * ai)
                   term2 = (1 + b * bi)
                   term3 = (1 - g * g)
                   dphi_da = 0.25 * ai * term2 * term3
                   dphi_db = 0.25 * term1 * bi * term3
                   dphi_dg = 0.25 * term1 * term2 * (-2 * g)

            DFIABG[j, 0, i] = dphi_da
            DFIABG[j, 1, i] = dphi_db
            DFIABG[j, 2, i] = dphi_dg

    return DFIABG

@njit # Залишаємо Numba тут
def compute_jacobian(AKT, NT, elem_id, DFIABG, j_gauss):
    """
    Обчислює матрицю Якобі
    """
    J = np.zeros((3, 3))
    elem_nodes = NT[:, elem_id - 1] - 1

    for i in range(20):
        global_node_idx = elem_nodes[i]
        node_coords = AKT[:, global_node_idx]
        dphi_dlocal = DFIABG[j_gauss, :, i]

        for m in range(3):
            for n in range(3):
                J[m, n] += dphi_dlocal[n] * node_coords[m]
    return J

@njit # Залишаємо Numba тут
def compute_DFIXYZ_for_element(elem_id, AKT, NT, DFIABG):
    """
    Обчислює масив DFIXYZ для одного елемента.
    Розмірність: (Точка Гаусса, Номер вузла, Глобальна координата) -> (27, 20, 3)
    """
    DFIXYZ = np.zeros((27, 20, 3))

    for j in range(27):
        J = compute_jacobian(AKT, NT, elem_id, DFIABG, j)

        # Прибираємо try-except, несумісний з Numba для LinAlgError
        # Якщо J сингулярна, помилка виникне тут і буде оброблена поза Numba
        J_inv = np.linalg.inv(J)

        for i in range(20):
            dphi_local = DFIABG[j, :, i]
            dphi_global = np.dot(J_inv, dphi_local)
            DFIXYZ[j, i, :] = dphi_global

    return DFIXYZ


# 4️⃣ Візуалізація сітки
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(AKT[0, :], AKT[1, :], AKT[2, :], c='red', s=40)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Сітка 20-вузлових елементів")

# Підписуємо вузли
for i in range(nqp):
    x, y, z = AKT[:, i]
    ax.text(x, y, z, str(i+1), size=8, color='blue')

# Додаємо грані для всієї фігури
corners = np.array([
    [0, 0, 0], [a_x, 0, 0], [a_x, a_y, 0], [0, a_y, 0],  # Нижня основа (Z = 0)
    [0, 0, a_z], [a_x, 0, a_z], [a_x, a_y, a_z], [0, a_y, a_z]  # Верхня основа (Z = a_z)
])
faces = [
    [corners[0], corners[1], corners[2], corners[3]],  # Нижня основа
    [corners[4], corners[5], corners[6], corners[7]],  # Верхня основа
    [corners[0], corners[1], corners[5], corners[4]],  # Передня грань
    [corners[2], corners[3], corners[7], corners[6]],  # Задня грань
    [corners[1], corners[2], corners[6], corners[5]],  # Бічна грань
    [corners[0], corners[3], corners[7], corners[4]]   # Протилежна грань
]
ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', edgecolors='black', linewidths=1, alpha=0.2))

# Ракурс
ax.view_init(elev=15, azim=-75)

# -------------------------------------------------------------
# Основна частина скрипта
# -------------------------------------------------------------

print(f"--- Параметри сітки ---")
print(f"nx={n_x}, ny={n_y}, nz={n_z}")
print(f"ax={a_x}, ay={a_y}, az={a_z}")
print(f"\n--- Генерація сітки ---")
print(f"Масив AKT (координати вузлів) shape: {AKT.shape}")
print(f"Масив NT (зв'язність елементів) shape: {NT.shape}")
print(f"Загальна кількість вузлів (nqp): {nqp}")
print(f"Загальна кількість елементів (nel): {nel}")

# --- Обчислення DFIABG ---
print(f"\n--- Обчислення DFIABG ---")
DFIABG = compute_DFIABG(alpha_flat_g, beta_flat_g, gamma_flat_g)
if DFIABG is None or DFIABG.size == 0 or np.isnan(DFIABG).any():
    print("Помилка: DFIABG не було обчислено належним чином.")
    exit()
else:
   print(f"Масив DFIABG обчислено успішно. Shape: {DFIABG.shape}")
   # Виводимо похідні у першій точці Гаусса (індекс 0) для прикладу
   j = 0
   print(f"\nПриклад похідних DFIABG в точці Гаусса #{j + 1}:")
   for i in range(5): # Виведемо для перших 5 вузлів
       dphi_da = DFIABG[j, 0, i]
       dphi_db = DFIABG[j, 1, i]
       dphi_dg = DFIABG[j, 2, i]
       print(f"Вузол {i + 1:2d}: ∂ϕ/∂α = {dphi_da:+.5f}, ∂ϕ/∂β = {dphi_db:+.5f}, ∂ϕ/∂γ = {dphi_dg:+.5f}")

# --- Обчислення Якобіана для першого елемента ---
print(f"\n--- Обчислення Якобіана (Приклад) ---")
target_elem_id = 1
target_gauss_point_idx = 0

try:
    DJ = compute_jacobian(AKT, NT, target_elem_id, DFIABG, target_gauss_point_idx)
    print(f"Матриця Якобі для елемента {target_elem_id} у точці Гаусса #{target_gauss_point_idx+1}:")
    print(DJ)
    det_J = np.linalg.det(DJ)
    print(f"Детермінант Якобіана: {det_J:.6f}")
    if det_J <= 1e-9:
        print("ПОПЕРЕДЖЕННЯ: Детермінант Якобіана близький до нуля або від'ємний!")
except Exception as e:
    print(f"Помилка при обчисленні Якобіана: {e}")
    DJ = None # Щоб не викликати помилку далі

# --- Обчислення DFIXYZ для першого елемента ---
print(f"\n--- Обчислення DFIXYZ (Приклад) ---")
DFIXYZ = None # Ініціалізуємо
if DJ is not None: # Перевіряємо, чи Якобіан було обчислено
    try:
        DFIXYZ = compute_DFIXYZ_for_element(target_elem_id, AKT, NT, DFIABG)
    except np.linalg.LinAlgError:
         print(f"ПОМИЛКА: Матриця Якобі сингулярна при обчисленні DFIXYZ для елемента {target_elem_id}!")
         # DFIXYZ залишиться None або можна присвоїти NaN
         DFIXYZ = np.full((27, 20, 3), np.nan) # Заповнюємо NaN
    except Exception as e:
         print(f"Невідома помилка при обчисленні DFIXYZ: {e}")
         DFIXYZ = np.full((27, 20, 3), np.nan)

# Перевірка та вивід результату DFIXYZ
if DFIXYZ is None:
    print(f"Помилка: DFIXYZ для елемента {target_elem_id} не було обчислено через попередні помилки.")
elif np.isnan(DFIXYZ).any():
    print(f"Помилка: DFIXYZ для елемента {target_elem_id} містить NaN (ймовірно через сингулярний Якобіан).")
else:
    print(f"Масив DFIXYZ для елемента {target_elem_id} обчислено успішно. Shape: {DFIXYZ.shape}")
    j = target_gauss_point_idx
    print(f"\nПриклад похідних DFIXYZ в точці Гаусса #{j + 1}:")
    for i in range(5):
        dphi_dx = DFIXYZ[j, i, 0]
        dphi_dy = DFIXYZ[j, i, 1]
        dphi_dz = DFIXYZ[j, i, 2]
        print(f"Вузол {i + 1:2d}: ∂ϕ/∂x = {dphi_dx:+.5f}, ∂ϕ/∂y = {dphi_dy:+.5f}, ∂ϕ/∂z = {dphi_dz:+.5f}")

# --- Кінець коду, що стосується DFIXYZ ---
print("\n--- Завершення етапу обчислення DFIXYZ ---")

plt.show()