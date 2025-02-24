import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numba import njit

# -------------------------------------------------------------
# 1️⃣ Параметри сітки
n_x, n_y, n_z = 2, 1, 1  # Кількість елементів у кожному напрямку
a_x, a_y, a_z = 2.0, 2.0, 2.0  # Розміри тіла

# -------------------------------------------------------------
# 2️⃣ Генерація вузлів із `Numba`
@njit
def is_corner_or_edge(ix, iy, iz):
    """Перевіряє, чи точка - це кут або середина ребра."""
    c_odd = (ix % 2) + (iy % 2) + (iz % 2)
    return c_odd <= 1  # максимум одна координата == 1 (не грань!)

@njit
def generate_mesh(n_x, n_y, n_z, a_x, a_y, a_z):
    """Генерує координати вузлів у форматі (3, nqp)"""
    Nx, Ny, Nz = 2 * n_x + 1, 2 * n_y + 1, 2 * n_z + 1
    nodes = np.zeros((3, Nx * Ny * Nz))  # Масив для координат
    node_map = {}
    iNode = 0

    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                if is_corner_or_edge(ix, iy, iz):
                    x, y, z = ix * (a_x / (Nx - 1)), iy * (a_y / (Ny - 1)), iz * (a_z / (Nz - 1))
                    node_map[(ix, iy, iz)] = iNode
                    nodes[:, iNode] = (x, y, z)  # Формат (3, nqp)
                    iNode += 1

    return nodes[:, :iNode], node_map  # Повертаємо правильний масив AKT

# Генеруємо сітку
AKT, node_map = generate_mesh(n_x, n_y, n_z, a_x, a_y, a_z)
nqp = AKT.shape[1]  # Загальна кількість вузлів

# -------------------------------------------------------------
# 3️⃣ Генерація таблиці `NT` для 20-вузлових елементів із `Numba`
@njit
def generate_connectivity(n_x, n_y, n_z, node_map):
    """Генерує таблицю `NT`, яка містить вузли кожного 20-вузлового елемента"""
    local_offsets_20 = []
    for lx in [0, 1, 2]:
        for ly in [0, 1, 2]:
            for lz in [0, 1, 2]:
                c1 = (lx == 1) + (ly == 1) + (lz == 1)
                if c1 <= 1:
                    local_offsets_20.append((lx, ly, lz))

    nel = n_x * n_y * n_z
    NT = np.zeros((20, nel), dtype=np.int32)

    elem_id = 0
    for ez in range(n_z):
        for ey in range(n_y):
            for ex in range(n_x):
                for i_loc, (lx, ly, lz) in enumerate(local_offsets_20):
                    gx, gy, gz = 2 * ex + lx, 2 * ey + ly, 2 * ez + lz
                    g_id = node_map[(gx, gy, gz)]
                    NT[i_loc, elem_id] = g_id
                elem_id += 1

    return NT

# Генеруємо таблицю `NT`
NT = generate_connectivity(n_x, n_y, n_z, node_map)
nel = NT.shape[1]  # Кількість елементів

# -------------------------------------------------------------
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

plt.show()

# -------------------------------------------------------------
# Виводимо кількість вузлів та елементів
print(f"Масив AKT (3, {nqp}):\n", AKT)
print(f"Загальна кількість вузлів (nqp): {nqp}")
print(f"Загальна кількість елементів (nel): {nel}")
