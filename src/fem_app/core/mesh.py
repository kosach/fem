import numpy as np
# Numba тут не використовується через несумісність node_map зі словниками

def is_corner_or_edge(ix, iy, iz):
    """Перевіряє, чи локальний індекс відповідає куту або середині ребра."""
    c_odd = (ix % 2) + (iy % 2) + (iz % 2)
    return c_odd <= 1

def generate_mesh(n_x, n_y, n_z, a_x, a_y, a_z):
    """
    Генерує координати вузлів AKT(3, nqp) та словник node_map.
    Повертає: AKT, node_map, nqp (кількість вузлів)
    """
    Nx, Ny, Nz = 2 * n_x + 1, 2 * n_y + 1, 2 * n_z + 1
    max_possible_nodes = Nx * Ny * Nz
    nodes = np.zeros((3, max_possible_nodes), dtype=np.float64)
    node_map = {}
    iNode = 0
    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                if is_corner_or_edge(ix, iy, iz):
                    x = ix * (a_x / (Nx - 1)) if Nx > 1 else 0.0
                    y = iy * (a_y / (Ny - 1)) if Ny > 1 else 0.0
                    z = iz * (a_z / (Nz - 1)) if Nz > 1 else 0.0
                    node_map[(ix, iy, iz)] = iNode
                    nodes[:, iNode] = np.array([x, y, z])
                    iNode += 1
    return nodes[:, :iNode], node_map, iNode

def generate_connectivity(n_x, n_y, n_z, node_map):
    """
    Генерує таблицю зв'язності NT(20, nel).
    Повертає: NT, nel (кількість елементів)
    """
    local_offsets_20 = [
        (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),   # 1-4
        (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),   # 5-8
        (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0),   # 9-12
        (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1),   # 13-16
        (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2)    # 17-20
    ]
    nel = n_x * n_y * n_z
    if nel <= 0:
        return np.zeros((20, 0), dtype=np.int32), 0

    NT = np.zeros((20, nel), dtype=np.int32)
    elem_id = 0
    for ez in range(n_z):
        for ey in range(n_y):
            for ex in range(n_x):
                valid_element = True
                for i_loc, (lx, ly, lz) in enumerate(local_offsets_20):
                    gx, gy, gz = 2 * ex + lx, 2 * ey + ly, 2 * ez + lz
                    node_num_0based = node_map.get((gx, gy, gz))
                    if node_num_0based is None:
                        print(f"Error: Node ({gx},{gy},{gz}) not found for elem {elem_id+1}, loc node {i_loc+1}")
                        NT[i_loc, elem_id] = -999 # Mark invalid
                        valid_element = False
                    else:
                        NT[i_loc, elem_id] = node_num_0based + 1 # 1-based index
                # if not valid_element: # Поки що не перериваємо, але фіксуємо помилку
                #     print(f"Error: Element {elem_id+1} has invalid nodes.")
                elem_id += 1
    return NT, nel