# src/fem_app/ui/visualization.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

def get_element_edges(nodes_coords_3d_T):
    """
    Повертає список пар координат для ребер 20-вузлового елемента.
    nodes_coords_3d_T: масив (3, 20) з координатами вузлів.
    """
    # Індекси вузлів, що утворюють ребра (0-based для 20 вузлів)
    # Кутові вузли: 0-7
    # Серединні вузли: 8-19
    edge_node_indices = [
        (0,8), (8,1), (1,9), (9,2), (2,10), (10,3), (3,11), (11,0), # Нижня грань
        (4,16), (16,5), (5,17), (17,6), (6,18), (18,7), (7,19), (19,4), # Верхня грань
        (0,12), (12,4), (1,13), (13,5), (2,14), (14,6), (3,15), (15,7)  # Вертикальні ребра
    ]
    
    edges_coords = []
    for n1_idx, n2_idx in edge_node_indices:
        if n1_idx < nodes_coords_3d_T.shape[1] and n2_idx < nodes_coords_3d_T.shape[1]:
            p1 = nodes_coords_3d_T[:, n1_idx]
            p2 = nodes_coords_3d_T[:, n2_idx]
            edges_coords.append([(p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2])])
    return edges_coords

def draw_mesh(ax, AKT_orig, NT, ax_dims,
              show_nodes=True, show_node_labels=True, show_outline=True,
              AKT_deformed=None, deformation_scale=1.0, show_orig_wireframe=True):
    """
    Малює 3D сітку. Може відображати оригінальну або деформовану сітку.
    ax_dims: кортеж (a_x, a_y, a_z) для габаритного контуру.
    """
    ax.clear()

    if AKT_orig is None or NT is None or AKT_orig.size == 0 or NT.size == 0:
        ax.text(0.5, 0.5, 0.5, "Сітку не згенеровано", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("Скінченно-елементна сітка")
        return

    a_x, a_y, a_z = ax_dims
    nqp = AKT_orig.shape[1]
    nel = NT.shape[1]

    current_AKT_to_draw = AKT_orig
    title_suffix = ""

    if AKT_deformed is not None:
        # Розраховуємо переміщення та масштабуємо їх
        displacements = AKT_deformed - AKT_orig
        scaled_displacements = displacements * deformation_scale
        current_AKT_to_draw = AKT_orig + scaled_displacements
        title_suffix = f" (Деформована, масштаб x{deformation_scale:.1f})"

        if show_orig_wireframe:
            # Малюємо каркас оригінальної сітки
            for i_elem in range(nel):
                elem_nodes_indices_1based = NT[:, i_elem]
                valid_nodes_mask = elem_nodes_indices_1based > 0 # Враховуємо тільки валідні
                elem_nodes_indices_0based = elem_nodes_indices_1based[valid_nodes_mask] - 1
                
                if elem_nodes_indices_0based.size == 20: # Переконуємось, що це 20-вузловий елемент
                    if np.all(elem_nodes_indices_0based < AKT_orig.shape[1]):
                        elem_coords_orig = AKT_orig[:, elem_nodes_indices_0based]
                        orig_edges = get_element_edges(elem_coords_orig)
                        lc_orig = Line3DCollection(orig_edges, colors='gray', linewidths=0.5, linestyles=':')
                        ax.add_collection(lc_orig)
                    else:
                        print(f"Попередження: Індекси вузлів для оригінального каркасу елемента {i_elem+1} виходять за межі AKT_orig.")


    # Малюємо основну сітку (оригінальну або деформовану)
    all_elem_edges = []
    for i_elem in range(nel):
        elem_nodes_indices_1based = NT[:, i_elem]
        valid_nodes_mask = elem_nodes_indices_1based > 0
        elem_nodes_indices_0based = elem_nodes_indices_1based[valid_nodes_mask] - 1

        if elem_nodes_indices_0based.size == 20: # Тільки для повних 20-вузлових елементів
            if np.all(elem_nodes_indices_0based < current_AKT_to_draw.shape[1]):
                elem_coords = current_AKT_to_draw[:, elem_nodes_indices_0based]
                elem_edges = get_element_edges(elem_coords)
                all_elem_edges.extend(elem_edges)
            else:
                 print(f"Попередження: Індекси вузлів для елемента {i_elem+1} виходять за межі current_AKT_to_draw.")


    if all_elem_edges:
        lc = Line3DCollection(all_elem_edges, colors='black', linewidths=1.0)
        ax.add_collection(lc)

    if show_nodes:
        ax.scatter(current_AKT_to_draw[0,:], current_AKT_to_draw[1,:], current_AKT_to_draw[2,:],
                   c='red' if AKT_deformed is not None else 'blue', 
                   s=20, depthshade=True, alpha=0.7)

    if show_node_labels:
        for i in range(nqp):
            # Перевірка, чи індекс не виходить за межі масиву
            if i < current_AKT_to_draw.shape[1]:
                ax.text(current_AKT_to_draw[0,i], current_AKT_to_draw[1,i], current_AKT_to_draw[2,i],
                        str(i+1), size=7, color='darkred', zorder=10)

    if show_outline:
        corners = np.array([[0,0,0],[a_x,0,0],[a_x,a_y,0],[0,a_y,0],
                            [0,0,a_z],[a_x,0,a_z],[a_x,a_y,a_z],[0,a_y,a_z]])
        faces_outline = [
            [corners[0],corners[1],corners[2],corners[3]], [corners[4],corners[5],corners[6],corners[7]],
            [corners[0],corners[1],corners[5],corners[4]], [corners[2],corners[3],corners[7],corners[6]],
            [corners[1],corners[2],corners[6],corners[5]], [corners[0],corners[3],corners[7],corners[4]]
        ]
        ax.add_collection3d(Poly3DCollection(faces_outline, facecolors='cyan',
                                              edgecolors='darkgrey', linewidths=0.5, alpha=0.05, zorder=-1))

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Скінченно-елементна сітка" + title_suffix)

    # Встановлення меж для кращої візуалізації
    all_coords_for_lims = AKT_orig # Використовуємо оригінальні для стабільних меж
    if AKT_deformed is not None: # Якщо є деформація, розширюємо межі
        all_coords_for_lims = np.hstack((AKT_orig, current_AKT_to_draw))

    min_c = all_coords_for_lims.min(axis=1)
    max_c = all_coords_for_lims.max(axis=1)
    center = (max_c + min_c) / 2.0
    auto_scale_range = (max_c - min_c).max()
    if auto_scale_range < 1e-6 : auto_scale_range = max(a_x,a_y,a_z,1.0) # Якщо тіло не деформується або дуже мале
    
    margin = auto_scale_range * 0.2 # Додаємо невеликий відступ
    ax.set_xlim(center[0] - auto_scale_range/2 - margin, center[0] + auto_scale_range/2 + margin)
    ax.set_ylim(center[1] - auto_scale_range/2 - margin, center[1] + auto_scale_range/2 + margin)
    ax.set_zlim(center[2] - auto_scale_range/2 - margin, center[2] + auto_scale_range/2 + margin)
    
    try:
        ax.set_aspect('equal', adjustable='box') # Спроба встановити рівні масштаби
    except NotImplementedError:
        ax.set_box_aspect(( (max_c[0]-min_c[0]) if (max_c[0]-min_c[0]) > 1e-6 else 1,
                            (max_c[1]-min_c[1]) if (max_c[1]-min_c[1]) > 1e-6 else 1,
                            (max_c[2]-min_c[2]) if (max_c[2]-min_c[2]) > 1e-6 else 1))
    ax.view_init(elev=20, azim=-65)

