# src/fem_app/ui/visualization.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

# Імпортуємо FACE_NODES_MAP для отримання даних про грані
try:
    from fem_app.core.boundary import FACE_NODES_MAP
except ImportError:
    print("ПОМИЛКА: Не вдалося імпортувати FACE_NODES_MAP з fem_app.core.boundary у visualization.py")
    # Створюємо заглушку, щоб уникнути падіння, але функціонал буде обмежений
    FACE_NODES_MAP = {i: {"nodes_3d_indices": [], "normal_vector_direction": [0,0,0]} for i in range(1, 7)}


def get_element_edges(nodes_coords_3d_T):
    """
    Повертає список пар координат для ребер 20-вузлового елемента.
    nodes_coords_3d_T: масив (3, 20) з координатами вузлів.
    """
    edge_node_indices = [
        (0,8), (8,1), (1,9), (9,2), (2,10), (10,3), (3,11), (11,0), # Нижня грань
        (4,16), (16,5), (5,17), (17,6), (6,18), (18,7), (7,19), (19,4), # Верхня грань
        (0,12), (12,4), (1,13), (13,5), (2,14), (14,6), (3,15), (15,7)  # Вертикальні ребра
    ]
    
    edges_coords = []
    if nodes_coords_3d_T.shape[1] < 20: # Перевірка на достатність вузлів
        # print(f"Warning: Not enough nodes ({nodes_coords_3d_T.shape[1]}) for full 20-node element edges.")
        return [] # Повертаємо порожній список, якщо вузлів недостатньо

    for n1_idx, n2_idx in edge_node_indices:
        # Переконуємося, що індекси в межах фактичної кількості переданих вузлів
        if n1_idx < nodes_coords_3d_T.shape[1] and n2_idx < nodes_coords_3d_T.shape[1]:
            p1 = nodes_coords_3d_T[:, n1_idx]
            p2 = nodes_coords_3d_T[:, n2_idx]
            edges_coords.append([(p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2])])
    return edges_coords

def get_face_center(face_node_coords_T):
    """ Обчислює центр грані (середнє арифметичне координат її кутових вузлів).
        face_node_coords_T: масив (3, num_face_nodes) координат вузлів грані.
    """
    if face_node_coords_T.shape[1] == 0:
        return np.array([0,0,0])
    # Використовуємо лише кутові вузли для центру (перші 4 з 8) для стабільності
    num_nodes_for_center = min(4, face_node_coords_T.shape[1])
    return np.mean(face_node_coords_T[:, :num_nodes_for_center], axis=1)


def draw_mesh(ax, AKT_orig, NT, ax_dims,
              show_nodes=True, show_node_labels=True, show_element_outline=True,
              AKT_deformed=None, deformation_scale=1.0, show_orig_wireframe=True,
              show_face_numbers=False, loaded_faces_data=None): # Нові параметри
    """
    Малює 3D сітку. Може відображати оригінальну або деформовану сітку,
    номери граней та напрямок тиску.
    ax_dims: кортеж (a_x, a_y, a_z) для габаритного контуру.
    loaded_faces_data: список кортежів (elem_idx_0based, face_id, signed_pressure_value, normal_vector)
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
        displacements = AKT_deformed - AKT_orig
        scaled_displacements = displacements * deformation_scale
        current_AKT_to_draw = AKT_orig + scaled_displacements
        title_suffix = f" (Деформована, масштаб x{deformation_scale:.1f})"

        if show_orig_wireframe:
            for i_elem in range(nel):
                elem_nodes_indices_1based = NT[:, i_elem]
                valid_nodes_mask = elem_nodes_indices_1based > 0
                elem_nodes_indices_0based = elem_nodes_indices_1based[valid_nodes_mask] - 1
                
                if elem_nodes_indices_0based.size > 0 and np.all(elem_nodes_indices_0based < AKT_orig.shape[1]):
                    elem_coords_orig = AKT_orig[:, elem_nodes_indices_0based]
                    # Переконуємося, що elem_coords_orig має достатньо вузлів для get_element_edges
                    if elem_coords_orig.shape[1] == 20:
                         orig_edges = get_element_edges(elem_coords_orig)
                         if orig_edges:
                             lc_orig = Line3DCollection(orig_edges, colors='gray', linewidths=0.5, linestyles=':')
                             ax.add_collection(lc_orig)
                # else:
                #     print(f"Попередження: Індекси вузлів для оригінального каркасу елемента {i_elem+1} виходять за межі AKT_orig або їх недостатньо.")

    # Малюємо основну сітку (оригінальну або деформовану)
    all_elem_edges = []
    for i_elem in range(nel):
        elem_nodes_indices_1based = NT[:, i_elem]
        valid_nodes_mask = elem_nodes_indices_1based > 0
        elem_nodes_indices_0based = elem_nodes_indices_1based[valid_nodes_mask] - 1

        if elem_nodes_indices_0based.size > 0 and np.all(elem_nodes_indices_0based < current_AKT_to_draw.shape[1]):
            elem_coords = current_AKT_to_draw[:, elem_nodes_indices_0based]
            if elem_coords.shape[1] == 20: # Тільки для повних 20-вузлових елементів
                elem_edges = get_element_edges(elem_coords)
                all_elem_edges.extend(elem_edges)

                # Нумерація граней для поточного елемента
                if show_face_numbers:
                    for face_id in range(1, 7): # Грані 1-6
                        if face_id in FACE_NODES_MAP:
                            face_local_node_indices = FACE_NODES_MAP[face_id]["nodes_3d_indices"]
                            # Переконуємося, що всі локальні індекси вузлів грані є в межах вузлів елемента
                            if all(idx < elem_coords.shape[1] for idx in face_local_node_indices):
                                face_actual_nodes_coords = elem_coords[:, face_local_node_indices]
                                if face_actual_nodes_coords.shape[1] > 0: # Якщо є хоча б один вузол для грані
                                    center_coords = get_face_center(face_actual_nodes_coords)
                                    ax.text(center_coords[0], center_coords[1], center_coords[2],
                                            str(face_id), color='purple', fontsize=8, ha='center', va='center', zorder=15)
        # else:
        #     print(f"Попередження: Індекси вузлів для елемента {i_elem+1} виходять за межі current_AKT_to_draw або їх недостатньо.")


    if all_elem_edges:
        lc = Line3DCollection(all_elem_edges, colors='black', linewidths=1.0 if AKT_deformed is None else 0.7)
        ax.add_collection(lc)

    # Відображення стрілок тиску
    if loaded_faces_data:
        arrow_length_scale = 0.1 * min(ax_dims) if min(ax_dims) > 0 else 0.1 # Масштаб для довжини стрілки
        for elem_idx_0based, face_id, signed_pressure_val, normal_vector_map in loaded_faces_data:
            if elem_idx_0based >= nel: continue # Перевірка меж

            elem_nodes_global_1based = NT[:, elem_idx_0based]
            valid_mask = elem_nodes_global_1based > 0
            elem_nodes_global_0based = elem_nodes_global_1based[valid_mask] - 1

            if elem_nodes_global_0based.size > 0 and np.all(elem_nodes_global_0based < current_AKT_to_draw.shape[1]):
                elem_coords_current = current_AKT_to_draw[:, elem_nodes_global_0based]
                
                if face_id in FACE_NODES_MAP:
                    face_local_node_indices_on_elem = FACE_NODES_MAP[face_id]["nodes_3d_indices"]
                    
                    # Відфільтрувати індекси, які є в межах фактичних вузлів елемента
                    valid_local_indices = [idx for idx in face_local_node_indices_on_elem if idx < elem_coords_current.shape[1]]

                    if valid_local_indices:
                        face_actual_nodes_coords_current = elem_coords_current[:, valid_local_indices]
                        if face_actual_nodes_coords_current.shape[1] > 0:
                            center_coords = get_face_center(face_actual_nodes_coords_current)
                            
                            # Напрямок стрілки: signed_pressure * normal_vector_map
                            # normal_vector_map - це напрямок зовнішньої нормалі
                            # Якщо signed_pressure > 0 (розтяг), стрілка вздовж нормалі (назовні)
                            # Якщо signed_pressure < 0 (стиск), стрілка проти нормалі (всередину)
                            # Отже, вектор стрілки = normal_vector_map (напрямок сили)
                            # Колір стрілки може залежати від знаку тиску
                            arrow_color = 'green' if signed_pressure_val > 0 else 'blue'
                            
                            # Компоненти вектора сили (для напрямку стрілки)
                            # Сила = Тиск * Площа * Нормаль. Напрямок сили = знак(Тиск) * Нормаль
                            # Але compute_element_load_vector використовує: force_scalar_on_node * normal_vector_direction_in
                            # Де force_scalar_on_node містить signed_pressure_val.
                            # Отже, напрямок сили = signed_pressure_val * normal_vector_map (якщо normal_vector_map одиничний)
                            # Або просто normal_vector_map, а знак тиску визначає напрямок вздовж/проти.
                            # Для візуалізації: стрілка вздовж normal_vector_map, якщо тиск розтягуючий (P > 0)
                            # стрілка проти normal_vector_map, якщо тиск стискаючий (P < 0)
                            # Але normal_vector_map це ВЖЕ напрям зовнішньої нормалі.
                            # Тоді вектор сили = signed_pressure_val * normal_vector_map (якщо нормаль одинична)
                            # Або напрям стрілки = normal_vector_map, а її "початок/кінець" залежить від знаку.
                            # Простіше: напрям стрілки = normal_vector_map * sign(signed_pressure_val)
                            # Ні, ще простіше: вектор стрілки = signed_pressure_val * normal_vector_map (якщо нормаль не нормована, то це вектор сили)
                            # Для quiver: ax.quiver(X, Y, Z, U, V, W) де U,V,W - компоненти вектора
                            
                            # Компоненти вектора стрілки (пропорційні тиску і вздовж нормалі)
                            # Якщо signed_pressure_val < 0 (стиск), то стрілка має йти ПРОТИ normal_vector_map
                            # Якщо signed_pressure_val > 0 (розтяг), то стрілка має йти ВЗДОВЖ normal_vector_map
                            # Отже, напрямок стрілки = normal_vector_map. Знак тиску визначає, чи це "вхідний" чи "вихідний" потік.
                            # Для візуалізації сили, напрямок стрілки = signed_pressure_val * normal_vector_map (якщо нормаль одинична)
                            # Або, якщо normal_vector_map це просто напрям зовнішньої нормалі:
                            # u,v,w = normal_vector_map. Компоненти стрілки.
                            # Якщо signed_pressure_val < 0, то стрілка має бути напрямлена всередину, тобто проти normal_vector_map.
                            # Якщо signed_pressure_val > 0, то стрілка назовні, вздовж normal_vector_map.
                            
                            # Отже, компоненти стрілки:
                            # nx, ny, nz = normal_vector_map[0], normal_vector_map[1], normal_vector_map[2]
                            # Якщо signed_pressure_val < 0 (стиск), стрілка має йти всередину.
                            # Тобто, якщо normal_vector_map це зовнішня нормаль, то стрілка = -normal_vector_map
                            # Якщо signed_pressure_val > 0 (розтяг), стрілка назовні, стрілка = normal_vector_map
                            # Значить, напрямок стрілки = normal_vector_map * np.sign(signed_pressure_val) -- це якщо signed_pressure_val це величина.
                            # Але signed_pressure_val вже має знак.
                            # Якщо signed_pressure_val < 0 (стиск), то сила діє проти зовнішньої нормалі.
                            # Якщо signed_pressure_val > 0 (розтяг), то сила діє вздовж зовнішньої нормалі.
                            # Отже, вектор сили = signed_pressure_val * (нормалізована зовнішня нормаль).
                            # normal_vector_map - це вже напрям зовнішньої нормалі.
                            # Тоді компоненти стрілки:
                            # u = normal_vector_map[0]
                            # v = normal_vector_map[1]
                            # w = normal_vector_map[2]
                            # Якщо signed_pressure_val < 0, то стрілку треба розвернути.
                            # Тобто, u_eff = u * np.sign(signed_pressure_val), v_eff = v * np.sign(signed_pressure_val), ... НІ.

                            # Логіка з boundary.py: fe_elem[idx] += force_scalar * normal_vector_direction_in[0]
                            # force_scalar = N_i_2d * pressure_value * det_J_surf * weight_gp
                            # Тобто, якщо pressure_value від'ємний, то сила буде проти normal_vector_direction_in.
                            # Отже, напрямок стрілки має бути вздовж normal_vector_map, а її "колір" або "початок"
                            # може вказувати на стиск/розтяг.
                            # Або просто малюємо стрілку в напрямку сили: signed_pressure_val * normal_vector_map (якщо нормаль одинична)
                            
                            # Для quiver, U,V,W це компоненти вектора.
                            # Вектор сили F = P * A * n. Напрямок сили n_force = sign(P) * n_outward.
                            # Або, якщо P вже зі знаком: n_force = P * n_outward (якщо P - тиск, а не сила).
                            # У нас signed_pressure_val. normal_vector_map - це зовнішня нормаль.
                            # Тоді вектор сили пропорційний signed_pressure_val * normal_vector_map.
                            # Компоненти для quiver:
                            u_comp = normal_vector_map[0]
                            v_comp = normal_vector_map[1]
                            w_comp = normal_vector_map[2]

                            # Якщо тиск стискаючий (signed_pressure_val < 0), стрілка має бути напрямлена всередину (проти зовнішньої нормалі)
                            # Якщо тиск розтягуючий (signed_pressure_val > 0), стрілка назовні (вздовж зовнішньої нормалі)
                            # Отже, якщо signed_pressure_val < 0, то компоненти u,v,w треба інвертувати.
                            # Це еквівалентно u_final = u_comp * np.sign(signed_pressure_val) ЯКЩО normal_vector_map це напрямок,
                            # а signed_pressure_val це величина.
                            # АЛЕ: signed_pressure_val вже має знак. normal_vector_map це зовнішня нормаль.
                            # Сила = signed_pressure_val * Площа * (нормалізована normal_vector_map).
                            # Напрямок сили = signed_pressure_val * normal_vector_map (якщо нормаль не нормована, а її довжина = 1).
                            # Тоді для quiver:
                            # U = signed_pressure_val * normal_vector_map[0]
                            # V = signed_pressure_val * normal_vector_map[1]
                            # W = signed_pressure_val * normal_vector_map[2]
                            # Це має працювати. Довжина стрілки буде пропорційна signed_pressure_val.
                            
                            # Довжина стрілки для візуалізації (не обов'язково фізична)
                            current_arrow_len = arrow_length_scale # фіксована довжина для візуалізації напрямку
                            
                            # Компоненти напрямку сили
                            force_dir_x = normal_vector_map[0]
                            force_dir_y = normal_vector_map[1]
                            force_dir_z = normal_vector_map[2]

                            # Якщо тиск стискаючий, розвертаємо напрямок стрілки
                            if signed_pressure_val < 0:
                                force_dir_x *= -1
                                force_dir_y *= -1
                                force_dir_z *= -1
                                arrow_color = 'blue' # Стиск
                            else:
                                arrow_color = 'green' # Розтяг


                            ax.quiver(center_coords[0], center_coords[1], center_coords[2],
                                      force_dir_x, force_dir_y, force_dir_z,
                                      length=current_arrow_len, normalize=True, color=arrow_color,
                                      arrow_length_ratio=0.3, zorder=20, linewidth=1.5)


    if show_nodes:
        ax.scatter(current_AKT_to_draw[0,:], current_AKT_to_draw[1,:], current_AKT_to_draw[2,:],
                   c='red' if AKT_deformed is not None else 'blue', 
                   s=20, depthshade=True, alpha=0.7, zorder=10)

    if show_node_labels:
        for i in range(nqp):
            if i < current_AKT_to_draw.shape[1]:
                ax.text(current_AKT_to_draw[0,i], current_AKT_to_draw[1,i], current_AKT_to_draw[2,i],
                        str(i+1), size=7, color='darkred', zorder=10)

    if show_element_outline: # Раніше було show_outline, перейменовано для ясності
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

    all_coords_for_lims = AKT_orig.copy()
    if AKT_deformed is not None: 
        all_coords_for_lims = np.hstack((all_coords_for_lims, current_AKT_to_draw))
    
    if all_coords_for_lims.size == 0: # Якщо масив порожній
        min_c = np.array([-0.5*max(1,a_x), -0.5*max(1,a_y), -0.5*max(1,a_z)])
        max_c = np.array([0.5*max(1,a_x), 0.5*max(1,a_y), 0.5*max(1,a_z)])
    else:
        min_c = all_coords_for_lims.min(axis=1)
        max_c = all_coords_for_lims.max(axis=1)

    center = (max_c + min_c) / 2.0
    # Якщо min_c і max_c однакові (наприклад, один вузол або плоска структура), діапазон буде 0
    range_dims = max_c - min_c
    # Запобігаємо нульовому діапазону, використовуючи розміри області, якщо діапазон занадто малий
    if np.all(range_dims < 1e-6):
        range_dims = np.array([max(1e-6, a_x), max(1e-6, a_y), max(1e-6, a_z)])
        min_c = center - range_dims / 2.0 # Перераховуємо min_c, max_c для встановлення меж
        max_c = center + range_dims / 2.0


    auto_scale_range = range_dims.max()
    if auto_scale_range < 1e-6 : auto_scale_range = max(a_x,a_y,a_z,1.0) 
    
    margin_factor = 0.2 
    # Якщо є стрілки тиску, може знадобитися більший відступ
    if loaded_faces_data and any(loaded_faces_data): margin_factor = 0.3

    margin = auto_scale_range * margin_factor
    
    ax.set_xlim(min_c[0] - margin, max_c[0] + margin)
    ax.set_ylim(min_c[1] - margin, max_c[1] + margin)
    ax.set_zlim(min_c[2] - margin, max_c[2] + margin)
    
    try:
        ax.set_aspect('equal', adjustable='box')
    except NotImplementedError:
        # Встановлюємо пропорції вручну, якщо set_aspect('equal') не працює
        # Це важливо для коректного відображення 3D об'єктів
        ranges = [max_c[i] - min_c[i] + 2 * margin for i in range(3)]
        ranges = [r if r > 1e-6 else 1.0 for r in ranges] # Уникаємо ділення на нуль або дуже малі значення
        ax.set_box_aspect(ranges) # Потрібно передати діапазони по кожній осі

    ax.view_init(elev=20, azim=-65)
