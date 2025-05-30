# fem_app/core/boundary.py
import numpy as np
from numba import njit


FACE_NODES_MAP = {
    1: { # Грань alpha = -1 (передня, YZ-площина; локальні 2D: eta=beta, tau=gamma)
        # 2D (-1,-1) -> 3D (-1,-1,-1) [loc_idx 0]
        # 2D ( 1,-1) -> 3D (-1, 1,-1) [loc_idx 3]
        # 2D ( 1, 1) -> 3D (-1, 1, 1) [loc_idx 7]
        # 2D (-1, 1) -> 3D (-1,-1, 1) [loc_idx 4]
        # 2D ( 0,-1) -> 3D (-1, 0,-1) [loc_idx 11]
        # 2D ( 1, 0) -> 3D (-1, 1, 0) [loc_idx 15]
        # 2D ( 0, 1) -> 3D (-1, 0, 1) [loc_idx 19]
        # 2D (-1, 0) -> 3D (-1,-1, 0) [loc_idx 12]
        "nodes_3d_indices": [0, 3, 7, 4, 11, 15, 19, 12],
        "normal_vector_direction": [-1, 0, 0]
    },
    2: { # Грань alpha = +1 (задня, YZ-площина; локальні 2D: eta=-beta, tau=gamma або інша відповідність)
         # Щоб узгодити з LOCAL_NODE_COORDS_2D_FACE, де (-1,-1) перший:
         # 2D (-1,-1) -> 3D (1, 1,-1) [loc_idx 2] (якщо eta -> -beta, tau -> -gamma)
         # Або 2D (-1,-1) -> 3D (1,-1,-1) [loc_idx 1] (якщо eta -> beta, tau -> -gamma) - ЦЕЙ ВАРІАНТ ОБРАНО
         # 2D ( 1,-1) -> 3D (1, 1,-1) [loc_idx 2]
         # 2D ( 1, 1) -> 3D (1, 1, 1) [loc_idx 6]
         # 2D (-1, 1) -> 3D (1,-1, 1) [loc_idx 5]
         # 2D ( 0,-1) -> 3D (1, 0,-1) [loc_idx 9]
         # 2D ( 1, 0) -> 3D (1, 1, 0) [loc_idx 14]
         # 2D ( 0, 1) -> 3D (1, 0, 1) [loc_idx 17]
         # 2D (-1, 0) -> 3D (1,-1, 0) [loc_idx 13]
        "nodes_3d_indices": [1, 2, 6, 5, 9, 14, 17, 13],
        "normal_vector_direction": [1, 0, 0]
    },
    3: { # Грань beta = -1 (ліва, XZ-площина; локальні 2D: eta=-alpha, tau=gamma)
         # 2D (-1,-1) -> 3D (1,-1,-1)  [loc_idx 1]
         # 2D ( 1,-1) -> 3D (-1,-1,-1) [loc_idx 0]
         # 2D ( 1, 1) -> 3D (-1,-1, 1) [loc_idx 4]
         # 2D (-1, 1) -> 3D (1,-1, 1)  [loc_idx 5]
         # 2D ( 0,-1) -> 3D (0,-1,-1)  [loc_idx 8]
         # 2D ( 1, 0) -> 3D (-1,-1, 0) [loc_idx 12]
         # 2D ( 0, 1) -> 3D (0,-1, 1)  [loc_idx 16]
         # 2D (-1, 0) -> 3D (1,-1, 0)  [loc_idx 13]
        "nodes_3d_indices": [1, 0, 4, 5, 8, 12, 16, 13], # Перевірити порядок!
        "normal_vector_direction": [0, -1, 0]
    },
    4: { # Грань beta = +1 (права, XZ-площина; локальні 2D: eta=alpha, tau=gamma)
         # 2D (-1,-1) -> 3D (-1,1,-1) [loc_idx 3]
         # 2D ( 1,-1) -> 3D (1,1,-1)  [loc_idx 2]
         # 2D ( 1, 1) -> 3D (1,1,1)   [loc_idx 6]
         # 2D (-1, 1) -> 3D (-1,1,1)  [loc_idx 7]
         # 2D ( 0,-1) -> 3D (0,1,-1)  [loc_idx 10]
         # 2D ( 1, 0) -> 3D (1,1,0)   [loc_idx 14]
         # 2D ( 0, 1) -> 3D (0,1,1)   [loc_idx 18]
         # 2D (-1, 0) -> 3D (-1,1,0)  [loc_idx 15]
        "nodes_3d_indices": [3, 2, 6, 7, 10, 14, 18, 15],
        "normal_vector_direction": [0, 1, 0]
    },
    5: { # Нижня грань (gamma = -1, XY-площина; локальні 2D: eta=alpha, tau=beta)
         # 2D (-1,-1) -> 3D (-1,-1,-1) [loc_idx 0]
         # 2D ( 1,-1) -> 3D (1,-1,-1)  [loc_idx 1]
         # 2D ( 1, 1) -> 3D (1,1,-1)   [loc_idx 2]
         # 2D (-1, 1) -> 3D (-1,1,-1)  [loc_idx 3]
         # 2D ( 0,-1) -> 3D (0,-1,-1)  [loc_idx 8]
         # 2D ( 1, 0) -> 3D (1,0,-1)   [loc_idx 9]
         # 2D ( 0, 1) -> 3D (0,1,-1)   [loc_idx 10]
         # 2D (-1, 0) -> 3D (-1,0,-1)  [loc_idx 11]
        "nodes_3d_indices": [0, 1, 2, 3, 8, 9, 10, 11],
        "normal_vector_direction": [0, 0, -1]
    },
    6: { # Верхня грань (gamma = +1, XY-площина; локальні 2D: eta=alpha, tau=-beta для збереження напрямку обходу)
         # Або eta=alpha, tau=beta і потім коригувати нормаль (простіше)
         # 2D (-1,-1) -> 3D (-1,-1,1) [loc_idx 4]
         # 2D ( 1,-1) -> 3D (1,-1,1)  [loc_idx 5]
         # 2D ( 1, 1) -> 3D (1,1,1)   [loc_idx 6]
         # 2D (-1, 1) -> 3D (-1,1,1)  [loc_idx 7]
         # 2D ( 0,-1) -> 3D (0,-1,1)  [loc_idx 16]
         # 2D ( 1, 0) -> 3D (1,0,1)   [loc_idx 17]
         # 2D ( 0, 1) -> 3D (0,1,1)   [loc_idx 18]
         # 2D (-1, 0) -> 3D (-1,0,1)  [loc_idx 19]
        "nodes_3d_indices": [4, 5, 6, 7, 16, 17, 18, 19],
        "normal_vector_direction": [0, 0, 1]
    },
}

LOCAL_NODE_COORDS_2D_FACE = np.array([
    [-1, -1], [ 1, -1], [ 1,  1], [-1,  1],
    [ 0, -1], [ 1,  0], [ 0,  1], [-1,  0]
], dtype=np.float64)

@njit
def N_2D(eta, tau, node_idx_2d, local_node_coords_2d):
    etai = local_node_coords_2d[node_idx_2d, 0]
    taui = local_node_coords_2d[node_idx_2d, 1]
    val = 0.0
    if node_idx_2d < 4: 
        val = 0.25 * (1 + eta * etai) * (1 + tau * taui) * (eta * etai + tau * taui - 1)
    else: 
        if abs(etai) > 1e-6 and abs(taui) < 1e-6 : 
            val = 0.5 * (1 + eta * etai) * (1 - tau * tau)
        elif abs(taui) > 1e-6 and abs(etai) < 1e-6: 
            val = 0.5 * (1 - eta * eta) * (1 + tau * taui)
    return val

# @njit
def dN_deta_2D(eta, tau, node_idx_2d, local_node_coords_2d):
    etai = local_node_coords_2d[node_idx_2d, 0]
    taui = local_node_coords_2d[node_idx_2d, 1]
    dval = 0.0
    # Додайте цей print на початку функції
    print(f"      DEBUG dN_deta_2D: node_2d={node_idx_2d}, eta_gp={eta:.4f}, tau_gp={tau:.4f}, etai_node={etai:.1f}, taui_node={taui:.1f}")

    if node_idx_2d < 4: # Кутові
        dval = 0.25 * etai * (1.0 + tau * taui) * (2.0 * eta * etai + tau * taui)
        print(f"        Кутовий, dval_deta={dval:.4f}")
    else: # Серединні
        if abs(etai) < 1e-6: # Вузол на ребрі eta = 0 (etai = 0, taui = +/-1)
            dval = 0.5 * (-2.0 * eta) * (1.0 + tau * taui)
            print(f"        Серединний (etai=0), dval_deta={dval:.4f}")
        elif abs(taui) < 1e-6: # Вузол на ребрі tau = 0 (taui = 0, etai = +/-1)
            dval = 0.5 * etai * (1.0 - tau * tau)
            print(f"        Серединний (taui=0), dval_deta={dval:.4f}")
        else:
            # Ця гілка не мала б виконуватися для стандартних 8 вузлів
            print(f"        Серединний, АЛЕ НЕ ПОТРАПИВ В УМОВИ! etai={etai}, taui={taui}")
    print(f"        FINAL dval_deta for node {node_idx_2d} = {dval:.4f}") # Остаточне значення
    return dval

# @njit
def dN_dtau_2D(eta, tau, node_idx_2d, local_node_coords_2d):
    etai = local_node_coords_2d[node_idx_2d, 0]
    taui = local_node_coords_2d[node_idx_2d, 1]
    dval = 0.0
    print(f"      DEBUG dN_dtau_2D: node_2d={node_idx_2d}, eta_gp={eta:.4f}, tau_gp={tau:.4f}, etai_node={etai:.1f}, taui_node={taui:.1f}")

    if node_idx_2d < 4: # Кутові
        dval = 0.25 * taui * (1.0 + eta * etai) * (eta * etai + 2.0 * tau * taui)
        print(f"        Кутовий, dval_dtau={dval:.4f}")
    else: # Серединні
        if abs(etai) < 1e-6: # Вузол на ребрі eta = 0 (etai = 0, taui = +/-1)
            dval = 0.5 * (1.0 - eta * eta) * taui
            print(f"        Серединний (etai=0), dval_dtau={dval:.4f}")
        elif abs(taui) < 1e-6: # Вузол на ребрі tau = 0 (taui = 0, etai = +/-1)
            dval = 0.5 * (1.0 + eta * etai) * (-2.0 * tau)
            print(f"        Серединний (taui=0), dval_dtau={dval:.4f}")
        else:
            print(f"        Серединний, АЛЕ НЕ ПОТРАПИВ В УМОВИ! etai={etai}, taui={taui}")
    print(f"        FINAL dval_dtau for node {node_idx_2d} = {dval:.4f}") # Остаточне значення
    return dval

_gp_1d_surf = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
_w_1d_surf = np.array([1.0, 1.0])

_ETA_G_SURF_MESH, _TAU_G_SURF_MESH = np.meshgrid(_gp_1d_surf, _gp_1d_surf, indexing='ij')
ETA_G_SURF_FLAT = _ETA_G_SURF_MESH.flatten()
TAU_G_SURF_FLAT = _TAU_G_SURF_MESH.flatten()

_W_ETA_SURF_MESH, _W_TAU_SURF_MESH = np.meshgrid(_w_1d_surf, _w_1d_surf, indexing='ij')
GAUSS_WEIGHTS_SURF_FLAT = (_W_ETA_SURF_MESH * _W_TAU_SURF_MESH).flatten()

# @njit
def compute_element_load_vector_numba_compatible(
    pressure_value,
    face_3d_node_indices_for_AKT_in,
    normal_vector_direction_in,
    AKT_elem_nodes_in,
    local_node_coords_2d_face_in,
    eta_g_surf_flat_in,
    tau_g_surf_flat_in,
    gauss_weights_surf_flat_in
    ):
    fe_elem = np.zeros(60)
        
    print(f"    INSIDE numba_compatible: pressure_value={pressure_value}")
    print(f"    INSIDE numba_compatible: face_3d_node_indices_for_AKT_in:\n{face_3d_node_indices_for_AKT_in}")
    print(f"    INSIDE numba_compatible: AKT_elem_nodes_in (форма {AKT_elem_nodes_in.shape}):\n{AKT_elem_nodes_in}")

    coords_face_nodes_3d = AKT_elem_nodes_in[:, face_3d_node_indices_for_AKT_in]
    
    print(f"    INSIDE numba_compatible: coords_face_nodes_3d (форма {coords_face_nodes_3d.shape}):\n{coords_face_nodes_3d}") # ДУЖЕ ВАЖЛИВО!
    num_gauss_points_surf = len(eta_g_surf_flat_in)

    for gp_surf_idx in range(num_gauss_points_surf):
        eta_gp = eta_g_surf_flat_in[gp_surf_idx]
        tau_gp = tau_g_surf_flat_in[gp_surf_idx]
        weight_gp = gauss_weights_surf_flat_in[gp_surf_idx] 
        print(f"      GP_surf {gp_surf_idx}: eta={eta_gp:.4f}, tau={tau_gp:.4f}, вага={weight_gp:.4f}")

        J_surf_3x2 = np.zeros((3, 2))
        for i_node_2d in range(8):
            dN_deta_i = dN_deta_2D(eta_gp, tau_gp, i_node_2d, local_node_coords_2d_face_in)
            dN_dtau_i = dN_dtau_2D(eta_gp, tau_gp, i_node_2d, local_node_coords_2d_face_in)
            x_i = coords_face_nodes_3d[0, i_node_2d]
            y_i = coords_face_nodes_3d[1, i_node_2d]
            z_i = coords_face_nodes_3d[2, i_node_2d]
            J_surf_3x2[0, 0] += dN_deta_i * x_i
            J_surf_3x2[0, 1] += dN_dtau_i * x_i
            J_surf_3x2[1, 0] += dN_deta_i * y_i
            J_surf_3x2[1, 1] += dN_dtau_i * y_i
            J_surf_3x2[2, 0] += dN_deta_i * z_i
            J_surf_3x2[2, 1] += dN_dtau_i * z_i


     #    print(f"      J_surf_3x2 для GP {gp_surf_idx}:\n{J_surf_3x2}")

        vec1_dXdeta = J_surf_3x2[:, 0]
        vec2_dXdtan = J_surf_3x2[:, 1]
        cross_product_vec = np.empty(3)
        cross_product_vec[0] = vec1_dXdeta[1]*vec2_dXdtan[2] - vec1_dXdeta[2]*vec2_dXdtan[1]
        cross_product_vec[1] = vec1_dXdeta[2]*vec2_dXdtan[0] - vec1_dXdeta[0]*vec2_dXdtan[2]
        cross_product_vec[2] = vec1_dXdeta[0]*vec2_dXdtan[1] - vec1_dXdeta[1]*vec2_dXdtan[0]
        det_J_surf = np.sqrt(cross_product_vec[0]**2 + cross_product_vec[1]**2 + cross_product_vec[2]**2)
        print(f"      det_J_surf: {det_J_surf}") # ДУЖЕ ВАЖЛИВО!

        if abs(det_J_surf) < 1e-12: # Збільшено допуск для numba
            continue
            
        for i_node_2d in range(8):
            N_i_2d = N_2D(eta_gp, tau_gp, i_node_2d, local_node_coords_2d_face_in)
            loc_node_3d_idx = face_3d_node_indices_for_AKT_in[i_node_2d]
            force_scalar_on_node = N_i_2d * pressure_value * det_J_surf * weight_gp
            
            print(f"        i_node_2d={i_node_2d}, loc_3d_idx={loc_node_3d_idx}, N_i_2d={N_i_2d:.4f}, force_scalar={force_scalar_on_node:.4e}")
            
            fe_elem[loc_node_3d_idx]      += force_scalar_on_node * normal_vector_direction_in[0]
            fe_elem[loc_node_3d_idx + 20] += force_scalar_on_node * normal_vector_direction_in[1]
            fe_elem[loc_node_3d_idx + 40] += force_scalar_on_node * normal_vector_direction_in[2]

    print(f"    INSIDE numba_compatible: Кінцевий fe_elem (сума abs: {np.sum(np.abs(fe_elem))} ):\n{fe_elem[fe_elem != 0]}")
    return fe_elem

def compute_element_load_vector(pressure_value, face_id, AKT_elem_nodes):
    print(f"  -- Всередині compute_element_load_vector (обгортка) --")
    print(f"     pressure_value: {pressure_value}, face_id: {face_id}")
    if face_id not in FACE_NODES_MAP: # Ця перевірка вже є
          print(f"     ПОМИЛКА: face_id {face_id} не знайдено в FACE_NODES_MAP!")
          return np.zeros(60)
    
    if face_id not in FACE_NODES_MAP:
        print(f"ПОПЕРЕДЖЕННЯ: Не визначено карту вузлів для грані {face_id}. Навантаження не застосовано.")
        return np.zeros(60)

    face_data = FACE_NODES_MAP[face_id]
    # Перетворюємо на numpy масиви перед передачею до numba-сумісної функції
    face_3d_node_indices_for_AKT = np.array(face_data["nodes_3d_indices"], dtype=np.int32)
    normal_vector_direction = np.array(face_data["normal_vector_direction"], dtype=np.float64)

    print(f"     Дані для грані {face_id} з FACE_NODES_MAP: {FACE_NODES_MAP[face_id]}")
    print(f"     face_3d_node_indices_for_AKT: {face_3d_node_indices_for_AKT}") # Вивести після перетворення на np.array
    print(f"     normal_vector_direction: {normal_vector_direction}")
    
    # Переконуємося, що інші аргументи також є numpy масивами потрібного типу
    # AKT_elem_nodes вже має бути np.array
    # LOCAL_NODE_COORDS_2D_FACE, ETA_G_SURF_FLAT, etc. визначені глобально як np.array

    return compute_element_load_vector_numba_compatible(
        pressure_value,
        face_3d_node_indices_for_AKT,
        normal_vector_direction,
        AKT_elem_nodes,
        LOCAL_NODE_COORDS_2D_FACE,
        ETA_G_SURF_FLAT,
        TAU_G_SURF_FLAT,
        GAUSS_WEIGHTS_SURF_FLAT
    )

def assemble_global_load(F, FE_elem, elem_nodes_global_0based, nqp):
    full_size = 3 * nqp
    for r_loc_node_idx in range(20):
        g_r_0based_node_idx = elem_nodes_global_0based[r_loc_node_idx]
        if g_r_0based_node_idx < 0:
            continue
        if g_r_0based_node_idx >= nqp:
            print(f"ПОПЕРЕДЖЕННЯ (assemble_global_load): Індекс глобального вузла {g_r_0based_node_idx} виходить за межі ({nqp}). Пропуск.")
            continue
        for dof_local_idx in range(3):
            idx_f_global = 3 * g_r_0based_node_idx + dof_local_idx
            fe_elem_idx = r_loc_node_idx + 20 * dof_local_idx
            if 0 <= idx_f_global < full_size and 0 <= fe_elem_idx < 60:
                 F[idx_f_global] += FE_elem[fe_elem_idx]
            # else: # Ця умова тепер менш імовірна через попередні перевірки
            #     print(f"ПОПЕРЕДЖЕННЯ (assemble_global_load): Некоректні індекси при збиранні F.")

def apply_boundary_conditions(MG, F, ZU_1based, nqp):
     print(" Застосування ГУ (метод великих діагоналей для повної матриці)...")
     F_mod = F.copy()
     MG_mod = MG.copy()
     full_size = 3 * nqp

     if ZU_1based is not None and ZU_1based.size > 0:
          fixed_dofs = []
          for node_idx_1based in ZU_1based:
               node_idx_0based = node_idx_1based - 1
               if 0 <= node_idx_0based < nqp:
                    for dof in range(3):
                        fixed_dofs.append(3 * node_idx_0based + dof)
               else:
                    print(f"ПОПЕРЕДЖЕННЯ (apply_boundary_conditions): Некоректний номер вузла {node_idx_1based} в ZU (макс {nqp}). Пропущено.")

          if fixed_dofs:
                diag_elements = np.abs(np.diag(MG_mod))
                valid_diag_elements = diag_elements[diag_elements > 1e-12]
                penalty_factor = 1e10
                if valid_diag_elements.size > 0:
                    penalty = valid_diag_elements.max() * penalty_factor
                else:
                    penalty = penalty_factor
                penalty = max(penalty, 1e7)
                print(f" Значення Penalty для ГУ: {penalty:.2e}")

                for idx_dof_global in fixed_dofs:
                     if 0 <= idx_dof_global < full_size:
                         for k_row in range(full_size):
                             if k_row != idx_dof_global:
                                 F_mod[k_row] -= MG_mod[k_row, idx_dof_global] * 0.0 # U_prescribed = 0
                         MG_mod[idx_dof_global, :] = 0.0
                         MG_mod[:, idx_dof_global] = 0.0
                         MG_mod[idx_dof_global, idx_dof_global] = penalty
                         F_mod[idx_dof_global] = 0.0 * penalty
                     else:
                         print(f"ПОПЕРЕДЖЕННЯ (apply_boundary_conditions): Індекс DOF {idx_dof_global} виходить за межі ({full_size}).")
          else:
               print("ПОПЕРЕДЖЕННЯ: Не знайдено валідних ступенів вільності для застосування з ZU.")
     else:
        print(" ZU не задано або порожній.")
     return MG_mod, F_mod