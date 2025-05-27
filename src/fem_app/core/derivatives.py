import numpy as np
from numba import njit

gp = np.sqrt(0.6)
g_pts_1d = np.array([-gp, 0.0, gp])
alpha_g, beta_g, gamma_g = np.meshgrid(g_pts_1d, g_pts_1d, g_pts_1d, indexing='ij')
alpha_flat_g = alpha_g.flatten()
beta_flat_g = beta_g.flatten()
gamma_flat_g = gamma_g.flatten()


@njit
def compute_DFIABG(alpha_flat, beta_flat, gamma_flat):
    """
    Обчислює масив DFIABG(27, 3, 20)
    """
    coords = np.array([
            [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], 
            [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1],
            [0, 1, -1], [1, 0, -1], [0, -1, -1], [-1, 0, -1],
            [-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [0, 1, 1], [1, 0, 1], [0, -1, 1], [-1, 0, 1]
        ], dtype=np.float64)
    a_i, b_i, g_i = coords[:, 0], coords[:, 1], coords[:, 2]
    DFIABG = np.zeros((27, 3, 20))

    for j in range(27):
        a, b, g = alpha_flat[j], beta_flat[j], gamma_flat[j]
        for i in range(20):
            ai, bi, gi = a_i[i], b_i[i], g_i[i]
            term1 = (1.0 + a * ai); term2 = (1.0 + b * bi); term3 = (1.0 + g * gi)

            if i < 8:
                dphi_da = 0.125 * ai * term2 * term3 * (2 * a * ai + b * bi + g * gi - 1.0)
                dphi_db = 0.125 * bi * term1 * term3 * (a * ai + 2 * b * bi + g * gi - 1.0)
                dphi_dg = 0.125 * gi * term1 * term2 * (a * ai + b * bi + 2 * g * gi - 1.0)
            else:
                ai2, bi2, gi2 = ai*ai, bi*bi, gi*gi
                term_s2 = (1.0 - (a*bi*gi)**2 - (b*ai*gi)**2 - (g*ai*bi)**2)
                ds2_da = -2.0 * a * bi2 * gi2
                ds2_db = -2.0 * b * ai2 * gi2
                ds2_dg = -2.0 * g * ai2 * bi2

                dphi_da = 0.25 * term2 * term3 * (ai * term_s2 + term1 * ds2_da)
                dphi_db = 0.25 * term1 * term3 * (bi * term_s2 + term2 * ds2_db)
                dphi_dg = 0.25 * term1 * term2 * (gi * term_s2 + term3 * ds2_dg)

            DFIABG[j, 0, i] = dphi_da
            DFIABG[j, 1, i] = dphi_db
            DFIABG[j, 2, i] = dphi_dg

    return DFIABG

@njit
def compute_PHI(alpha_flat, beta_flat, gamma_flat):
    """
    Обчислює значення функцій форми PHI(27, 20) у точках Гаусса.
    """
    coords = np.array([
        [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1], # 1-4
        [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1], # 5-8
        [ 0,-1,-1],[ 1, 0,-1],[ 0, 1,-1],[-1, 0,-1], # 9-12
        [-1,-1, 0],[ 1,-1, 0],[ 1, 1, 0],[-1, 1, 0], # 13-16
        [ 0,-1, 1],[ 1, 0, 1],[ 0, 1, 1],[-1, 0, 1]  # 17-20
    ], dtype=np.float64)
    a_i, b_i, g_i = coords[:, 0], coords[:, 1], coords[:, 2]
    PHI = np.zeros((27, 20)) # Масив для значень функцій форми

    for j in range(27): # Цикл по точках Гаусса
        a, b, g = alpha_flat[j], beta_flat[j], gamma_flat[j]
        for i in range(20): # Цикл по вузлах / функціях форми
            ai, bi, gi = a_i[i], b_i[i], g_i[i]
            val = 0.0 # Значення поточної функції форми phi_i

            if i < 8: # Кутові вузли
                term1 = (1 + a * ai); term2 = (1 + b * bi); term3 = (1 + g * gi)
                sum_abc = a * ai + b * bi + g * gi
                val = 0.125 * term1 * term2 * term3 * (sum_abc - 2.0)
            else: # Вузли на ребрах
                if abs(ai) < 1e-9:   # ai = 0
                    val = 0.25 * (1.0 - a*a) * (1.0 + b*bi) * (1.0 + g*gi)
                elif abs(bi) < 1e-9: # bi = 0
                    val = 0.25 * (1.0 + a*ai) * (1.0 - b*b) * (1.0 + g*gi)
                else: # gi = 0
                    val = 0.25 * (1.0 + a*ai) * (1.0 + b*bi) * (1.0 - g*g)

            PHI[j, i] = val

    return PHI

@njit
def compute_jacobian(AKT, NT, elem_id, DFIABG, j_gauss):
    """Обчислює матрицю Якобі J(3,3)."""
    J = np.zeros((3, 3))
    if elem_id <= 0 or elem_id > NT.shape[1]: return J 
    elem_nodes = NT[:, elem_id - 1] - 1
    for i in range(20):
        g_idx = elem_nodes[i]
        if g_idx < 0 or g_idx >= AKT.shape[1]: continue
        node_coords = AKT[:, g_idx]
        dphi_dlocal = DFIABG[j_gauss, :, i]
        for m in range(3):
            for n in range(3): J[m, n] += dphi_dlocal[n] * node_coords[m]
    return J


def compute_DFIXYZ_for_element(elem_id, AKT, NT, DFIABG):
    """
    Обчислює DFIXYZ(27, 20, 3) та масив детермінантів Якобіана DJ_dets(27).
    Повертає: DFIXYZ, DJ_dets, all_dets_positive (bool)
    """
    DFIXYZ = np.zeros((27, 20, 3))
    DJ_dets = np.zeros(27)
    all_dets_positive = True

    for j in range(27): # Цикл по точках Гаусса
        J = compute_jacobian(AKT, NT, elem_id, DFIABG, j)
        det_J = np.linalg.det(J)
        DJ_dets[j] = det_J

        if abs(det_J) < 1e-9:
            all_dets_positive = False
            DFIXYZ[j, :, :] = np.nan
            continue
        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
             all_dets_positive = False
             DFIXYZ[j, :, :] = np.nan
             DJ_dets[j] = 0.0
             continue

        for i in range(20):
            dphi_local = DFIABG[j, :, i]
            dphi_global = np.dot(J_inv, dphi_local)
            DFIXYZ[j, i, :] = dphi_global

    return DFIXYZ, DJ_dets, all_dets_positive
    