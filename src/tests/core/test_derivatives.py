# tests/core/test_derivatives.py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fem_app.core import mesh, derivatives

TOL = 1e-9 # Допуск

# --- ФІКСТУРИ ---
@pytest.fixture(scope="module")
def dfiabg_data():
    print("\n(Fixture) Обчислення DFIABG...")
    a, b, g = derivatives.alpha_flat_g, derivatives.beta_flat_g, derivatives.gamma_flat_g
    dfiabg = derivatives.compute_DFIABG(a, b, g)
    print("(Fixture) DFIABG обчислено.")
    return dfiabg

@pytest.fixture(scope="module")
def manual_mesh_example_data():
    print("\n(Fixture) Генерація сітки з методички...")
    nx, ny, nz = 2, 1, 2
    ax, ay, az = 2.0, 1.0, 2.0
    AKT, node_map, nqp = mesh.generate_mesh(nx, ny, nz, ax, ay, az)
    NT, nel = mesh.generate_connectivity(nx, ny, nz, node_map)
    assert nqp == 51 and nel == 4
    print("(Fixture) Сітку з методички згенеровано.")
    return {"AKT": AKT, "NT": NT, "nqp": nqp, "nel": nel}

@pytest.fixture(scope="module")
def scaled_cube_element_data():
     print("\n(Fixture) Генерація даних для елемента-куба...")
     L = 2.0
     # --- ВИКОРИСТОВУЄМО НОВІ КООРДИНАТИ ---
     coords_local = np.array([
         [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], # 1-4
         [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1],   # 5-8
         [0, 1, -1], [1, 0, -1], [0, -1, -1], [-1, 0, -1], # 9-12
         [-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0],   # 13-16
         [0, 1, 1], [1, 0, 1], [0, -1, 1], [-1, 0, 1]    # 17-20
     ], dtype=np.float64)
     # --- AKT_test тепер буде мати інвертований Y ---
     AKT_test = (L / 2.0) * coords_local.T
     NT_test = np.arange(1, 21, dtype=np.int32).reshape((20, 1))
     print("(Fixture) Дані для куба згенеровано.")
     return {"AKT": AKT_test, "NT": NT_test, "L": L}

# --- ТЕСТИ ДЛЯ DFIABG ---
def test_dfiabg_shape(dfiabg_data):
    assert dfiabg_data.shape == (27, 3, 20)

def test_phi_partition_of_unity(dfiabg_data):
    a, b, g = derivatives.alpha_flat_g, derivatives.beta_flat_g, derivatives.gamma_flat_g
    phi_values = derivatives.compute_PHI(a, b, g)
    assert phi_values.shape == (27, 20)
    sum_phi = np.sum(phi_values, axis=-1)
    assert_allclose(sum_phi, 1.0, atol=TOL, err_msg="Сума функцій форми Phi_i не дорівнює 1")

def test_dfiabg_center_point_node1(dfiabg_data):
    # --- Правильно, проходить ---
    expected = np.array([0.125, -0.125, 0.125])
    actual = dfiabg_data[13, :, 0]
    assert_allclose(actual, expected, atol=TOL)

def test_dfiabg_center_point_node9(dfiabg_data):
    # --- Правильно, проходить ---
    expected = np.array([0.0, 0.25, -0.25])
    actual = dfiabg_data[13, :, 8]
    assert_allclose(actual, expected, atol=TOL)

# --- ТЕСТ ДЛЯ ЯКОБІАНА ---
@pytest.mark.parametrize("gauss_point_index", [0, 13, 26])
def test_jacobian_scaled_cube(scaled_cube_element_data, dfiabg_data, gauss_point_index):
    """Тест: Перевірка Якобіана для ідеального куба."""
    AKT_test = scaled_cube_element_data["AKT"]
    NT_test = scaled_cube_element_data["NT"]
    L = scaled_cube_element_data["L"]
    J_calculated = derivatives.compute_jacobian(AKT_test, NT_test, 1, dfiabg_data, gauss_point_index)

    J_expected = np.diag([L/2.0] * 3)
    assert_allclose(J_calculated, J_expected, atol=TOL, err_msg=f"Неправильний Якобіан (GP={gauss_point_index})")
    det_J_calculated = np.linalg.det(J_calculated)

    det_J_expected = (L/2.0)**3
    assert_allclose(det_J_calculated, det_J_expected, atol=TOL, err_msg=f"Неправильний det(J) (GP={gauss_point_index})")

# --- ТЕСТИ ДЛЯ DFIXYZ ---
def test_dfixyz_sum_property(manual_mesh_example_data, dfiabg_data):
    """Тест: Перевірка суми глобальних похідних DFIXYZ ~ 0."""
    AKT = manual_mesh_example_data["AKT"]
    NT = manual_mesh_example_data["NT"]
    target_elem_id = 1
    print(f"\nТестування DFIXYZ (Sum Property) для елемента {target_elem_id}...")
    dfixyz_elem, dj_dets_elem, dets_ok = derivatives.compute_DFIXYZ_for_element(
        target_elem_id, AKT, NT, dfiabg_data
    )

    assert dets_ok, f"Якобіан для елемента {target_elem_id} мав проблеми."
    assert not np.isnan(dfixyz_elem).any(), f"DFIXYZ для елемента {target_elem_id} містить NaN."
    assert not np.isnan(dj_dets_elem).any(), f"DJ_dets для елемента {target_elem_id} містить NaN."
    sum_dfixyz = np.sum(dfixyz_elem, axis=1)
    assert_allclose(sum_dfixyz, 0, atol=TOL, err_msg=f"Сума dPhi_i/dX для елемента {target_elem_id} не = 0")

def test_dfixyz_scaled_cube(scaled_cube_element_data, dfiabg_data):
    """Тест: Перевірка DFIXYZ для ідеального куба."""
    AKT_test = scaled_cube_element_data["AKT"]
    NT_test = scaled_cube_element_data["NT"]
    L = scaled_cube_element_data["L"]
    target_elem_id = 1
    inv_scale = 2.0 / L
    print(f"\nТестування DFIXYZ (Scaled Cube) для елемента {target_elem_id} (L={L})...")
    dfixyz_elem, dj_dets_elem, dets_ok = derivatives.compute_DFIXYZ_for_element(
        target_elem_id, AKT_test, NT_test, dfiabg_data
    )

    assert dets_ok, f"Якобіан для куба мав проблеми."
    assert not np.isnan(dfixyz_elem).any(), f"DFIXYZ для куба містить NaN."
    assert not np.isnan(dj_dets_elem).any(), f"DJ_dets для куба містить NaN."

    # Перевіряємо зв'язок з DFIABG
    for j in range(27):
        for i in range(20):
            for k in range(3): # 0=x, 1=y, 2=z
                expected_val = dfiabg_data[j, k, i] * inv_scale
                actual_val = dfixyz_elem[j, i, k]
                assert_allclose(actual_val, expected_val, atol=TOL,
                                err_msg=f"DFIXYZ[{j},{i},{k}] != DFIABG[{j},{k},{i}]*{inv_scale:.2f}")