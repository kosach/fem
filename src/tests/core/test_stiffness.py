# tests/core/test_stiffness.py
import numpy as np
import pytest
from numpy.testing import assert_allclose

# Імпортуємо потрібні модулі та функції
from fem_app.core import mesh
from fem_app.core import derivatives
from fem_app.core import stiffness
from fem_app.core.stiffness import gauss_weights_3D # Імпортуємо ваги

TOL = 1e-9 # Допуск

# --- Фікстури ---
@pytest.fixture(scope="module")
def dfiabg_data():
    a, b, g = derivatives.alpha_flat_g, derivatives.beta_flat_g, derivatives.gamma_flat_g
    return derivatives.compute_DFIABG(a, b, g)

@pytest.fixture(scope="module")
def scaled_cube_element_data():
    L = 2.0
    coords_local = np.array([
        [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
        [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1],
        [ 0,-1,-1],[ 1, 0,-1],[ 0, 1,-1],[-1, 0,-1],
        [-1,-1, 0],[ 1,-1, 0],[ 1, 1, 0],[-1, 1, 0],
        [ 0,-1, 1],[ 1, 0, 1],[ 0, 1, 1],[-1, 0, 1]
    ], dtype=np.float64)
    AKT_test = (L / 2.0) * coords_local.T
    NT_test = np.arange(1, 21, dtype=np.int32).reshape((20, 1))
    return {"AKT": AKT_test, "NT": NT_test, "L": L}

# --- Тест на симетрію MGE ---
def test_mge_symmetry(scaled_cube_element_data, dfiabg_data):
    """Тест: Перевіряє симетрію MGE."""
    print("\nТестування симетрії MGE...")
    AKT_test = scaled_cube_element_data["AKT"]
    NT_test = scaled_cube_element_data["NT"]
    elem_id = 1
    E_test = 2.1e5
    nu_test = 0.3

    DFIXYZ_elem, DJ_dets_elem, dets_ok = derivatives.compute_DFIXYZ_for_element(
        elem_id, AKT_test, NT_test, dfiabg_data
    )
    assert dets_ok and not np.isnan(DFIXYZ_elem).any() and not np.isnan(DJ_dets_elem).any()

    lambda_, mu_ = stiffness.calculate_lambda_mu(E_test, nu_test)
    assert not (np.isnan(lambda_) or np.isnan(mu_))

    gauss_weights = gauss_weights_3D # Використовуємо імпортовану змінну

    MGE_calculated = stiffness.compute_element_stiffness_MGE(
        DFIXYZ_elem, DJ_dets_elem, lambda_, mu_, gauss_weights
    )

    print(f" Перевірка симетрії MGE shape: {MGE_calculated.shape}...")
    assert_allclose(MGE_calculated, MGE_calculated.T, atol=TOL,
                    err_msg="Обчислена матриця MGE не є симетричною!")
    print(" Тест симетрії MGE пройдено успішно.")