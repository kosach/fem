# tests/core/test_mesh.py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# Імпортуємо функції, які тестуємо
from fem_app.core import mesh

# --- Тестовий клас для конкретного прикладу з методички ---
class TestMeshGenerationManualExample:

    # Визначаємо параметри тесту
    nx, ny, nz = 2, 1, 2
    ax, ay, az = 2.0, 1.0, 2.0 # Розміри для перевірки координат

    @pytest.fixture(scope="class")
    def generated_mesh_data(self):
        """Генерує дані сітки один раз для всіх тестів цього класу."""
        print(f"\n(Fixture) Генерація сітки для тесту: nx={self.nx}, ny={self.ny}, nz={self.nz}")
        AKT, node_map, nqp = mesh.generate_mesh(self.nx, self.ny, self.nz, self.ax, self.ay, self.az)
        NT, nel = mesh.generate_connectivity(self.nx, self.ny, self.nz, node_map)
        print(f"(Fixture) Згенеровано: nqp={nqp}, nel={nel}")
        # Перевіряємо базові очікування тут
        assert nqp == 51, "Неправильна кількість вузлів (nqp)"
        assert nel == 4, "Неправильна кількість елементів (nel)"
        assert AKT.shape == (3, 51), "Неправильна розмірність AKT"
        assert NT.shape == (20, 4), "Неправильна розмірність NT"
        return {"AKT": AKT, "NT": NT, "node_map": node_map, "nqp": nqp, "nel": nel}

    def test_node_count(self, generated_mesh_data):
        """Тест: Перевірка загальної кількості вузлів."""
        assert generated_mesh_data["nqp"] == 51

    def test_element_count(self, generated_mesh_data):
        """Тест: Перевірка загальної кількості елементів."""
        assert generated_mesh_data["nel"] == 4

    def test_akt_shape(self, generated_mesh_data):
        """Тест: Перевірка розмірності масиву координат AKT."""
        assert generated_mesh_data["AKT"].shape == (3, 51)

    def test_nt_shape(self, generated_mesh_data):
        """Тест: Перевірка розмірності масиву зв'язності NT."""
        assert generated_mesh_data["NT"].shape == (20, 4)

    # --- Тести для перевірки конкретних координат вузлів ---
    # !!! Переконайся, що цей декоратор АКТИВНИЙ (не закоментований) !!!
    @pytest.mark.parametrize("ix, iy, iz, expected_coords", [
        (0, 0, 0, [0.0, 0.0, 0.0]),  # Вузол 1
        (4, 0, 0, [2.0, 0.0, 0.0]),  # Вузол 5
        (0, 0, 4, [0.0, 0.0, 2.0]),  # Вузол 19
        (4, 0, 4, [2.0, 0.0, 2.0]),  # Вузол 38
        (4, 2, 4, [2.0, 1.0, 2.0]),  # Вузол 51
        (1, 0, 2, [0.5, 0.0, 1.0]),  # Вузол 21
        (2, 1, 2, [1.0, 0.5, 1.0]),  # Вузол 26 (середина ребра на y=max?) Ні, (ix=2, iy=1, iz=2)
        (2, 0, 1, [1.0, 0.0, 0.5]),  # Вузол 15
    ])
    def test_node_coordinates(self, generated_mesh_data, ix, iy, iz, expected_coords):
        """Тест: Перевірка координат конкретних вузлів."""
        node_map = generated_mesh_data["node_map"]
        AKT = generated_mesh_data["AKT"]
        node_index_0based = node_map.get((ix, iy, iz))
        assert node_index_0based is not None, f"Вузол з індексами ({ix},{iy},{iz}) не знайдено в node_map"
        assert_allclose(AKT[:, node_index_0based], expected_coords, atol=1e-9,
                        err_msg=f"Неправильні координати для вузла ({ix},{iy},{iz}) -> Глоб. індекс {node_index_0based+1}")

    # --- Тести для перевірки конкретних стовпців NT ---
    def test_nt_element_1(self, generated_mesh_data):
        """Тест: Перевірка списку вузлів для елемента 1."""
        NT = generated_mesh_data["NT"]
        expected_nodes_elem1 = np.array([1, 3, 11, 9, 20, 22, 30, 28, 2, 7, 10, 6, 14, 15, 18, 17, 21, 26, 29, 25], dtype=np.int32)
        assert_array_equal(NT[:, 0], expected_nodes_elem1, err_msg="Масив вузлів для елемента 1 не відповідає методичці")

    def test_nt_element_2(self, generated_mesh_data):
        """Тест: Перевірка списку вузлів для елемента 2."""
        NT = generated_mesh_data["NT"]
        expected_nodes_elem2 = np.array([3, 5, 13, 11, 22, 24, 32, 30, 4, 8, 12, 7, 15, 16, 19, 18, 23, 27, 31, 26], dtype=np.int32)
        assert_array_equal(NT[:, 1], expected_nodes_elem2, err_msg="Масив вузлів для елемента 2 не відповідає методичці")