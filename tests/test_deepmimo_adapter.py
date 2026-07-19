from types import SimpleNamespace

import numpy as np

from tcsm_rt.data.deepmimo_adapter import (
    _datasets,
    _interaction_counts,
    _los_from_paths,
    _path_coefficients,
    _valid_receiver_mask,
)


class MatrixDataset:
    def __init__(self):
        self._data = {}
        self.power = np.array([[-80.0, np.nan], [np.nan, np.nan], [-90.0, -95.0]])
        self.phase = np.array([[0.0, np.nan], [np.nan, np.nan], [90.0, -90.0]])
        self.inter = np.array([[0.0, np.nan], [np.nan, np.nan], [1.0, 2.0]])
        self.inter_pos = np.full((3, 2, 2, 3), np.nan)
        self.inter_pos[2, 0, 0] = [1.0, 2.0, 3.0]
        self.inter_pos[2, 1, :2] = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

    def __getattr__(self, name):
        raise KeyError(name)


def test_dataset_detection_does_not_trigger_deepmimo_matrix_lookup():
    dataset = MatrixDataset()
    assert list(_datasets(dataset)) == [dataset]
    macro = SimpleNamespace(datasets=[dataset, dataset])
    assert list(_datasets(macro)) == [dataset, dataset]


def test_receiver_filter_los_and_interaction_count_follow_ray_data():
    dataset = MatrixDataset()
    np.testing.assert_array_equal(_valid_receiver_mask(dataset), [True, False, True])
    np.testing.assert_array_equal(_los_from_paths(dataset), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(_interaction_counts(dataset), [[0, 0], [0, 0], [1, 2]])


def test_deepmimo_power_and_phase_form_complex_path_amplitude():
    coefficients = _path_coefficients(MatrixDataset())
    np.testing.assert_allclose(coefficients[0, 0], 1e-4 + 0j, rtol=1e-6)
    np.testing.assert_allclose(coefficients[2, 0], 1j * 10 ** (-90.0 / 20.0), rtol=1e-6)
    assert coefficients[1, 0] == 0j
