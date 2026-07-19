import numpy as np

from tcsm_rt.sampling import sample_indices, sample_scene_indices


def test_sampling_modes_are_unique_and_deterministic():
    x, y = np.meshgrid(np.arange(7), np.arange(7), indexing="xy")
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    for mode in ("scatter", "trajectory", "coverage_trajectory"):
        first = sample_indices(points, 8, mode, 17)
        second = sample_indices(points, 8, mode, 17)
        assert len(first) == len(np.unique(first)) == 8
        np.testing.assert_array_equal(first, second)


def test_scene_sampling_excludes_invalid_building_cells():
    x, y = np.meshgrid(np.arange(7), np.arange(7), indexing="xy")
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    valid = np.ones(len(points), dtype=bool)
    valid[[0, 1, 7, 8, 24]] = False
    arrays = {"query_xyz_m": points, "valid_query_mask": valid}
    for mode in ("scatter", "trajectory", "coverage_trajectory"):
        selected = sample_scene_indices(arrays, 12, mode, 17)
        assert np.all(valid[selected])
