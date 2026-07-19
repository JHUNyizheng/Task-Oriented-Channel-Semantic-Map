import numpy as np
import torch

from tcsm_rt.grid_learning import build_grid_batch, decode_grid_output, new_grid_model
from tcsm_rt.physics import make_task_labels


def _scene() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(9)
    coordinate = np.arange(7, dtype=np.float32)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size, dtype=np.float32)])
    channel = (rng.normal(size=(49, 16)) + 1j * rng.normal(size=(49, 16))).astype(np.complex64) * 1e-5
    labels = make_task_labels(channel, 28e9, 17, 17, 5, distance_m=np.linspace(5.0, 200.0, 49))
    return {
        "query_xyz_m": points,
        "environment": rng.normal(size=(49, 10)).astype(np.float32),
        "channel": channel,
        **{key: value for key, value in labels.items() if key not in {"near_ranges_m", "near_angles_rad"}},
    }


def test_grid_model_decodes_all_task_heads():
    config = {"model": {"hidden": 32, "far_beams": 17, "near_angles": 17, "near_ranges": 5}}
    batch = build_grid_batch(_scene(), np.array([0, 8, 16, 24]), config["model"], torch.device("cpu"))
    model = new_grid_model("radiounet", batch.inputs.shape[1], config)
    prediction = decode_grid_output(model(batch.inputs), config["model"])
    assert prediction["rss"].shape == (1, 49)
    assert prediction["regime"].shape == (1, 49, 3)
    assert prediction["far"].shape == (1, 49, 17)
    assert prediction["near_range"].shape == (1, 49, 5)


def test_grid_loss_mask_excludes_support_and_invalid_cells():
    arrays = _scene()
    arrays["valid_query_mask"] = np.ones(49, dtype=bool)
    arrays["valid_query_mask"][[0, 1, 2]] = False
    support = np.array([8, 16, 24])
    config = {"hidden": 32, "far_beams": 17, "near_angles": 17, "near_ranges": 5}
    batch = build_grid_batch(arrays, support, config, torch.device("cpu"))
    mask = batch.query_mask[0].numpy()
    assert not np.any(mask[[0, 1, 2, 8, 16, 24]])
    assert int(mask.sum()) == 43
