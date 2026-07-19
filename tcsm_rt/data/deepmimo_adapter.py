from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..physics import LIGHT_SPEED_M_S, centered_ula_positions, make_task_labels, reconstruct_spherical_channel
from ..provenance import sha256_file, write_json_atomic
from ..schema import save_scene
from .common import spatial_split_ids


def _datasets(dataset: Any) -> Iterable[Any]:
    # DeepMIMO Dataset.__getattr__ raises KeyError for unknown matrices, so
    # getattr(..., default) is not safe for distinguishing Dataset/MacroDataset.
    children = vars(dataset).get("datasets")
    if children is None:
        children = vars(dataset).get("_datasets")
    if children is None:
        yield dataset
    else:
        yield from children


def _path_coefficients(dataset: Any) -> np.ndarray:
    power_db = np.asarray(dataset.power, dtype=np.float64)
    phase_deg = np.asarray(dataset.phase, dtype=np.float64)
    amplitude = np.power(10.0, power_db / 20.0)
    coefficients = amplitude * np.exp(1j * np.deg2rad(phase_deg))
    return np.nan_to_num(coefficients, nan=0.0).astype(np.complex64)


def _interaction_counts(dataset: Any) -> np.ndarray:
    positions = np.asarray(dataset.inter_pos, dtype=np.float64)
    return np.sum(np.all(np.isfinite(positions), axis=-1), axis=-1).astype(np.int16)


def _valid_receiver_mask(dataset: Any) -> np.ndarray:
    power_db = np.asarray(dataset.power, dtype=np.float64)
    return np.any(np.isfinite(power_db), axis=1)


def _los_from_paths(dataset: Any) -> np.ndarray:
    power_db = np.asarray(dataset.power, dtype=np.float64)
    interaction_codes = np.asarray(dataset.inter, dtype=np.float64)
    valid_path = np.isfinite(power_db) & np.isfinite(interaction_codes)
    return np.any(valid_path & (interaction_codes == 0.0), axis=1).astype(np.float32)


def _environment(
    rx: np.ndarray,
    tx: np.ndarray,
    frequency_hz: float,
    los: np.ndarray,
    interaction_counts: np.ndarray,
    labels: dict[str, np.ndarray],
) -> np.ndarray:
    rx = np.asarray(rx, dtype=np.float32)
    tx = np.asarray(tx, dtype=np.float32).reshape(3)
    relative = rx - tx[None, :]
    los = np.asarray(los, dtype=np.float32).reshape(-1)
    wavelength = LIGHT_SPEED_M_S / float(frequency_hz)
    distance = np.linalg.norm(relative, axis=1)
    free_space_db = 20.0 * np.log10(np.maximum(4.0 * np.pi * distance / wavelength, 1e-12))
    excess_loss = np.maximum(-labels["rss_db"] - free_space_db, 0.0)
    interaction_density = np.asarray(interaction_counts, dtype=np.float32).mean(axis=1)
    unavailable_building = np.zeros(len(rx), dtype=np.float32)
    modality_mask = np.zeros(len(rx), dtype=np.float32)
    return np.column_stack(
        [
            unavailable_building,
            unavailable_building,
            relative[:, 0],
            relative[:, 1],
            relative[:, 2],
            los,
            1.0 - los,
            excess_loss,
            interaction_density,
            modality_mask,
        ]
    ).astype(np.float32)


def generate_deepmimo_scenario(
    scenario: str,
    config: dict[str, Any],
    raw_dir: str | Path,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    import deepmimo as dm  # type: ignore

    settings = config["data"]["deepmimo"]
    raw_path = Path(raw_dir).resolve()
    raw_path.mkdir(parents=True, exist_ok=True)
    dm.config.set("scenarios_folder", str(raw_path))
    scenario_folder = Path(dm.get_scenario_folder(scenario)).resolve()
    if not scenario_folder.exists():
        dm.download(scenario, output_dir=str(raw_path))
        scenario_folder = Path(dm.get_scenario_folder(scenario)).resolve()
    if not scenario_folder.exists():
        raise FileNotFoundError(f"DeepMIMO scenario was not materialized: {scenario_folder}")
    loaded = dm.load(
        str(scenario_folder),
        tx_sets="all",
        rx_sets="rx_only",
        matrices=["power", "phase", "delay", "inter", "inter_pos", "rx_pos", "tx_pos"],
        max_paths=int(settings["max_paths"]),
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(_datasets(loaded)):
        frequency_hz = float(dataset.rt_params.frequency)
        valid_receiver = _valid_receiver_mask(dataset)
        total_receiver_count = int(valid_receiver.size)
        if not np.any(valid_receiver):
            continue
        coefficients = _path_coefficients(dataset)[valid_receiver]
        vertices = np.asarray(dataset.inter_pos, dtype=np.float64)[valid_receiver]
        counts = _interaction_counts(dataset)[valid_receiver]
        los = _los_from_paths(dataset)[valid_receiver]
        rx = np.asarray(dataset.rx_pos, dtype=np.float32)[valid_receiver]
        tx = np.asarray(dataset.tx_pos, dtype=np.float64).reshape(-1, 3)[0]
        array_size = 128
        offsets = centered_ula_positions(array_size, LIGHT_SPEED_M_S / frequency_hz)
        channels = np.empty((len(rx), array_size), dtype=np.complex64)
        for index in range(len(rx)):
            channels[index] = reconstruct_spherical_channel(
                tx,
                offsets,
                rx[index],
                coefficients[index],
                vertices[index],
                counts[index],
                frequency_hz,
            )
        labels = make_task_labels(
            channels,
            frequency_hz,
            int(config["model"]["far_beams"]),
            int(config["model"]["near_angles"]),
            int(config["model"]["near_ranges"]),
            distance_m=np.linalg.norm(rx - tx[None, :], axis=1),
            tx_power_dbm=float(config["system"]["tx_power_dbm"]),
            bandwidth_hz=float(config["system"]["bandwidth_hz"]),
            noise_figure_db=float(config["system"]["noise_figure_db"]),
            low_margin_bps_hz=float(config["system"]["regime_low_margin_bps_hz"]),
            high_margin_bps_hz=float(config["system"]["regime_high_margin_bps_hz"]),
        )
        environment = _environment(rx, tx, frequency_hz, los, counts, labels)
        cache_path = output_path / f"deepmimo_{scenario.lower()}_tx{dataset_index:02d}.npz"
        arrays = {
            "query_xyz_m": rx,
            "environment": environment,
            "valid_query_mask": np.ones(len(rx), dtype=bool),
            "task_availability": np.tile(
                np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                (len(rx), 1),
            ),
            "channel": channels,
            "spatial_split": spatial_split_ids(rx[:, :2]),
            **{key: value for key, value in labels.items() if key not in {"near_ranges_m", "near_angles_rad"}},
            "near_ranges_axis_m": labels["near_ranges_m"],
            "near_angles_axis_rad": labels["near_angles_rad"],
        }
        save_scene(cache_path, arrays)
        row = {
            "source": "deepmimo_v4",
            "scenario": scenario,
            "dataset_index": dataset_index,
            "query_count": len(rx),
            "raw_receiver_count": total_receiver_count,
            "discarded_no_path_count": total_receiver_count - len(rx),
            "frequency_hz": frequency_hz,
            "array_size": array_size,
            "external_task_scope": ["rss", "far_beam"],
            "near_field_evidence": "unsupported_by_standard_synthetic_array_dataset",
            "channel_construction": "center-ray spherical phase diagnostic",
            "cache": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
        }
        write_json_atomic(cache_path.with_suffix(".json"), row)
        metadata_rows.append(row)
    return metadata_rows
