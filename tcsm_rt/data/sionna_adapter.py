from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..physics import (
    LIGHT_SPEED_M_S,
    centered_ula_positions,
    channel_correlation,
    make_task_labels,
    reconstruct_spherical_channel,
)
from ..provenance import sha256_file, write_json_atomic
from ..schema import save_scene
from ..sionna_backend import configure_mitsuba_variant
from .common import RTConfiguration, grid_xyz


configure_mitsuba_variant()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scene_constant(name: str) -> str:
    import sionna.rt  # type: ignore

    try:
        return getattr(sionna.rt.scene, name)
    except AttributeError as error:
        raise ValueError(f"unknown Sionna scene: {name}") from error


def _configure_itu_material_frequency(
    scene: Any,
    requested_frequency_hz: float,
    policy: str,
) -> dict[str, Any]:
    """Freeze Sionna ITU materials at auditable in-range frequencies."""
    from sionna.rt.radio_materials.itu import ITU_MATERIALS_PROPERTIES  # type: ignore

    if policy not in {"strict", "clamp_to_itu_range"}:
        raise ValueError(f"unknown ITU material frequency policy: {policy}")
    requested_frequency_ghz = float(requested_frequency_hz) / 1e9
    material_records: list[dict[str, Any]] = []
    for name, material in sorted(scene.radio_materials.items()):
        itu_type = getattr(material, "itu_type", None)
        # Some Sionna/Mitsuba builds reconstruct an ITU XML plugin as a
        # generic radio material while retaining the canonical ITU name.
        if itu_type is None and str(name) in ITU_MATERIALS_PROPERTIES:
            itu_type = str(name)
        if itu_type is None:
            continue
        intervals = ITU_MATERIALS_PROPERTIES[str(itu_type)]
        selected_interval = next(
            (
                (float(lower), float(upper), coefficients)
                for (lower, upper), coefficients in intervals.items()
                if float(lower) <= requested_frequency_ghz <= float(upper)
            ),
            None,
        )
        clamped = selected_interval is None
        if clamped:
            valid_ranges = [[float(lower), float(upper)] for lower, upper in intervals]
            if policy == "strict":
                raise ValueError(
                    f"ITU material {itu_type!r} is undefined at "
                    f"{requested_frequency_ghz:g} GHz; valid ranges are {valid_ranges} GHz"
                )
            lower, upper, coefficients = min(
                (
                    (float(lower), float(upper), coefficients)
                    for (lower, upper), coefficients in intervals.items()
                ),
                key=lambda item: min(
                    abs(requested_frequency_ghz - item[0]),
                    abs(requested_frequency_ghz - item[1]),
                ),
            )
            evaluation_frequency_ghz = min(
                (lower, upper),
                key=lambda boundary: abs(requested_frequency_ghz - boundary),
            )
        else:
            lower, upper, coefficients = selected_interval
            evaluation_frequency_ghz = requested_frequency_ghz
        a, b, c, d = (float(value) for value in coefficients)
        relative_permittivity = a * evaluation_frequency_ghz**b
        conductivity_s_m = c * evaluation_frequency_ghz**d

        # Removing the callback prevents Scene.frequency from re-evaluating an
        # ITU fit outside its documented range or at a numerically unstable edge.
        material.frequency_update_callback = None
        material.relative_permittivity = relative_permittivity
        material.conductivity = conductivity_s_m
        material_records.append(
            {
                "scene_material_name": str(name),
                "itu_type": str(itu_type),
                "requested_frequency_ghz": requested_frequency_ghz,
                "evaluation_frequency_ghz": evaluation_frequency_ghz,
                "selected_valid_range_ghz": [lower, upper],
                "all_valid_ranges_ghz": [
                    [float(valid_lower), float(valid_upper)]
                    for valid_lower, valid_upper in intervals
                ],
                "clamped": clamped,
                "relative_permittivity": relative_permittivity,
                "conductivity_s_m": conductivity_s_m,
            }
        )
    unfrozen_itu_materials = []
    for name, material in sorted(scene.radio_materials.items()):
        itu_type = getattr(material, "itu_type", None)
        if itu_type is None and str(name) in ITU_MATERIALS_PROPERTIES:
            itu_type = str(name)
        if (
            itu_type in ITU_MATERIALS_PROPERTIES
            and getattr(material, "frequency_update_callback", None) is not None
        ):
            unfrozen_itu_materials.append(str(name))
    if unfrozen_itu_materials:
        raise RuntimeError(
            "failed to freeze ITU frequency callbacks before updating the scene: "
            f"{unfrozen_itu_materials}"
        )
    scene.frequency = float(requested_frequency_hz)
    return {
        "policy": policy,
        "application_stage": "before_scene_frequency_and_path_tracing",
        "requested_frequency_ghz": requested_frequency_ghz,
        "materials": material_records,
        "clamped_material_count": sum(record["clamped"] for record in material_records),
    }


def _bbox(scene: Any) -> tuple[np.ndarray, np.ndarray]:
    bounds = scene.mi_scene.bbox()
    lower = _to_numpy(bounds.min).astype(np.float64).reshape(3)
    upper = _to_numpy(bounds.max).astype(np.float64).reshape(3)
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("Sionna scene bounding box is not finite")
    return lower, upper


def _placement(
    scene: Any,
    record: RTConfiguration,
    grid_size: int,
    cell_size_m: float,
    footprints: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _bbox(scene)
    footprint = (grid_size - 1) * cell_size_m
    rng = np.random.default_rng(record.seed)
    if footprints:
        rooftop_lower, rooftop_upper = footprints[record.seed % len(footprints)]
        transmitter = (rooftop_lower + rooftop_upper) / 2.0
        transmitter[2] = rooftop_upper[2] + 3.0
        rooftop_radius = 0.5 * np.linalg.norm(rooftop_upper[:2] - rooftop_lower[:2])
    else:
        transmitter = (lower + upper) / 2.0
        transmitter[2] = 12.0
        rooftop_radius = 0.0
    margin = footprint / 2.0 + 2.0
    center_lower = lower[:2] + margin
    center_upper = upper[:2] - margin
    if np.any(center_upper <= center_lower):
        center_lower = lower[:2]
        center_upper = upper[:2]
    aperture = (record.array_size - 1) * (LIGHT_SPEED_M_S / record.frequency_hz) / 2.0
    rayleigh = 2.0 * aperture**2 / (LIGHT_SPEED_M_S / record.frequency_hz)
    configuration_index = int(record.config_id.rsplit("_", 1)[-1])
    factor = (0.25, 0.55, 0.9, 1.4)[configuration_index % 4]
    clearance = rooftop_radius + footprint / np.sqrt(2.0) + 2.0
    target_distance = float(np.clip(rayleigh * factor, clearance, max(clearance, footprint * 1.75)))
    candidates: list[tuple[float, np.ndarray]] = []
    angle_offset = rng.uniform(-np.pi, np.pi)
    for angle in angle_offset + np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
        center_xy = transmitter[:2] + target_distance * np.array([np.cos(angle), np.sin(angle)])
        center_xy = np.clip(center_xy, center_lower, center_upper)
        center = np.array([center_xy[0], center_xy[1], 1.5], dtype=np.float64)
        candidate_points = grid_xyz(center, grid_size, cell_size_m)
        occupied, _ = _geometry_features(candidate_points, footprints)
        realized_distance = np.linalg.norm(center_xy - transmitter[:2])
        score = float(np.mean(occupied) + 0.15 * abs(realized_distance - target_distance) / max(target_distance, 1.0))
        candidates.append((score, center))
    _, center = min(candidates, key=lambda item: item[0])
    return transmitter.astype(np.float64), center.astype(np.float64)


def _object_footprints(scene: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    scene_lower, scene_upper = _bbox(scene)
    scene_xy_area = float(np.prod(np.maximum(scene_upper[:2] - scene_lower[:2], 0.0)))
    background_names = {"terrain", "ground", "floor", "plane"}
    footprints: list[tuple[np.ndarray, np.ndarray]] = []
    for name, obj in scene.objects.items():
        try:
            bounds = obj.mi_mesh.bbox()
            lower = _to_numpy(bounds.min).astype(np.float64).reshape(3)
            upper = _to_numpy(bounds.max).astype(np.float64).reshape(3)
        except (AttributeError, TypeError, ValueError):
            continue
        height = upper[2] - lower[2]
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and height >= 2.0):
            continue
        normalized_name = str(name).lower()
        if normalized_name in background_names:
            continue
        object_xy_area = float(np.prod(np.maximum(upper[:2] - lower[:2], 0.0)))
        # Large terrain meshes can have a tall bounding box on sloped city scenes.
        # Treating that box as a solid building invalidates every receiver location.
        if scene_xy_area > 0.0 and object_xy_area / scene_xy_area > 0.10:
            continue
        footprints.append((lower, upper))
    return footprints


def _geometry_features(points: np.ndarray, footprints: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.zeros(points.shape[0], dtype=np.float32)
    height = np.zeros(points.shape[0], dtype=np.float32)
    for lower, upper in footprints:
        inside = (
            (points[:, 0] >= lower[0])
            & (points[:, 0] <= upper[0])
            & (points[:, 1] >= lower[1])
            & (points[:, 1] <= upper[1])
        )
        occupied[inside] = 1.0
        height[inside] = np.maximum(height[inside], float(max(upper[2] - lower[2], 0.0)))
    return occupied, height


def _extract_paths(paths: Any, tx_index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficients, _ = paths.cir(normalize_delays=False, out_type="numpy")
    coefficients = np.asarray(coefficients)[..., 0]
    interactions = _to_numpy(paths.interactions)
    vertices = _to_numpy(paths.vertices)
    if coefficients.ndim != 5:
        raise ValueError(f"unexpected Sionna coefficient shape: {coefficients.shape}")
    coefficients = coefficients[:, 0, tx_index, 0, :]
    if interactions.ndim != 4 or vertices.ndim != 5:
        raise ValueError(
            f"unexpected Sionna path geometry: interactions={interactions.shape}, vertices={vertices.shape}"
        )
    interactions = np.moveaxis(interactions[:, :, tx_index, :], 0, 2)
    vertices = np.moveaxis(vertices[:, :, tx_index, :, :], 0, 2)
    counts = np.sum(interactions != 0, axis=2).astype(np.int16)
    vertices = np.where((interactions != 0)[..., None], vertices, np.nan)
    return coefficients, vertices, counts


def _trace_batch(
    scene: Any,
    receiver_points: np.ndarray,
    settings: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sionna.rt import PathSolver, Receiver  # type: ignore

    receiver_names: list[str] = []
    for index, point in enumerate(receiver_points):
        name = f"rx_{index:05d}"
        scene.add(Receiver(name=name, position=point.tolist(), orientation=[0.0, 0.0, 0.0]))
        receiver_names.append(name)
    solver = PathSolver()
    paths = solver(
        scene=scene,
        max_depth=int(settings["max_depth"]),
        samples_per_src=int(settings["samples_per_source"]),
        synthetic_array=True,
        los=True,
        specular_reflection=True,
        diffuse_reflection=bool(settings["diffuse_reflection"]),
        refraction=True,
        diffraction=True,
        seed=int(seed),
    )
    coefficients, vertices, counts = _extract_paths(paths)
    interactions = _to_numpy(paths.interactions)
    for name in receiver_names:
        scene.remove(name)
    return coefficients, vertices, counts, interactions


def _concatenate_path_batches(
    coefficient_batches: list[np.ndarray],
    vertex_batches: list[np.ndarray],
    count_batches: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_paths = max(batch.shape[1] for batch in coefficient_batches)
    max_depth = max(batch.shape[2] for batch in vertex_batches)
    padded_coefficients: list[np.ndarray] = []
    padded_vertices: list[np.ndarray] = []
    padded_counts: list[np.ndarray] = []
    for coefficients, vertices, counts in zip(
        coefficient_batches, vertex_batches, count_batches, strict=True
    ):
        coefficient_pad = np.zeros((len(coefficients), max_paths), dtype=coefficients.dtype)
        coefficient_pad[:, : coefficients.shape[1]] = coefficients
        vertex_pad = np.full((len(vertices), max_paths, max_depth, 3), np.nan, dtype=vertices.dtype)
        vertex_pad[:, : vertices.shape[1], : vertices.shape[2]] = vertices
        count_pad = np.zeros((len(counts), max_paths), dtype=counts.dtype)
        count_pad[:, : counts.shape[1]] = counts
        padded_coefficients.append(coefficient_pad)
        padded_vertices.append(vertex_pad)
        padded_counts.append(count_pad)
    return (
        np.concatenate(padded_coefficients, axis=0),
        np.concatenate(padded_vertices, axis=0),
        np.concatenate(padded_counts, axis=0),
    )


def _validate_explicit_array(
    scene: Any,
    receiver_points: np.ndarray,
    reconstructed_channels: np.ndarray,
    record: RTConfiguration,
    config: dict[str, Any],
) -> dict[str, Any]:
    from sionna.rt import PathSolver, PlanarArray, Receiver  # type: ignore

    settings = config["data"]["sionna"]
    original_array = scene.tx_array
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=record.array_size,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    names: list[str] = []
    try:
        for index, point in enumerate(receiver_points):
            name = f"validation_rx_{index:03d}"
            scene.add(Receiver(name=name, position=point.tolist(), orientation=[0.0, 0.0, 0.0]))
            names.append(name)
        paths = PathSolver()(
            scene=scene,
            max_depth=int(settings["max_depth"]),
            samples_per_src=int(settings["samples_per_source"]),
            synthetic_array=False,
            los=True,
            specular_reflection=True,
            diffuse_reflection=bool(settings["diffuse_reflection"]),
            refraction=True,
            diffraction=True,
            seed=int(record.seed),
        )
        coefficients, _ = paths.cir(normalize_delays=False, out_type="numpy")
        coefficients = np.asarray(coefficients)[..., 0]
        if coefficients.ndim != 5:
            raise ValueError(f"unexpected explicit-array coefficient shape: {coefficients.shape}")
        explicit = np.sum(coefficients[:, 0, 0, :, :], axis=-1).astype(np.complex64)
    finally:
        for name in names:
            scene.remove(name)
        scene.tx_array = original_array
    transmitter = _to_numpy(scene.get("bs").position).astype(np.float64).reshape(3)
    return _channel_comparison_report(
        explicit,
        reconstructed_channels,
        receiver_points,
        transmitter,
        record,
        config,
    )


def _trace_explicit_channel_batch(
    scene: Any,
    receiver_points: np.ndarray,
    record: RTConfiguration,
    settings: dict[str, Any],
    seed: int,
) -> np.ndarray:
    from sionna.rt import PathSolver, PlanarArray, Receiver  # type: ignore

    original_array = scene.tx_array
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=record.array_size,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    names: list[str] = []
    try:
        for index, point in enumerate(receiver_points):
            name = f"explicit_rx_{seed}_{index:04d}"
            scene.add(Receiver(name=name, position=point.tolist(), orientation=[0.0, 0.0, 0.0]))
            names.append(name)
        paths = PathSolver()(
            scene=scene,
            max_depth=int(settings["max_depth"]),
            samples_per_src=int(settings["samples_per_source"]),
            synthetic_array=False,
            los=True,
            specular_reflection=True,
            diffuse_reflection=bool(settings["diffuse_reflection"]),
            refraction=True,
            diffraction=True,
            seed=int(seed),
        )
        coefficients, _ = paths.cir(normalize_delays=False, out_type="numpy")
        coefficients = np.asarray(coefficients)[..., 0]
        if coefficients.ndim != 5:
            raise ValueError(f"unexpected explicit-array coefficient shape: {coefficients.shape}")
        return np.sum(coefficients[:, 0, 0, :, :], axis=-1).astype(np.complex64)
    finally:
        for name in names:
            scene.remove(name)
        scene.tx_array = original_array


def _channel_comparison_report(
    explicit: np.ndarray,
    reconstructed: np.ndarray,
    receiver_points: np.ndarray,
    transmitter: np.ndarray,
    record: RTConfiguration,
    config: dict[str, Any],
) -> dict[str, Any]:
    correlation = channel_correlation(explicit, reconstructed)
    distance = np.linalg.norm(receiver_points - transmitter[None, :], axis=1)
    label_arguments = {
        "frequency_hz": record.frequency_hz,
        "far_count": int(config["model"]["far_beams"]),
        "near_angle_count": int(config["model"]["near_angles"]),
        "near_range_count": int(config["model"]["near_ranges"]),
        "distance_m": distance,
        "tx_power_dbm": float(config["system"]["tx_power_dbm"]),
        "bandwidth_hz": float(config["system"]["bandwidth_hz"]),
        "noise_figure_db": float(config["system"]["noise_figure_db"]),
        "low_margin_bps_hz": float(config["system"]["regime_low_margin_bps_hz"]),
        "high_margin_bps_hz": float(config["system"]["regime_high_margin_bps_hz"]),
    }
    explicit_labels = make_task_labels(explicit, **label_arguments)
    reconstructed_labels = make_task_labels(reconstructed, **label_arguments)
    agreements = {
        "regime": float(np.mean(explicit_labels["regime"] == reconstructed_labels["regime"])),
        "far": float(np.mean(explicit_labels["best_far_idx"] == reconstructed_labels["best_far_idx"])),
        "near_angle": float(
            np.mean(explicit_labels["best_near_angle"] == reconstructed_labels["best_near_angle"])
        ),
        "near_range": float(
            np.mean(explicit_labels["best_near_range"] == reconstructed_labels["best_near_range"])
        ),
    }
    return {
        "points": len(receiver_points),
        "median_channel_correlation": float(np.median(correlation)),
        "p10_channel_correlation": float(np.quantile(correlation, 0.1)),
        "label_agreement": agreements,
        "minimum_required_correlation": 0.95,
        "minimum_required_label_agreement": 0.90,
        "passed": bool(
            np.median(correlation) >= 0.95
            and agreements["far"] >= 0.90
            and agreements["near_angle"] >= 0.90
            and agreements["near_range"] >= 0.90
        ),
    }


def generate_sionna_scene(
    record: RTConfiguration,
    config: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    import sionna.rt  # type: ignore
    from sionna.rt import PlanarArray, Transmitter, load_scene  # type: ignore

    settings = config["data"]["sionna"]
    scene = load_scene(_scene_constant(record.scene), merge_shapes=False)
    material_frequency = _configure_itu_material_frequency(
        scene,
        float(record.frequency_hz),
        str(settings.get("material_frequency_policy", "strict")),
    )
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = scene.tx_array
    footprints = _object_footprints(scene)
    transmitter_xyz, grid_center = _placement(
        scene,
        record,
        int(config["data"]["grid_size"]),
        float(config["data"]["cell_size_m"]),
        footprints,
    )
    scene.add(Transmitter(name="bs", position=transmitter_xyz.tolist(), orientation=[0.0, 0.0, 0.0]))
    query_xyz = grid_xyz(
        grid_center,
        int(config["data"]["grid_size"]),
        float(config["data"]["cell_size_m"]),
    )
    all_coefficients: list[np.ndarray] = []
    all_vertices: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    batch_size = 64
    for start in range(0, len(query_xyz), batch_size):
        stop = min(start + batch_size, len(query_xyz))
        coefficients, vertices, counts, _ = _trace_batch(
            scene,
            query_xyz[start:stop],
            settings,
            record.seed + start,
        )
        all_coefficients.append(coefficients)
        all_vertices.append(vertices)
        all_counts.append(counts)
    coefficients, vertices, counts = _concatenate_path_batches(
        all_coefficients, all_vertices, all_counts
    )
    wavelength = LIGHT_SPEED_M_S / record.frequency_hz
    element_offsets = centered_ula_positions(record.array_size, wavelength)
    reconstructed_channels = np.empty((len(query_xyz), record.array_size), dtype=np.complex64)
    for index in range(len(query_xyz)):
        reconstructed_channels[index] = reconstruct_spherical_channel(
            transmitter_xyz,
            element_offsets,
            query_xyz[index],
            coefficients[index],
            vertices[index],
            counts[index],
            record.frequency_hz,
        )
    channel_mode = str(settings.get("channel_mode", "explicit_array"))
    if channel_mode == "explicit_array":
        explicit_batches: list[np.ndarray] = []
        explicit_batch_size = int(settings.get("explicit_batch_size", 16))
        for start in range(0, len(query_xyz), explicit_batch_size):
            stop = min(start + explicit_batch_size, len(query_xyz))
            explicit_batches.append(
                _trace_explicit_channel_batch(
                    scene,
                    query_xyz[start:stop],
                    record,
                    settings,
                    record.seed + start,
                )
            )
        channels = np.concatenate(explicit_batches, axis=0)
    elif channel_mode == "spherical_reconstruction":
        channels = reconstructed_channels
    else:
        raise ValueError(f"unknown Sionna channel mode: {channel_mode}")
    labels = make_task_labels(
        channels,
        record.frequency_hz,
        int(config["model"]["far_beams"]),
        int(config["model"]["near_angles"]),
        int(config["model"]["near_ranges"]),
        distance_m=np.linalg.norm(query_xyz - transmitter_xyz[None, :], axis=1),
        tx_power_dbm=float(config["system"]["tx_power_dbm"]),
        bandwidth_hz=float(config["system"]["bandwidth_hz"]),
        noise_figure_db=float(config["system"]["noise_figure_db"]),
        low_margin_bps_hz=float(config["system"]["regime_low_margin_bps_hz"]),
        high_margin_bps_hz=float(config["system"]["regime_high_margin_bps_hz"]),
    )
    occupied, building_height = _geometry_features(query_xyz, footprints)
    los = np.any((counts == 0) & (np.abs(coefficients) > 0), axis=1).astype(np.float32)
    direct_distance = np.linalg.norm(query_xyz - transmitter_xyz[None, :], axis=1)
    free_space_db = 20.0 * np.log10(np.maximum(4.0 * np.pi * direct_distance / wavelength, 1e-12))
    path_loss_db = -labels["rss_db"]
    excess_loss_db = np.maximum(path_loss_db - free_space_db, 0.0).astype(np.float32)
    relative = query_xyz - transmitter_xyz[None, :]
    valid_path = np.abs(coefficients) > 0
    interaction_density = (
        np.sum(counts * valid_path, axis=1) / np.maximum(np.sum(valid_path, axis=1), 1)
    ).astype(np.float32)
    environment = np.column_stack(
        [
            occupied,
            building_height,
            relative[:, 0],
            relative[:, 1],
            relative[:, 2],
            los,
            1.0 - los,
            excess_loss_db,
            interaction_density,
            np.ones(len(query_xyz), dtype=np.float32),
        ]
    ).astype(np.float32)
    arrays = {
        "query_xyz_m": query_xyz,
        "environment": environment,
        "valid_query_mask": (occupied < 0.5),
        "channel": channels,
        **{key: value for key, value in labels.items() if key not in {"near_ranges_m", "near_angles_rad"}},
        "near_ranges_axis_m": labels["near_ranges_m"],
        "near_angles_axis_rad": labels["near_angles_rad"],
    }
    save_scene(output_path, arrays)
    validation_count = min(
        int(settings.get("explicit_validation_points_per_scene", 0)),
        len(query_xyz),
    )
    if validation_count and channel_mode == "explicit_array":
        validation_indices = np.arange(validation_count, dtype=np.int64)
        explicit_validation = _channel_comparison_report(
            channels[validation_indices],
            reconstructed_channels[validation_indices],
            query_xyz[validation_indices],
            transmitter_xyz,
            record,
            config,
        )
        explicit_validation["production_channel_source"] = "explicit_array"
        explicit_validation["reconstruction_accepted"] = False
    elif validation_count:
        validation_indices = np.arange(validation_count, dtype=np.int64)
        explicit_validation = _validate_explicit_array(
            scene,
            query_xyz[validation_indices],
            reconstructed_channels[validation_indices],
            record,
            config,
        )
        explicit_validation["production_channel_source"] = "spherical_reconstruction"
        explicit_validation["reconstruction_accepted"] = bool(explicit_validation["passed"])
    else:
        explicit_validation = None
    metadata = {
        **record.__dict__,
        "query_count": len(query_xyz),
        "transmitter_xyz_m": transmitter_xyz.tolist(),
        "grid_center_xyz_m": grid_center.tolist(),
        "cache_sha256": sha256_file(output_path),
        "sionna_version": getattr(sionna.rt, "__version__", "2.0.1"),
        "solver": {
            "max_depth": int(settings["max_depth"]),
            "samples_per_source": int(settings["samples_per_source"]),
            "synthetic_array_for_path_search": True,
            "element_channel": channel_mode,
        },
        "material_frequency": material_frequency,
        "system": {
            "tx_power_dbm": float(config["system"]["tx_power_dbm"]),
            "bandwidth_hz": float(config["system"]["bandwidth_hz"]),
            "noise_figure_db": float(config["system"]["noise_figure_db"]),
            "regime_low_margin_bps_hz": float(config["system"]["regime_low_margin_bps_hz"]),
            "regime_high_margin_bps_hz": float(config["system"]["regime_high_margin_bps_hz"]),
            "regime_codes": {"near": 0, "cross": 1, "far": 2},
        },
        "explicit_array_validation": explicit_validation,
    }
    write_json_atomic(Path(output_path).with_suffix(".json"), metadata)
    return metadata
