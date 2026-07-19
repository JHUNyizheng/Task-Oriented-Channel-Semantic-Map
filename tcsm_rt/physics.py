from __future__ import annotations

import numpy as np


LIGHT_SPEED_M_S = 299_792_458.0
THERMAL_NOISE_DBM_HZ = -174.0


def centered_ula_positions(array_size: int, wavelength_m: float) -> np.ndarray:
    spacing = wavelength_m / 2.0
    coordinate = np.arange(array_size, dtype=np.float64) - (array_size - 1) / 2.0
    result = np.zeros((array_size, 3), dtype=np.float64)
    result[:, 1] = coordinate * spacing
    return result


def reconstruct_spherical_channel(
    tx_center_xyz_m: np.ndarray,
    tx_element_offsets_m: np.ndarray,
    rx_xyz_m: np.ndarray,
    path_coefficients: np.ndarray,
    path_vertices_xyz_m: np.ndarray,
    path_vertex_counts: np.ndarray,
    frequency_hz: float,
) -> np.ndarray:
    """Reconstruct element-wise channels from center-device ray paths.

    The RT coefficient at the array center is retained. Element-specific phase and spreading
    corrections use the exact distance from each element to the first interaction point. For
    LoS, the receiver is the first endpoint. This approximation is audited against explicit
    per-element path tracing before its outputs can enter the manuscript.
    """
    tx_center = np.asarray(tx_center_xyz_m, dtype=np.float64).reshape(3)
    elements = tx_center[None, :] + np.asarray(tx_element_offsets_m, dtype=np.float64)
    rx = np.asarray(rx_xyz_m, dtype=np.float64).reshape(3)
    coefficients = np.asarray(path_coefficients, dtype=np.complex128).reshape(-1)
    vertices = np.asarray(path_vertices_xyz_m, dtype=np.float64)
    counts = np.asarray(path_vertex_counts, dtype=np.int64).reshape(-1)
    if vertices.shape[0] != coefficients.size or counts.size != coefficients.size:
        raise ValueError("path arrays have inconsistent path dimensions")
    wavelength = LIGHT_SPEED_M_S / float(frequency_hz)
    wavenumber = 2.0 * np.pi / wavelength
    channel = np.zeros(elements.shape[0], dtype=np.complex128)
    for path_index, coefficient in enumerate(coefficients):
        if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
            continue
        count = int(counts[path_index])
        first_endpoint = rx if count == 0 else vertices[path_index, 0]
        center_first = np.linalg.norm(first_endpoint - tx_center)
        element_first = np.linalg.norm(first_endpoint[None, :] - elements, axis=1)
        center_first = max(center_first, 1e-9)
        element_first = np.maximum(element_first, 1e-9)
        delta_distance = element_first - center_first
        spreading = center_first / element_first
        channel += coefficient * spreading * np.exp(-1j * wavenumber * delta_distance)
    return channel.astype(np.complex64)


def far_field_codebook(array_size: int, frequency_hz: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    wavelength = LIGHT_SPEED_M_S / frequency_hz
    elements = centered_ula_positions(array_size, wavelength)[:, 1]
    angles = np.deg2rad(np.linspace(-70.0, 70.0, count, dtype=np.float64))
    weights = np.exp(-1j * 2.0 * np.pi * elements[:, None] * np.sin(angles)[None, :] / wavelength)
    weights /= np.sqrt(array_size)
    return angles.astype(np.float32), weights.astype(np.complex64)


def near_field_codebook(
    array_size: int,
    frequency_hz: float,
    angle_count: int,
    range_count: int,
    min_range_m: float = 4.0,
    max_range_m: float = 180.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelength = LIGHT_SPEED_M_S / frequency_hz
    elements = centered_ula_positions(array_size, wavelength)
    angles = np.deg2rad(np.linspace(-70.0, 70.0, angle_count, dtype=np.float64))
    ranges = np.geomspace(min_range_m, max_range_m, range_count, dtype=np.float64)
    beams: list[np.ndarray] = []
    for focus_range in ranges:
        for angle in angles:
            focus = np.array(
                [focus_range * np.cos(angle), focus_range * np.sin(angle), 0.0],
                dtype=np.float64,
            )
            distance = np.linalg.norm(focus[None, :] - elements, axis=1)
            response = np.exp(-1j * 2.0 * np.pi * distance / wavelength) / np.maximum(distance, 1e-9)
            weight = np.conj(response)
            weight /= max(np.linalg.norm(weight), 1e-12)
            beams.append(weight.astype(np.complex64))
    return ranges.astype(np.float32), angles.astype(np.float32), np.column_stack(beams)


def codebook_rates(
    channel: np.ndarray,
    codebook: np.ndarray,
    snr_linear: float,
) -> np.ndarray:
    h = np.asarray(channel, dtype=np.complex64)
    w = np.asarray(codebook, dtype=np.complex64)
    # The stored channel is the forward element response h. A transmit
    # codeword therefore produces h^T w; conjugation is already carried by w.
    gain = np.abs(np.einsum("na,ab->nb", h, w, optimize=False)) ** 2
    return np.log2(1.0 + float(snr_linear) * gain).astype(np.float32)


def link_budget_snr_scale(
    tx_power_dbm: float,
    bandwidth_hz: float,
    noise_figure_db: float,
) -> float:
    """Return P_tx/P_noise for dimensionless RT channel coefficients."""
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    noise_dbm = THERMAL_NOISE_DBM_HZ + 10.0 * np.log10(bandwidth_hz) + noise_figure_db
    return float(10.0 ** ((tx_power_dbm - noise_dbm) / 10.0))


def classify_near_cross_far(
    distance_m: np.ndarray,
    rayleigh_distance_m: np.ndarray | float,
    far_codebook_loss_bps_hz: np.ndarray,
    low_margin_bps_hz: np.ndarray | float,
    high_margin_bps_hz: np.ndarray | float,
) -> np.ndarray:
    """Apply the paper's geometry-and-rate near/cross/far definition.

    Coding is 0=near, 1=cross, and 2=far. The Rayleigh boundary belongs to
    the near-geometry region. Rate margins are inclusive at their respective
    near and far decisions.
    """
    loss = np.asarray(far_codebook_loss_bps_hz, dtype=np.float64)
    distance = np.broadcast_to(np.asarray(distance_m, dtype=np.float64), loss.shape)
    rayleigh = np.broadcast_to(np.asarray(rayleigh_distance_m, dtype=np.float64), loss.shape)
    low = np.broadcast_to(np.asarray(low_margin_bps_hz, dtype=np.float64), loss.shape)
    high = np.broadcast_to(np.asarray(high_margin_bps_hz, dtype=np.float64), loss.shape)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError("distance_m must contain positive finite values")
    if np.any(~np.isfinite(rayleigh)) or np.any(rayleigh <= 0.0):
        raise ValueError("rayleigh_distance_m must contain positive finite values")
    if np.any(~np.isfinite(loss)) or np.any(loss < -1e-6):
        raise ValueError("far_codebook_loss_bps_hz must contain finite nonnegative values")
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        raise ValueError("regime margins must be finite")
    if np.any(low < 0.0) or np.any(low >= high):
        raise ValueError("all regime margins must satisfy 0 <= low < high")
    regime = np.full(loss.shape, 1, dtype=np.int8)
    regime[(distance <= rayleigh) & (loss >= high)] = 0
    regime[(distance > rayleigh) & (loss <= low)] = 2
    return regime


def make_task_labels(
    channel: np.ndarray,
    frequency_hz: float,
    far_count: int,
    near_angle_count: int,
    near_range_count: int,
    distance_m: np.ndarray | None = None,
    tx_power_dbm: float = 30.0,
    bandwidth_hz: float = 100e6,
    noise_figure_db: float = 7.0,
    low_margin_bps_hz: float = 0.20,
    high_margin_bps_hz: float = 0.75,
) -> dict[str, np.ndarray]:
    """Construct geometry-and-rate near/cross/far task labels.

    Label coding is fixed to 0=near, 1=cross, and 2=far. Near requires both
    an in-Rayleigh location and a material far-codebook loss. Far requires an
    out-of-Rayleigh location with a negligible far-codebook loss. All other
    points are cross-field.
    """
    channels = np.asarray(channel, dtype=np.complex64)
    if channels.ndim != 2:
        raise ValueError("channel must have shape [query, tx_element]")
    if not 0.0 <= low_margin_bps_hz < high_margin_bps_hz:
        raise ValueError("regime margins must satisfy 0 <= low < high")
    array_size = channels.shape[1]
    _, far_codebook = far_field_codebook(array_size, frequency_hz, far_count)
    ranges, angles, near_codebook = near_field_codebook(
        array_size,
        frequency_hz,
        near_angle_count,
        near_range_count,
    )
    snr_scale = link_budget_snr_scale(tx_power_dbm, bandwidth_hz, noise_figure_db)
    far_rates = codebook_rates(channels, far_codebook, snr_scale)
    near_rates = codebook_rates(channels, near_codebook, snr_scale)
    best_far = np.argmax(far_rates, axis=1).astype(np.int16)
    best_near_flat = np.argmax(near_rates, axis=1)
    best_near_range = (best_near_flat // near_angle_count).astype(np.int16)
    best_near_angle = (best_near_flat % near_angle_count).astype(np.int16)
    far_best_rate = np.max(far_rates, axis=1)
    near_best_rate = np.max(near_rates, axis=1)
    oracle_rate = np.maximum(far_best_rate, near_best_rate)
    far_codebook_loss = oracle_rate - far_best_rate
    advantage = near_best_rate - far_best_rate
    if distance_m is None:
        distances = np.full(channels.shape[0], np.nan, dtype=np.float64)
        regime = np.full(channels.shape[0], 1, dtype=np.int8)
        regime[far_codebook_loss >= high_margin_bps_hz] = 0
        regime[far_codebook_loss <= low_margin_bps_hz] = 2
    else:
        distances = np.asarray(distance_m, dtype=np.float64).reshape(-1)
        if distances.shape[0] != channels.shape[0] or np.any(distances <= 0):
            raise ValueError("distance_m must contain one positive value per query")
        boundary = rayleigh_distance(array_size, frequency_hz)
        regime = classify_near_cross_far(
            distances,
            boundary,
            far_codebook_loss,
            low_margin_bps_hz,
            high_margin_bps_hz,
        )
    rss_linear = np.sum(np.abs(channels) ** 2, axis=1)
    return {
        "rss_db": (10.0 * np.log10(np.maximum(rss_linear, 1e-30))).astype(np.float32),
        "regime": regime,
        "best_far_idx": best_far,
        "best_near_angle": best_near_angle,
        "best_near_range": best_near_range,
        "far_rates": far_rates,
        "near_rates": near_rates,
        "oracle_rate_bps_hz": oracle_rate.astype(np.float32),
        "far_codebook_loss_bps_hz": far_codebook_loss.astype(np.float32),
        "near_advantage_bps_hz": advantage.astype(np.float32),
        "distance_m": distances.astype(np.float32),
        "rayleigh_distance_m": np.full(
            channels.shape[0], rayleigh_distance(array_size, frequency_hz), dtype=np.float32
        ),
        "link_snr_scale": np.full(channels.shape[0], snr_scale, dtype=np.float64),
        "near_ranges_m": ranges,
        "near_angles_rad": angles,
    }


def channel_correlation(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    numerator = np.abs(np.sum(np.conj(reference) * estimate, axis=-1))
    denominator = np.linalg.norm(reference, axis=-1) * np.linalg.norm(estimate, axis=-1)
    result = np.zeros_like(denominator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return np.clip(result, 0.0, 1.0)


def empirical_ddri(channel_groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(channel_groups, dtype=np.complex128)
    gram = np.einsum("gka,gla->gkl", groups, np.conj(groups))
    diagonal = np.abs(np.diagonal(gram, axis1=1, axis2=2))
    off_diagonal = np.sum(np.abs(gram), axis=2) - diagonal
    ratio = diagonal / np.maximum(off_diagonal, 1e-12)
    return np.min(ratio, axis=1).astype(np.float64)


def rayleigh_distance(array_size: int, frequency_hz: float) -> float:
    wavelength = LIGHT_SPEED_M_S / frequency_hz
    aperture = (array_size - 1) * wavelength / 2.0
    return 2.0 * aperture**2 / wavelength
