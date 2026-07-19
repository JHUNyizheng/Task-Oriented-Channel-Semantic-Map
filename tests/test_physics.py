import numpy as np

from tcsm_rt.physics import (
    codebook_rates,
    centered_ula_positions,
    channel_correlation,
    empirical_ddri,
    far_field_codebook,
    link_budget_snr_scale,
    make_task_labels,
    reconstruct_spherical_channel,
)


def test_los_reconstruction_matches_direct_geometry():
    frequency = 28e9
    wavelength = 299_792_458.0 / frequency
    offsets = centered_ula_positions(16, wavelength)
    tx = np.zeros(3)
    rx = np.array([12.0, 3.0, 0.0])
    center_distance = np.linalg.norm(rx - tx)
    coefficient = np.exp(-1j * 2 * np.pi * center_distance / wavelength) / center_distance
    reconstructed = reconstruct_spherical_channel(
        tx,
        offsets,
        rx,
        np.array([coefficient]),
        np.full((1, 1, 3), np.nan),
        np.array([0]),
        frequency,
    )
    element_distance = np.linalg.norm(rx[None, :] - offsets, axis=1)
    direct = np.exp(-1j * 2 * np.pi * element_distance / wavelength) / element_distance
    assert np.min(channel_correlation(direct[None], reconstructed[None])) > 0.999999


def test_task_label_shapes_and_nonnegative_rates():
    rng = np.random.default_rng(4)
    channel = (rng.normal(size=(9, 16)) + 1j * rng.normal(size=(9, 16))).astype(np.complex64)
    labels = make_task_labels(channel, 28e9, 17, 17, 5)
    assert labels["far_rates"].shape == (9, 17)
    assert labels["near_rates"].shape == (9, 85)
    assert np.all(labels["oracle_rate_bps_hz"] >= 0)
    assert set(np.unique(labels["regime"])) <= {0, 1, 2}


def test_ddri_is_large_for_orthogonal_columns():
    channel = np.eye(4, dtype=np.complex64)[None]
    assert empirical_ddri(channel)[0] > 1e6


def test_far_codebook_uses_forward_channel_convention():
    frequency = 28e9
    _, codebook = far_field_codebook(32, frequency, 17)
    broadside = codebook[:, 8].conj()[None, :]
    rates = codebook_rates(broadside, codebook, 10.0)
    assert int(np.argmax(rates)) == 8


def test_link_budget_and_regime_labels_are_physical():
    scale = link_budget_snr_scale(30.0, 100e6, 7.0)
    assert 1e11 < scale < 1e12
    frequency = 28e9
    _, codebook = far_field_codebook(16, frequency, 17)
    channel = np.repeat(codebook[:, 8].conj()[None, :], 3, axis=0) * 1e-5
    labels = make_task_labels(
        channel,
        frequency,
        17,
        17,
        5,
        distance_m=np.array([5.0, 50.0, 500.0]),
    )
    assert labels["regime"].shape == (3,)
    assert np.all(labels["far_codebook_loss_bps_hz"] >= -1e-6)
    assert np.all(labels["oracle_rate_bps_hz"] >= 0.0)
