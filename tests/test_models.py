import torch

from tcsm_rt.models import FNOOperator, GatedHLG, RadioUNet, StormRMEOperator, WNOOperator


def test_gated_hlg_shapes_and_gate_semantics():
    model = GatedHLG(24, 14, 32, 17, 17, 5)
    support = torch.randn(2, 8, 24)
    query = torch.randn(2, 11, 14)
    local_indices = torch.randint(0, 8, (2, 11, 4))
    prior = {
        "rss": torch.randn(2, 11),
        "regime": torch.randn(2, 11, 3),
        "far": torch.randn(2, 11, 17),
        "near_angle": torch.randn(2, 11, 17),
        "near_range": torch.randn(2, 11, 5),
    }
    output = model(support, query, local_indices, prior)
    assert output["rss"].shape == (2, 11)
    assert output["regime"].shape == (2, 11, 3)
    for task, gate in output["gates"].items():
        assert torch.all((gate >= 0) & (gate <= 1))
        gamma = gate.unsqueeze(-1) if output[task].ndim == 3 else gate
        torch.testing.assert_close(
            output[task],
            gamma * output[f"neural_{task}"]
            + (1.0 - gamma) * output[f"local_prior_{task}"],
        )


def test_gated_hlg_component_ablations_preserve_interface():
    support = torch.randn(1, 7, 24)
    query = torch.randn(1, 5, 14)
    local_indices = torch.randint(0, 7, (1, 5, 3))
    prior = {
        "rss": torch.randn(1, 5),
        "regime": torch.randn(1, 5, 3),
        "far": torch.randn(1, 5, 17),
        "near_angle": torch.randn(1, 5, 17),
        "near_range": torch.randn(1, 5, 5),
    }
    for ablation in (
        "no_environment",
        "no_global",
        "no_local_attention",
        "no_local_prior",
        "fixed_gate",
    ):
        model = GatedHLG(24, 14, 32, 17, 17, 5, ablation=ablation)
        output = model(support, query, local_indices, prior)
        assert output["far"].shape == (1, 5, 17)
        if ablation == "fixed_gate":
            torch.testing.assert_close(output["gates"]["far"], torch.full((1, 5), 0.5))


def test_grid_baselines_preserve_odd_grid_shape():
    x = torch.randn(2, 8, 35, 35)
    for model in (
        RadioUNet(8, 64, 12),
        FNOOperator(8, 64, 12),
        WNOOperator(8, 64, 12),
    ):
        assert model(x).shape == (2, 12, 35, 35)


def test_storm_adaptation_is_translation_invariant_in_spatial_coordinates():
    model = StormRMEOperator(61, 14, 32, (17, 17, 5)).eval()
    support = torch.randn(1, 8, 61)
    query = torch.randn(1, 6, 14)
    shift = torch.tensor([3.0, -7.0, 1.5])
    shifted_support = support.clone()
    shifted_query = query.clone()
    shifted_support[..., :3] += shift
    shifted_query[..., :3] += shift
    with torch.inference_mode():
        reference = model(support, query)
        translated = model(shifted_support, shifted_query)
    for task in ("rss", "regime", "far", "near_angle", "near_range"):
        torch.testing.assert_close(reference[task], translated[task], rtol=1e-5, atol=1e-6)
