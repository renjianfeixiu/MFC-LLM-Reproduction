"""针对实际风险的检查：混入变量、因果FPT、注意力计算、缺失RUL标签梯度。"""
import numpy as np
import pytest
import torch
from scipy.io import savemat

from mfc.data import detect_fpt, read_cwru
from mfc.model import CrossAttention, ModelConfig, SignalPHM, multitask_loss, standardize


def test_cwru_mat_with_two_loads_chooses_filename_id(tmp_path):
    file = tmp_path / "99.mat"
    savemat(file, {"X098_DE_time": np.ones(20), "X098_FE_time": np.ones(20),
                  "X099_DE_time": np.full(20, 2.), "X099_FE_time": np.full(20, 3.)})
    signals = read_cwru(file)
    assert np.all(signals[0] == 2) and np.all(signals[1] == 3)


def test_attention_matches_hand_computed_values_and_gradients():
    module = CrossAttention(2)
    for layer in (module.q, module.k, module.v):
        layer.weight.data.copy_(torch.eye(2))
    q = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    k = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    v = torch.tensor([[[2., 4.], [6., 8.]]], requires_grad=True)
    actual, weights = module(q, k, v, True)
    a = np.exp(1 / np.sqrt(2)) / (np.exp(1 / np.sqrt(2)) + 1)
    expected = np.array([[[2 * a + 4 * (1 - a), 2 * (1 - a) + 4 * a],
                          [6 * a + 8 * (1 - a), 6 * (1 - a) + 8 * a]]])
    assert np.allclose(actual.detach().numpy(), expected, atol=1e-6)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 2))
    actual.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in (q, k, v))


def test_fpt_is_causal_not_shifted_to_first_of_three():
    health = np.r_[np.ones(30), 1. + np.arange(1, 31) ** 2]
    fpt, _ = detect_fpt(health, healthy_records=20, smooth_width=1)
    assert fpt == 32
    prefix_fpt, _ = detect_fpt(health[:40], healthy_records=20, smooth_width=1)
    assert prefix_fpt == fpt


def test_masked_rul_has_no_fake_regression_gradient():
    logits = torch.randn(4, 3, requires_grad=True)
    rul = torch.randn(4, requires_grad=True)
    loss, terms = multitask_loss({"logits": logits, "rul": rul}, torch.tensor([0, 1, 2, 0]),
                                  torch.full((4,), float("nan")), torch.zeros(4, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and terms["regression"] == 0
    assert torch.equal(rul.grad, torch.zeros_like(rul))


@pytest.mark.parametrize("variant", ["full", "cnn", "no_channel", "no_cross", "no_multiscale"])
def test_full_model_both_heads_and_normalization_are_finite(variant):
    torch.set_num_threads(2)
    model = SignalPHM(ModelConfig(num_classes=3, variant=variant))
    query, reference = torch.randn(4, 2, 4096), torch.randn(4, 2, 4096)
    outputs = model(query, reference)
    loss, _ = multitask_loss(outputs, torch.tensor([0, 1, 2, 0]), torch.tensor([.9, .7, .4, .1]),
                              torch.ones(4, dtype=torch.bool))
    loss.backward()
    assert outputs["logits"].shape == (4, 3) and outputs["rul"].shape == (4,)
    assert model.classifier[-1].weight.grad.abs().sum() > 0
    assert model.regressor[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(standardize(torch.ones(2, 2, 32))).all()
