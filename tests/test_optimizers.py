import pytest
import torch

from src.optimizers import OPTIMIZER_KWARGS, OPTIMIZERS, SpecMuon, build_optimizer


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_build_optimizer_returns_correct_class(name: str) -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = build_optimizer(name, [p], lr=1e-2, weight_decay=0.0)
    assert isinstance(opt, OPTIMIZERS[name])


def test_build_optimizer_filters_unaccepted_kwargs() -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = build_optimizer("adamw", [p], lr=1e-2, weight_decay=0.0, top_k=5, sigma_mode="clip")
    assert isinstance(opt, OPTIMIZERS["adamw"])


def test_optimizer_kwargs_schema_is_complete() -> None:
    assert set(OPTIMIZER_KWARGS) == set(OPTIMIZERS)


def test_specmuon_rejects_invalid_sigma_mode() -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="sigma_mode"):
        SpecMuon([p], lr=1e-2, sigma_mode="invalid")


def test_specmuon_step_decreases_loss_on_quadratic() -> None:
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(8, 8))
    target = torch.zeros(8, 8)
    opt = SpecMuon([W], lr=1e-1, sigma_mode="baseline")

    def loss_fn() -> torch.Tensor:
        return ((W - target) ** 2).mean()

    initial = loss_fn().item()
    opt.zero_grad()
    loss = loss_fn()
    loss.backward()
    opt.step(loss=loss)
    assert loss_fn().item() < initial


def test_specmuon_handles_4d_conv_gradient() -> None:
    """4D Conv2d-shaped parameters must train without error (flatten extension)."""
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(8, 3, 3, 3))
    target = torch.zeros_like(W)
    opt = SpecMuon([W], lr=1e-1, sigma_mode="baseline")

    def loss_fn() -> torch.Tensor:
        return ((W - target) ** 2).mean()

    initial = loss_fn().item()
    opt.zero_grad()
    loss = loss_fn()
    loss.backward()
    opt.step(loss=loss)
    assert W.shape == (8, 3, 3, 3)
    assert loss_fn().item() < initial


def test_specmuon_rejects_1d_gradient() -> None:
    p = torch.nn.Parameter(torch.randn(8))
    opt = SpecMuon([p], lr=1e-1)
    loss = (p ** 2).sum()
    loss.backward()
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        opt.step(loss=loss)


def test_specmuon_paper_defaults_match_algorithm_1() -> None:
    """Paper PINN/Burgers Table 1 defaults: μ=0.9, k=6, ξ=0.2."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = SpecMuon([p])
    g = opt.param_groups[0]
    assert g["momentum"] == 0.9
    assert g["top_k"] == 6
    assert g["sav_smooth"] == 0.2
    assert g["sigma_mode"] == "power"
    assert g["power_beta"] == 1.0
    assert g["adjust_lr_fn"] is None
    assert g["kappa"] == 0.0
    assert g["gate_threshold"] == 0.0
    assert g["gate_window"] >= 2


def test_specmuon_kappa_shifts_loss_for_negative_inputs() -> None:
    """Paper §2.1: f(Θ) + κ must be > 0 for the SAV update to be defined."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    p.grad = torch.randn(4, 4)
    opt_no_kappa = SpecMuon([p], lr=1e-2, kappa=0.0)
    with pytest.raises(ValueError, match="loss \\+ kappa"):
        opt_no_kappa.step(loss=torch.tensor(-0.5))
    opt_with_kappa = SpecMuon([p], lr=1e-2, kappa=1.0)
    opt_with_kappa.step(loss=torch.tensor(-0.5))   # should not raise


def test_specmuon_gate_threshold_zero_matches_no_gate() -> None:
    """gate_threshold=0 must produce a byte-identical update to no-gate config."""
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(8, 8))
    target = torch.zeros(8, 8)
    initial = W.detach().clone()
    opt = SpecMuon([W], lr=1e-2, momentum=0.0, top_k=4,
                   sigma_mode="baseline", gate_threshold=0.0, gate_window=10)
    loss = ((W - target) ** 2).sum()
    loss.backward()
    opt.step(loss=loss)
    delta_gated = W.detach() - initial

    torch.manual_seed(0)
    W2 = torch.nn.Parameter(torch.randn(8, 8))
    initial2 = W2.detach().clone()
    opt2 = SpecMuon([W2], lr=1e-2, momentum=0.0, top_k=4, sigma_mode="baseline")
    loss2 = ((W2 - target) ** 2).sum()
    loss2.backward()
    opt2.step(loss=loss2)
    delta_default = W2.detach() - initial2
    assert torch.allclose(delta_gated, delta_default, atol=1e-7), \
        f"gate_threshold=0 deviates; max diff = {(delta_gated - delta_default).abs().max()}"


def test_specmuon_gate_disengages_on_fast_loss_drop() -> None:
    """When loss drops fast over the window, gate sets sav_active=False."""
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(8, 8))
    opt = SpecMuon([W], lr=1e-2, momentum=0.0, top_k=4, sav_smooth=0.2,
                   sigma_mode="baseline", gate_threshold=0.01, gate_window=3)
    for L_val in (10.0, 5.0, 1.0):
        opt.zero_grad()
        W.grad = torch.randn_like(W)
        opt.step(loss=torch.tensor(L_val))
    assert opt.state[W]["sav_active_prev"] is False


def test_specmuon_stashes_iota_metric() -> None:
    """state['last_iota'] / 'last_iota_w' should be populated after step()."""
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(8, 8))
    opt = SpecMuon([W], lr=1e-2, momentum=0.0, top_k=4, sigma_mode="baseline")
    loss = ((W - 0) ** 2).sum()
    loss.backward()
    opt.step(loss=loss)
    state = opt.state[W]
    assert "last_iota" in state
    assert "last_iota_w" in state
    assert isinstance(state["last_iota"], float)
    assert state["last_iota"] >= 0.0


def _run_one_step(opt_kwargs: dict, seed: int = 0, shape: tuple = (8, 8)) -> torch.Tensor:
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(*shape))
    target = torch.zeros(*shape)
    initial = W.detach().clone()
    opt = SpecMuon([W], lr=1e-2, momentum=0.0, **opt_kwargs)
    loss = ((W - target) ** 2).sum()
    loss.backward()
    opt.step(loss=loss)
    return W.detach() - initial


def test_power_beta_1_equals_baseline() -> None:
    delta_pow = _run_one_step({"sigma_mode": "power", "power_beta": 1.0,
                               "top_k": 4, "sav_smooth": 0.2})
    delta_base = _run_one_step({"sigma_mode": "baseline",
                                "top_k": 4, "sav_smooth": 0.2})
    assert torch.allclose(delta_pow, delta_base, atol=1e-7)


def test_power_beta_0_5_equals_sqrt() -> None:
    delta_pow = _run_one_step({"sigma_mode": "power", "power_beta": 0.5,
                               "top_k": 4, "sav_smooth": 0.2})
    delta_sqrt = _run_one_step({"sigma_mode": "sqrt",
                                "top_k": 4, "sav_smooth": 0.2})
    assert torch.allclose(delta_pow, delta_sqrt, atol=1e-7)


def test_power_beta_changes_update_on_skewed_spectrum() -> None:
    """β=0 vs β=1 must differ visibly on a strongly skewed gradient spectrum."""
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(4, 4))
    V, _ = torch.linalg.qr(torch.randn(4, 4))
    s = torch.tensor([1.0, 0.1, 0.01, 0.001])
    grad = U @ torch.diag(s) @ V.T

    def step(power_beta: float) -> torch.Tensor:
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.zeros(4, 4))
        p.grad = grad.clone()
        opt = SpecMuon([p], lr=1.0, momentum=0.0, top_k=4,
                       sav_smooth=0.2, sigma_mode="power", power_beta=power_beta)
        opt.step(loss=torch.tensor(1.0))
        return p.detach().clone()

    rel_diff = (step(0.0) - step(1.0)).norm() / (step(0.0).norm() + 1e-12)
    assert rel_diff > 0.1
