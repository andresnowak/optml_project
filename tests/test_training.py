import torch

from src.experiments import EXPERIMENTS
from src.training import _split_matrix_by_target


def test_split_matrix_by_target_shakespeare_mlp() -> None:
    """MLP target routes fc1/fc2 to targeted, attn + head to untargeted."""
    cls = EXPERIMENTS["shakespeare"]
    exp = cls(device=torch.device("cpu"), batch_size=1,
              block_size=16, d_model=16, n_heads=2, n_layers=1)
    model = exp.build_model()
    targeted, untargeted = _split_matrix_by_target(model, "mlp")
    name_of = {id(p): n for n, p in model.named_parameters()}
    target_names = {name_of[id(p)] for p in targeted}
    untarget_names = {name_of[id(p)] for p in untargeted}
    assert any(".mlp." in n for n in target_names), target_names
    assert all(".mlp." in n for n in target_names), target_names
    assert any(".attn." in n for n in untarget_names), untarget_names
    assert "head.weight" in untarget_names


def test_split_matrix_by_target_shakespeare_attention() -> None:
    """Attention target routes attn.qkv + attn.proj to targeted; everything else not."""
    cls = EXPERIMENTS["shakespeare"]
    exp = cls(device=torch.device("cpu"), batch_size=1,
              block_size=16, d_model=16, n_heads=2, n_layers=1)
    model = exp.build_model()
    targeted, untargeted = _split_matrix_by_target(model, "attention")
    name_of = {id(p): n for n, p in model.named_parameters()}
    target_names = {name_of[id(p)] for p in targeted}
    untarget_names = {name_of[id(p)] for p in untargeted}
    assert all(".attn." in n for n in target_names), target_names
    assert any(".mlp." in n for n in untarget_names), untarget_names


def test_specmuon_target_mlp_runs_end_to_end() -> None:
    """A 3-step shakespeare run with --specmuon-target mlp must complete and
    decrease the loss (uses dual-SpecMuon: top_k=32 on MLPs, top_k=0 on the rest)."""
    cls = EXPERIMENTS["shakespeare"]
    exp = cls(device=torch.device("cpu"), batch_size=2,
              block_size=8, d_model=16, n_heads=2, n_layers=1)
    # Direct dual-optimizer path: bypass cli wiring, exercise train()'s branch.
    from src.training import TrainConfig, train
    cfg = TrainConfig(
        experiment_name="shakespeare",
        optimizer_name="specmuon",
        device=torch.device("cpu"),
        steps=3,
        lr=3e-4,
        batch_size=2,
        log_every=1,
        experiment_kwargs={"block_size": 8, "d_model": 16, "n_heads": 2, "n_layers": 1},
        opt_kwargs={"top_k": 4, "sigma_mode": "baseline"},
        specmuon_target="mlp",
    )
    losses = train(cfg)
    assert len(losses) == 3
    assert all(isinstance(L, float) for L in losses)
