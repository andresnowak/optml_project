import os
import textwrap

import pytest

from src.utils import load_config


def write(tmp_path, name: str, content: str) -> str:
    path = os.path.join(tmp_path, name)
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))
    return path


def test_load_config_basic(tmp_path) -> None:
    path = write(tmp_path, "a.yaml", "experiment: linear_regression\nlr: 0.01\n")
    cfg = load_config(path)
    assert cfg == {"experiment": "linear_regression", "lr": 0.01}


def test_extends_merges_keys(tmp_path) -> None:
    write(tmp_path, "base.yaml", "lr: 1.0e-3\nsteps: 100\n")
    child = write(tmp_path, "child.yaml", "extends: base.yaml\nsteps: 200\nexperiment: x\n")
    cfg = load_config(child)
    assert cfg["lr"] == 1e-3            # inherited
    assert cfg["steps"] == 200           # child wins
    assert cfg["experiment"] == "x"      # child-only
    assert "extends" not in cfg


def test_extends_cycle_detected(tmp_path) -> None:
    a = write(tmp_path, "a.yaml", "extends: b.yaml\n")
    write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(a)


def test_dashes_in_keys_normalized(tmp_path) -> None:
    path = write(tmp_path, "a.yaml", "block-size: 64\n")
    cfg = load_config(path)
    assert cfg == {"block_size": 64}
