"""Training entry point. All run customization lives in ``configs/*.yaml``.

    python train.py --config configs/gpt124m.yaml
    python train.py --config configs/small.yaml --max-steps 50
    python train.py --config configs/gpt124m.yaml --routing-mode global_schedule --wandb
"""

from __future__ import annotations

import argparse

from dynmuon import load_config, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    # Optional CLI overrides for the most common knobs (anything in the YAML
    # can also be overridden here by adding the matching dest).
    ap.add_argument("--model", choices=["small", "gpt124m"])
    ap.add_argument("--routing-mode", dest="routing_mode")
    ap.add_argument("--compute-mode", dest="compute_mode")
    ap.add_argument("--max-steps", dest="max_steps", type=int)
    ap.add_argument("--noise-lambda", dest="noise_lambda", type=float)
    ap.add_argument("--run-name", dest="run_name")
    ap.add_argument("--device")
    ap.add_argument("--wandb", action="store_true", default=None)
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
