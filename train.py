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
    # CLI overrides for the most common knobs. Unset flags default to None and
    # are dropped by load_config, so the YAML value stands.
    ap.add_argument("--model", choices=["small", "gpt124m"])
    ap.add_argument("--mlp", choices=["gelu", "gated"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--routing-mode", dest="routing_mode")
    ap.add_argument("--compute-mode", dest="compute_mode")
    ap.add_argument("--ns-variant", dest="ns_variant", choices=["quintic", "cubic"])
    ap.add_argument("--max-steps", dest="max_steps", type=int)
    ap.add_argument("--muon-lr", dest="muon_lr", type=float)
    ap.add_argument("--noise-lambda", dest="noise_lambda", type=float)
    # Router knobs (override the active route block).
    ap.add_argument("--beta", type=float)
    ap.add_argument("--modulate-metric", dest="modulate_metric",
                    choices=["stable_rank", "snr", "alignment"])
    ap.add_argument("--dynamic-ref", dest="dynamic_ref", action=argparse.BooleanOptionalAction)
    ap.add_argument("--run-name", dest="run_name")
    ap.add_argument("--wandb-group", dest="wandb_group")
    ap.add_argument("--device")
    ap.add_argument("--wandb", action="store_true", default=None)
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
