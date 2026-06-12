from __future__ import annotations

import argparse
import sys

from .config import load_config
from .trainer import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--routing-mode", dest="routing_mode")
    ap.add_argument("--compute-mode", dest="compute_mode",
                    choices=["reference", "svd", "ns"])
    ap.add_argument("--ns-variant", dest="ns_variant", choices=["quintic", "cubic"])
    ap.add_argument("--orthogonalize", choices=["ns", "svd"])
    ap.add_argument("--magnitude", choices=["none", "polar_fro"])
    ap.add_argument("--spectrum", choices=["power", "random", "inverted"])
    ap.add_argument("--track-proxies", dest="track_proxies",
                    action=argparse.BooleanOptionalAction)
    ap.add_argument("--snr-ema-decay", dest="snr_ema_decay", type=float)
    ap.add_argument("--train-steps", dest="train_steps", type=int)
    ap.add_argument("--batch-size", dest="batch_size", type=int,
                    help="sequences per optimizer step")
    ap.add_argument("--sequence-length", dest="sequence_length", type=int)
    ap.add_argument("--mbs", type=int, help="sequences per microbatch")
    ap.add_argument("--val-tokens", dest="val_tokens", type=int)
    ap.add_argument("--val-batch-size", dest="val_batch_size", type=int,
                    help="validation sequences per forward pass")
    ap.add_argument("--val-loss-every", dest="val_loss_every", type=int)
    ap.add_argument("--warmup-steps", dest="warmup_steps", type=int)
    ap.add_argument("--min-lr-ratio", dest="min_lr_ratio", type=float)
    ap.add_argument("--log-every", dest="log_every", type=int)
    ap.add_argument("--muon-lr", dest="muon_lr", type=float)
    ap.add_argument("--adam-lr", dest="adam_lr", type=float)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float)
    ap.add_argument("--adjust-lr-fn", dest="adjust_lr_fn",
                    choices=["none", "spectral_norm", "rms_norm", "keller_jordan"])
    ap.add_argument("--relmuon-scale-mode", dest="relmuon_scale_mode",
                    choices=["log1p", "rms", "complete", "log1p_aligned"])
    ap.add_argument("--relmuon-scale-cap", dest="relmuon_scale_cap", type=float)
    ap.add_argument("--kaon-steps", dest="kaon_steps", type=int)
    ap.add_argument("--kaon-lambda", dest="kaon_lambda", type=float)
    ap.add_argument("--kaon-output-scale", dest="kaon_output_scale", type=float)
    ap.add_argument("--gate-tau", dest="gate_tau", type=float)
    ap.add_argument("--homogeneous-p", dest="homogeneous_p", type=float)
    ap.add_argument("--noise-lambda", dest="noise_lambda", type=float)
    ap.add_argument("--beta", type=float)
    ap.add_argument("--lean-norm", dest="lean_norm", choices=["raw", "zscore"])
    ap.add_argument("--lean-max", dest="lean_max", type=float)
    ap.add_argument("--embed-lr", dest="embed_lr", type=float)
    ap.add_argument("--modulate-metric", dest="modulate_metric",
                    choices=["stable_rank", "snr", "snr_ema", "alignment"])
    ap.add_argument("--dynamic-ref", dest="dynamic_ref", action=argparse.BooleanOptionalAction)
    ap.add_argument("--run-name", dest="run_name")
    ap.add_argument("--wandb-group", dest="wandb_group")
    ap.add_argument("--wandb-project", dest="wandb_project")
    ap.add_argument("--wandb-entity", dest="wandb_entity")
    ap.add_argument("--device")
    ap.add_argument("--compile", dest="compile",
                    action=argparse.BooleanOptionalAction)
    ap.add_argument("--wandb", action="store_true", default=None)

    processed_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            if "=" in arg:
                flag, val = arg.split("=", 1)
                processed_args.append(f"{flag.replace('_', '-') }={val}")
            else:
                processed_args.append(arg.replace("_", "-"))
        else:
            processed_args.append(arg)

    args = ap.parse_args(processed_args)
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    train(load_config(args.config, overrides))



if __name__ == "__main__":
    main()
