import argparse
import json
from pathlib import Path

from config import TrainingConfig
from trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meshtron training entry point.")
    p.add_argument("--config", type=Path, default=None,
                   help="Optional path to a JSON config; CLI flags override its fields.")
    # Daten
    p.add_argument("--data-path", type=str)
    p.add_argument("--quantization", type=int)
    p.add_argument("--sorting-strategy", type=int)
    # Modell
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--stage-layers", type=int, nargs="+")
    p.add_argument("--n-latents", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--ffn-mult", type=int)
    # Optimierung
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation-steps", type=int)
    p.add_argument("--num-epochs", type=int)
    p.add_argument("--early-stopping-patience", type=int)
    # Laufzeit
    p.add_argument("--seed", type=int)
    p.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"])
    p.add_argument("--log-dir", type=str)
    p.add_argument("--save-best", action="store_true")
    p.add_argument("--save-last", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    base = (
        TrainingConfig.from_dict(json.loads(Path(args.config).read_text()))
        if args.config
        else TrainingConfig()
    )
    overrides = {
        k: v for k, v in vars(args).items()
        if k != "config" and v is not None and v is not False
    }
    if "stage_layers" in overrides:
        overrides["stage_layers"] = tuple(overrides["stage_layers"])
    merged = {**base.to_dict(), **overrides}
    return TrainingConfig.from_dict(merged)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    result = Trainer(cfg).run()
    print(json.dumps(result.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
