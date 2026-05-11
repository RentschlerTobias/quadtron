from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
import json


@dataclass(frozen=True)
class TrainingConfig:
    # Daten
    data_path: str = "./centered_blades_cleaned.pt"
    train_val_ratio: float = 0.8
    sorting_strategy: int = 1
    quantization: int = 256
    n_sample_points: int = 1000

    # Modell
    d_model: int = 512
    n_heads: int = 8
    stage_layers: tuple = (8, 8, 8)
    n_latents: int = 64
    dropout: float = 0.1
    ffn_mult: int = 4

    # Optimierung
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    batch_size: int = 16
    accumulation_steps: int = 1
    num_epochs: int = 15
    early_stopping_patience: int = 25
    max_val_batches: int = 0  # 0 -> ganzer Val-Loader

    # Validierung
    val_every_n_epochs: int = 1

    # Laufzeit
    seed: int = 0
    precision: str = "bf16"  # "fp32" | "bf16" | "fp16"
    cudnn_deterministic: bool = False
    num_workers: int = 0
    pin_memory: bool = True

    # Persistenz / Logging
    log_dir: str = "runs"
    save_best: bool = False
    save_last: bool = False

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return sha1(payload.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        if "stage_layers" in data and isinstance(data["stage_layers"], list):
            data = {**data, "stage_layers": tuple(data["stage_layers"])}
        return cls(**data)
