"""Build the predictor and read the files required by each execution stage."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from .datasets.retinal import StrokeDataset, _validate_feature_schema, collate_fn
from .models.vascmedi import StrokeCausalModelModular

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "model"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def input_paths(directory):
    directory = Path(directory).resolve()
    paths = (directory / "tabular.csv", directory / "labels.csv")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def make_dataset(directory, tokenizer, cfg, split, training_dataset=None):
    table, labels = input_paths(directory)
    order_path = table.with_name("patient_order.json")
    patient_order = None
    if order_path.exists():
        patient_order = read_json(order_path)
        if not isinstance(patient_order, list):
            raise ValueError(f"Expected a patient ID list in {order_path}")
    data = StrokeDataset(
        str(table), str(labels), tokenizer, context_length=cfg.context_length,
        split=split, training_dataset=training_dataset, patient_order=patient_order,
    )
    schema = ROOT / "configs" / cfg.endpoint / "feature_schema.json"
    _validate_feature_schema(schema, data.feature_names)
    return data


def make_loader(data, cfg, *, train=False, sampler=None, generator=None):
    return torch.utils.data.DataLoader(
        data, batch_size=cfg.batch_size_training if train else cfg.val_batch_size,
        shuffle=train and sampler is None, sampler=sampler, generator=generator,
        num_workers=cfg.num_workers_dataloader, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def feature_manifest(directory):
    root = Path(directory).resolve()
    for name in ("pooler_output", "last_hidden_state"):
        if not (root / name).is_dir():
            raise FileNotFoundError(root / name)
    provenance = read_json(root / "feature_provenance.json")
    if (provenance.get("feature_shape") != [257, 1024]
            or provenance.get("encoder_frozen") is not True
            or provenance.get("task_supervised_training") is not False
            or provenance.get("export_complete") is not True
            or provenance.get("training_steps") != 0
            or provenance.get("smoke_only")):
        raise ValueError("Expected a complete frozen CLIP feature export")
    return provenance


def encoder_identity(provenance):
    identity = provenance.get("encoder_sha256") or provenance.get("base_weight_sha256")
    if not identity:
        raise ValueError("Frozen feature manifest is missing its encoder fingerprint")
    return identity


def config_namespace(config):
    defaults = {
        "enable_fsdp": False,
        "low_cpu_fsdp": False,
        "use_fast_kernels": False,
        "gradient_checkpointing": False,
        "reference_chunk_size": 4,
        "tokenizer_name": None,
    }
    cfg = SimpleNamespace(**(defaults | config))
    if cfg.endpoint not in {"ais", "chd"} or cfg.threshold != 0.5:
        raise ValueError("Choose AIS or CHD with the fixed threshold of 0.5")
    if cfg.num_epochs < 1 or cfg.num_heads < 1:
        raise ValueError("Expected positive training budget and number of heads")
    if cfg.enable_fsdp:
        raise ValueError("This entry point supports single-device training")
    return cfg


def make_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.model_name, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|image_feature|>", "[Img]", "[/Img]", "[MRG]", "[VQA]"],
    })
    return tokenizer


def make_model(cfg, tokenizer, class_weights):
    if cfg.quantization and not torch.cuda.is_available():
        raise RuntimeError("4-bit training and evaluation require CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrokeCausalModelModular(
        train_config=cfg, use_cache=False, tokenizer=tokenizer,
        image_token_index=tokenizer.encode("<|image_feature|>")[0],
        kwargs={}, wandb_run=None, class_weights=class_weights,
        tabular_input_dim=cfg.tabular_input_dim, hidden_dim=cfg.hidden_dim,
        top_k=cfg.top_k, num_heads=cfg.num_heads,
        alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma, delta=cfg.delta,
    )
    if cfg.quantization:
        model.move_auxiliary_modules(device)
    else:
        model.to(device)
    return model, device
