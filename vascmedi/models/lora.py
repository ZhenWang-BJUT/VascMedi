"""LoRA settings used by the VascMedi LLaMA encoder."""

from peft import LoraConfig


def generate_peft_config(train_config, kwargs):
    if train_config.peft_method != "lora":
        raise ValueError("VascMedi supports LoRA adapters")
    options = {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "lora_dropout": 0.05,
        "inference_mode": False,
    }
    for key in options.keys() & kwargs.keys():
        options[key] = kwargs[key]
    return LoraConfig(**options)
