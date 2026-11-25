# VascMedi

VascMedi uses multiview fundus images, 58 retinal vascular measurements in eight modules, and LLaMA semantic features to predict AIS or CHD. The model combines three mediator representations with queries from training reference patients.

| Location | Contents |
|---|---|
| `configs/ais/`, `configs/chd/` | Full model, image-only and simple-fusion comparisons, seven ablations, and a feature schema for each endpoint |
| `vascmedi/models/` | Shared mediator model, LLaMA implementation, and LoRA configuration |
| `vascmedi/datasets/` | Patient partitions, tabular preprocessing, view validation, and frozen CLIP extraction |
| `vascmedi/interpretability/` | Pathway responses and MR, vascular module responses and IIE, retinal patch occlusion, and plots |
| `scripts/` | Command-line entry point |
| `examples/` | Synthetic examples for checking the input format |

## Setup

Use Python 3.11 and install `requirements.txt`. The supplied 4-bit configurations use a CUDA-compatible PyTorch build and bitsandbytes. Supply local copies of the LLaMA 3 8B Instruct tokenizer/model and CLIP ViT-L/14 processor/model. 

```sh
python -m pip install -r requirements.txt
python -m scripts.run --help
```

## Prepare data

Provide separate `train/`, `validation/`, and `test/` directories, each with `tabular.csv` and `labels.csv`. Use `sample_id` consistently for patients and `v_idx` 0–3 for views. The feature columns must follow the schema in the endpoint configuration. Fit imputation and normalization statistics on training patients. Compute reference queries from training patients.

If starting with one authorized cohort and creating a new split, use the split command below. It stratifies the labeled patients with tabular features using the original two-stage 70/10/20 procedure. To reuse an existing patient assignment and order, add `--split-manifest /path/to/split_manifest.json` with the local `train_ids`, `val_ids`, and `test_ids` lists. Keep the generated `patient_order.json` with each split to preserve seeded sampling. For existing split directories without this file, patient IDs are read in sorted order. Use the same partitions for comparisons and ablations. Image manifests contain `sample_id,v_idx,image_path`, with paths relative to the manifest.

```sh
python -m scripts.run split --tabular /path/to/tabular.csv --labels /path/to/labels.csv --output data/ais
python -m scripts.run extract --images /path/to/development_images.csv --clip /path/to/clip --output cache/ais-develop
python -m scripts.run extract --images /path/to/test_images.csv --clip /path/to/clip --output cache/ais-test
```

## Train and evaluate

Run from the repository root. Training monitors validation metrics. Final testing uses the checkpoint from the last training epoch.

```sh
python -m scripts.run train --config configs/ais/full.json \
  --train-dir data/ais/train --validation-dir data/ais/validation \
  --features cache/ais-develop --model /path/to/llama --output runs/ais-full

python -m scripts.run test --run runs/ais-full \
  --train-dir data/ais/train --test-dir data/ais/test \
  --test-features cache/ais-test --output outputs/ais-full-test
```

Select `configs/chd/` for CHD. For comparisons and ablations, use the corresponding configuration and a separate output path for each run. See [configs/README.md](configs/README.md) for variant definitions. Keep run and analysis outputs local because they can contain patient IDs, images, or predictions.

## Analyze responses

```sh
python -m scripts.run explain --run runs/ais-full \
  --train-dir data/ais/train --test-dir data/ais/test \
  --test-features cache/ais-test --output outputs/ais-explain

python -m scripts.run plot-responses --source outputs/ais-explain \
  --endpoint ais --output outputs/ais-response-plots

python -m scripts.run patch --run runs/ais-full \
  --train-dir data/ais/train --test-dir data/ais/test --test-features cache/ais-test \
  --images /path/to/test_views.csv --clip /path/to/clip --output outputs/ais-patches
```

Use `plot-patch` with a saved NPZ and optionally a comparison NPZ from the same patient view. Intervention outputs quantify responses of the fitted model.

The included synthetic examples check the input format. Paper results require the corresponding authorized data and documented evaluation setup. This repository documents its current run procedure. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing status.
