# Synthetic input examples

The CSVs are artificially generated. They illustrate one label per patient, one feature row per available view, and the schema's 58 feature columns. A missing view is an omitted row. An empty feature cell represents an unobserved measurement. The image manifest demonstrates the expected columns. Supply your own images for its paths before running CLIP extraction.

From the repository root, generate three sample partitions with:

```sh
python -m scripts.run split --tabular examples/tabular.csv --labels examples/labels.csv --output outputs/example-splits
```
