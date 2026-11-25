# Experiment configurations

Each endpoint has the same set of model variants. Use the same data partitions for all variants.

| File | Variant |
|---|---|
| `full.json` | Full multimodal mediator model |
| `image_only.json` | LLaMA image-only comparison |
| `simple_fusion.json` | LLaMA and vascular table simple fusion |
| `e01.json` | Remove semantic mediator output |
| `e02.json` | Remove fundus mediator output |
| `e03.json` | Remove vascular mediator output |
| `e04.json` | Remove fundus mediator output and bidirectional interaction |
| `e05.json` | Remove vascular mediator output, bidirectional interaction, and vascular losses |
| `e06.json` | Remove bidirectional interaction |
| `e07.json` | Remove reference fusion and supervision |
| `feature_schema.json` | Ordered feature columns and vascular module ranges |
