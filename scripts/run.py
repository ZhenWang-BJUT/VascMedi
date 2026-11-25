"""Command-line entry points for VascMedi."""

import argparse
import logging
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    def command(name, description, required):
        entry = commands.add_parser(name, help=description)
        for option in required:
            entry.add_argument("--" + option, type=Path, required=True)
        return entry

    split = command("split", "Prepare disjoint patient CSVs", ("tabular", "labels", "output"))
    split.add_argument("--seed", type=int, default=2026)
    split.add_argument("--split-manifest", type=Path,
                       help="Reuse a local JSON manifest with train_ids, val_ids and test_ids")
    clip = command("extract", "Export frozen CLIP features", ("images", "clip", "output"))
    clip.add_argument("--batch-size", type=int, default=8)
    command("train", "Fit with train and validation files", (
        "config", "train-dir", "validation-dir", "features", "model", "output",
    ))
    common = ("run", "train-dir", "test-dir", "test-features", "output")
    command("test", "Evaluate the completed checkpoint on test files", common)
    explain = command("explain", "Compute pathway and vascular responses", common)
    explain.add_argument("--bootstrap", type=int, default=1000)
    command("patch", "Compute retinal patch responses", (*common, "images", "clip"))
    plot = command("plot-responses", "Plot saved pathway and vascular results", ("source", "output"))
    plot.add_argument("--endpoint", choices=("ais", "chd"), required=True)
    patch = command("plot-patch", "Plot a patch map with an optional comparison", ("source", "output"))
    patch.add_argument("--comparison", type=Path)
    return root


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parser().parse_args()
    if args.command == "split":
        from vascmedi.datasets.splits import prepare_splits

        print(prepare_splits(args.tabular, args.labels, args.output, args.seed,
                             split_manifest=args.split_manifest))
    elif args.command == "extract":
        from vascmedi.datasets.visual import export_features

        export_features(args.images, args.clip, args.output, args.batch_size)
    elif args.command == "train":
        from vascmedi.training import train

        train(args.config, train_directory=args.train_dir, validation_directory=args.validation_dir,
              features=args.features, model_path=args.model, output=args.output)
    elif args.command == "plot-responses":
        from vascmedi.interpretability.report import plot_responses

        plot_responses(args.source, args.output, args.endpoint)
    elif args.command == "plot-patch":
        from vascmedi.interpretability.patches import plot_patch

        plot_patch(args.source, args.output, args.comparison)
    else:
        from vascmedi.evaluation import load_for_test, test

        options = {"train_directory": args.train_dir, "test_directory": args.test_dir,
                   "test_features": args.test_features}
        if args.command == "test":
            print(test(args.run, **options, output=args.output))
        else:
            bundle = load_for_test(args.run, **options)
            if args.command == "explain":
                from vascmedi.interpretability.report import explain

                explain(bundle, args.output, args.bootstrap)
            else:
                from vascmedi.interpretability.patches import analyze_patches

                analyze_patches(bundle, args.images, args.clip, args.output)


if __name__ == "__main__":
    main()
