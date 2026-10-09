"""Command-line interface for wheat disease classification."""

import argparse
import sys
from pathlib import Path

from .model import build_model, load_model
from .predict import predict

RECOMMENDATIONS: dict[str, str] = {
    "BlackPoint": (
        "Use certified clean seed and apply fungicide seed treatments. "
        "Ensure grain dries quickly after harvest to limit spread."
    ),
    "FusariumFootRot": (
        "Rotate crops with non-cereal species. Apply recommended seed "
        "treatments and avoid waterlogged soils."
    ),
    "HealthyLeaf": (
        "No disease detected. Continue standard agronomic practices "
        "and monitor the crop regularly."
    ),
    "LeafBlight": (
        "Apply foliar fungicide at early symptom onset. Remove and destroy "
        "heavily infected crop residue after harvest."
    ),
    "WheatBlast": (
        "Report suspected Wheat Blast to local agricultural authorities "
        "immediately — it is a quarantine pathogen in many regions. "
        "Do not move infected material off-site."
    ),
}

DISCLAIMER = (
    "\n[DISCLAIMER] This is an automated machine-learning classification. "
    "It is NOT a substitute for professional agricultural diagnosis. "
    "Always consult a qualified agronomist before taking any "
    "crop-management action."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and return the parsed argument namespace."""
    parser = argparse.ArgumentParser(
        prog="wheat_classifier",
        description=(
            "Classify a wheat leaf image into one of 5 disease classes."
        ),
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the wheat leaf image.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a .pth weights file. Omit to use demo mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = parse_args(argv)

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.model is not None:
        if not args.model.exists():
            print(f"Error: model not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        model = load_model(str(args.model))
    else:
        model = build_model(pretrained=False)

    result = predict(args.image, model=model)
    cls = result["class"]
    conf = result["confidence"]

    print(f"\nPredicted class : {cls}")
    print(f"Confidence      : {conf:.1%}")
    print("\nAll probabilities:")
    for name, prob in result["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  {name:<20} {prob:.3f}  {bar}")

    print(f"\nRecommendation:\n  {RECOMMENDATIONS[cls]}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
