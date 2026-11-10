"""Inspect local metadata only: python -m models.nemotron35.check CHECKPOINT."""
import argparse
import json
from .config import ConfigurationError, inspect_checkpoint

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2))
    except ConfigurationError as exc:
        parser.exit(2, f"Nemotron check: {exc}\n")

if __name__ == "__main__":
    main()
