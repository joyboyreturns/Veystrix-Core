"""Inspect local metadata only: python -m models.nemotron35.check CHECKPOINT."""
import argparse
import json
from .config import ConfigurationError, inspect_checkpoint

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--hashes", action="store_true", help="stream every shard through SHA-256 and compare with the recorded digest")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_checkpoint(args.checkpoint, hashes=args.hashes), indent=2))
    except ConfigurationError as exc:
        parser.exit(2, f"Nemotron check: {exc}\n")

if __name__ == "__main__":
    main()
