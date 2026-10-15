"""List the ATen operators of an exported PyTorch model: python -m sofie.pytorch model.pt2"""

import argparse

from .parser import list_operators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="a .pt2 file written by torch.export.save")
    parser.add_argument(
        "--no-decompose", action="store_true", help="list the operators of the exported program as they are"
    )
    args = parser.parse_args()
    ops = list_operators(args.model, decompose=not args.no_decompose)
    width = max((len(o["op"]) for o in ops), default=0)
    for o in ops:
        print(f"{o['op']:<{width}}  x{o['count']:<4} {'supported' if o['supported'] else 'NOT SUPPORTED'}")
    missing = [o["op"] for o in ops if not o["supported"]]
    print(f"\n{len(ops)} operators, {len(missing)} not supported")
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
