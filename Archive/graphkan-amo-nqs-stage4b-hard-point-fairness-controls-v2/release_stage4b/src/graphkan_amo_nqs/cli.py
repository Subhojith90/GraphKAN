from __future__ import annotations

import argparse

from .environment import collect_environment


def main(argv=None):
    parser = argparse.ArgumentParser(description="GraphKAN AMO NQS utility CLI")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("env", help="print environment summary")
    args = parser.parse_args(argv)
    if args.cmd == "env":
        import json
        print(json.dumps(collect_environment(), indent=2))
        return 0
    parser.print_help()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
