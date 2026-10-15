"""Run a complete offline suite or one disjoint shard from a spawnable file."""
import argparse
from pathlib import Path
import unittest


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error('Require 0 <= shard < shards and shards >= 1')
    discovered = list(cases(unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parent), pattern='test_go2_*.py')))
    if not discovered:
        parser.error('No regression tests found')
    selected = discovered[args.shard::args.shards]
    print(f'Shard {args.shard}: {len(selected)} of {len(discovered)} discovered tests', flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
    return int(not result.wasSuccessful())


if __name__ == '__main__':
    raise SystemExit(main())
