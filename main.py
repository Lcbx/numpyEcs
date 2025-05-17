import argparse
import sys

parser = argparse.ArgumentParser(
    prog='python game engine',
    description='WIP python game engine with custom ecs',
    epilog='link: https://github.com/Lcbx/numpyEcs')
parser.add_argument('-t', '--tests', action='store_true', help='launches unit tests')
parser.add_argument('--test', help='launches specific test')
parser.add_argument('-s', '--scene', help='run provided scene script')
args = parser.parse_args()

# run particular test
if args.test:
    import tests
    getattr(tests, args.test)()

# run whole test suite using pytest
if args.tests:
    import pytest
    ret = pytest.main("-q tests.py".split())
    sys.exit(ret)

# runs the scene
if args.scene: __import__( args.scene )
