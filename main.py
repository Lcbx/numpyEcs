from tests import *
import argparse
import sys

parser = argparse.ArgumentParser(
    prog='python game engine',
    description='WIP python game engine with custom ecs',
    epilog='link: https://github.com/Lcbx/numpyEcs')
parser.add_argument('-t', '--tests', action='store_true', help='launches unit tests')
args = parser.parse_args()

if args.tests:
    ret = pytest.main("-q tests.py".split())
    sys.exit(ret)

print('hi')
