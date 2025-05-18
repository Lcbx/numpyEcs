import argparse
import sys
from glob import glob

parser = argparse.ArgumentParser(
    prog='python game engine',
    description='WIP python game engine with custom ecs',
    epilog='link: https://github.com/Lcbx/numpyEcs')
parser.add_argument('-t', '--tests', action='store_true', help='launches unit tests')
parser.add_argument('--test', help='launches specific test')
parser.add_argument('-s', '--scene', help='set scene script (to run or compile)')
parser.add_argument('-c', '--compile', action='store_true', help='compile scene into standalone executable')
args = parser.parse_args()

# run particular test
if args.test:
    import tests
    getattr(tests, args.test)()

# run whole test suite using pytest
if args.tests:
    import pytest
    ret = pytest.main(['-q'] + glob('./tests/*.py') )
    sys.exit(ret)

# runs the scene
if args.scene:
    if args.compile:
        import subprocess
        subprocess.run( f'py -m nuitka --output-dir=build --standalone {args.scene}'.split(' ') )
    else:
        __import__( args.scene.replace('/', '.') )
