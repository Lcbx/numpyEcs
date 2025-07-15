#!python

import argparse
import sys
from pathlib import Path as path 

parser = argparse.ArgumentParser(
	prog='python game engine',
	description='WIP python game engine with custom ecs',
	epilog='link: https://github.com/Lcbx/numpyEcs')
parser.add_argument('-t', '--tests', action='store_true', help='launches unit tests')
parser.add_argument('--test', help='launches specific test')
parser.add_argument('-s', '--scene', help='set scene script (to run or compile)')
parser.add_argument('-c', '--compile', action='store_true', help='compile scene into standalone executable')
args = parser.parse_args()

normalize_path = lambda p: path(p).as_posix()
to_module = lambda p: p.replace('/', '.').replace('.py', '')

if args.test or args.tests:
	from glob import glob
	test_paths = list(map(normalize_path, glob('./tests/*.py')))
	test_paths.remove('tests/__init__.py')

# run particular test
if args.test:
	from importlib import import_module
	for test_path in test_paths:
		module = import_module( to_module(test_path) )
		if hasattr(module, args.test): getattr(module, args.test)()

# run whole test suite using pytest
elif args.tests:
	import pytest
	ret = pytest.main(['-q'] + test_paths )
	sys.exit(ret)

# run or compile the scene
else:
	scene = args.scene or 'scenes/shadows.py'
	if args.compile:
		import subprocess
		subprocess.run( f'py -m nuitka --output-dir=build --standalone {scene}'.split(' ') )
	else:
		__import__( to_module(normalize_path(scene)) )
