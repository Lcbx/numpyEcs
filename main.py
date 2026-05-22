#!python

import argparse
import sys
from pathlib import Path as path
from importlib import import_module

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
to_module = lambda p: normalize_path(p).replace('/', '.').replace('.py', '')

if args.test or args.tests:
	from glob import glob
	test_paths = glob('./tests/*.py')

# run particular test
if args.test:
	modules = [ import_module( to_module(test_path) ) for test_path in test_paths ]
	failed = True
	for m in modules:
		try:
			testFunc = getattr(m, args.test.split('::')[-1])
			failed = False
			testFunc()
			print(f'{args.test} executed.')
		except AttributeError as ex:
			if not str(ex).startswith('module'):
				raise ex
	if failed:
		print(f'could not find "{args.test}" test. candidates:')
		from pprint import pp 
		pp( list(f for m in modules for f in m.__dict__ )) # if 'test' in f) )

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
		subprocess.run( f'py -m mypyc -m common')
		subprocess.run( f'py -m mypyc -m ECS')
		subprocess.run( f'py -m nuitka --output-dir=build --standalone {scene}'.split(' ') )
	else:
		import_module( to_module(scene) )
