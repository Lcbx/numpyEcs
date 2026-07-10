#!python

from argparse import ArgumentParser
from pathlib import Path as path
from importlib import import_module

parser = ArgumentParser(
	prog='python game engine',
	description='WIP python game engine with custom ecs',
	epilog='link: https://github.com/Lcbx/numpyEcs')
parser.add_argument('scene', nargs='?', help='set scene script (to run or compile)')
parser.add_argument('-t', '--test', nargs='?', const=True, default=False, help='launches unit tests or a single test if specified by name')
parser.add_argument('-c', '--compile', action='store_true', help='compile scene into standalone executable')
args = parser.parse_args()

normalize_path = lambda p: path(p).as_posix()
to_module = lambda p: normalize_path(p).replace('/', '.').replace('.py', '')

if not (args.scene or args.test or args.compile):
    parser.error("scene is required unless --test or --compile are used")

# run test suite using pytest
elif args.test:
	import pytest
	from glob import glob
	from sys import exit
	test_paths = glob('./tests/*.py')
	pytest_args = test_paths

	# run particular test
	if type(args.test) is str and (test_name := args.test).startswith('test'):
		module_path = [ tp for tp in test_paths
			if hasattr((mod := import_module( to_module(tp) )), test_name)
		]

		if not module_path:
			print(f'could not find "{args.test}" test. candidates:')
			from pprint import pp 
			pp( list(f for m in modules for f in m.__dict__ ))
			exit(1)

		pytest_args = [ f'{module_path[0]}::{test_name}']
	ret = pytest.main(['-qs', *pytest_args] )
	exit(ret)
		

# run or compile the scene
else:
	scene = args.scene
	if args.compile:
		from subprocess import run
		run( f'py -m mypyc -m common')
		run( f'py -m mypyc -m ECS')
		if scene: run( f'py -m nuitka --output-dir=build --standalone {scene}'.split(' ') )
	else:
		import_module( to_module(scene) )
