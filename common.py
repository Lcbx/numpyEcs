from typing import Any, Type, Sequence, Iterator, Iterable, List, Dict, Tuple, Callable
from collections import deque

# useful for buffer resizing
# 0->0, 1->2, 2->4, 3->4, 4->8, 5->8
def higher_pow2(n: int) -> int:
	return 1 << n.bit_length()

def is_pow2(n:int) -> int:
	return  n & (n-1) == 0


# equivalent to:
#from functools import lru_cache
#cache_1 = lru_cache(maxsize=1)
# but does not need args to be hashable

def cache_1(func:Callable[...,Any]) -> Callable[...,Any]:
	prev_args:Any= None; cached:Any = None 
	def wrapper(*args:Any)->Any:
		nonlocal cached, prev_args
		if args==prev_args: return cached
		prev_args = args; cached = func(*args)
		return cached
	return wrapper

def cache(func:Callable[...,Any], size=16) -> Callable[...,Any]:
	cached:Any = deque(maxlen=size) # ring buffer
	def wrapper(*args:Any)->Any:
		nonlocal cached
		for prev_args, value in cached:
			if args==prev_args: return value
		res = func(*args) ; cached.append( (args, res) )
		return res
	return wrapper