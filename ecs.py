import numpy as np
from inspect import signature as inspect_signature
from typing import Type, Dict, Callable, Any, List, Tuple
from dataclasses import dataclass
from enum import Flag, IntFlag, auto


# useful when we'll instanciate scenes, so we can translate entity ids stored in components
class Entity(int): pass

component = dataclass

# forward declaration
ComponentProxy = lambda s,e: None


class ComponentStorage:
	"""
	Stores one component type using:
	  - structured numpy array (_dense) for all fields
	  - numpy arrays for sparse (entity -> head index)
	"""
	NONE = -1

	def __init__(self,
				 component_cls: Type,
				 mult_comp : bool = False,
				 capacity: int = 128):
		self.component_cls = component_cls
		self.mult_comp = mult_comp

		
		self._capacity    : int = capacity
		self._size        : int = 0
		self._dense = {}
		self._sparse : np.array = np.full(self._capacity, ComponentStorage.NONE, dtype=int)
		self._dense['entity']   = np.full(self._capacity, ComponentStorage.NONE, dtype=int)
		if self.mult_comp: self._count : Dict[int, np.uint16] = {}

		# Inspect dataclass annotations to build structured dtype
		ann = getattr(component_cls, "__annotations__", {})
		self.fields = list(ann.keys())

		for field, typ in ann.items():
			dtype = (
				np.float64 if typ is float else
				# issubclass is for IntFlag
				int if issubclass(typ, int) else
				object
			) if isinstance(typ, type) else object
			#print(field, typ, dtype)
			self._dense[field] = np.zeros(self._capacity, dtype=dtype)

	@property
	def capacity(self) -> int:
		return self._capacity

	@property
	def sparse_size(self) -> int:
		return self._sparse.shape[0]

	@property
	def entities_contained(self) -> np.ndarray:
		return self._dense['entity']

	def _grow_sparse(self, entity: int) -> None:
		size = self.sparse_size
		new_cap = max(size * 2, entity + 32)
		sp = np.full(new_cap, ComponentStorage.NONE, dtype=int)
		sp[:size] = self._sparse
		self._sparse = sp

	def _grow_dense(self, needed: int = 0) -> None:
		size = self._size
		new_cap =  max(self._capacity * 2, needed + 32)
		for field_name, arr in self._dense.items():
			new_arr = np.zeros(new_cap, dtype=arr.dtype)
			new_arr[:size] = arr[:size]
			self._dense[field_name] = new_arr
		self._capacity = new_cap

	def _add(self, entity: Entity, component: Any) -> None:
		
		# ensure sparse space
		if entity >= self.sparse_size:
			self._grow_sparse(entity)
		
		idx = self._size

		def _simpleAdd():
			new_size = idx + 1
			# ensure dense space
			if new_size >= self.capacity:
				self._grow_dense(new_size)
			self._size = new_size
			# write record at idx
			self._set(idx, entity, component)


		if self.mult_comp:
			count = self._count.get(entity, np.uint16(0))
			newCount = count+1
			self._count[entity] = newCount

			if count > 0:
				head = self._sparse[entity]
				last = head+count

				if self.entities_contained[last] == ComponentStorage.NONE:
					if last == idx:
						_simpleAdd()
						return
					else:
						self._set(last, entity, component)
						return

				needed = idx + newCount
				newLast = idx + count
				if needed >= self._capacity:
					self._grow_dense(needed)
				self._size = needed
				new_block = slice(idx, newLast)
				old_block = slice(head, last)
				for arr in self._dense.values():
					arr[new_block] = arr[old_block]
				self.entities_contained[old_block] = ComponentStorage.NONE
				self._set(newLast, entity, component)
				self._sparse[entity] = idx
				return

		self._sparse[entity] = idx
		_simpleAdd()

	def _set(self, idx:int, entity:Entity, component:Any) -> None:
		self.entities_contained[idx] = entity
		for field in self.fields:
			self._dense[field][idx] = getattr(component, field)

	# NOTE: bool return tells ecs to update entity_masks for this entity
	def _remove(self, component:ComponentProxy) -> bool:
		entity = component._entity
		head = self._sparse[entity]
		if head == ComponentStorage.NONE: return True
		
		if self.mult_comp:
			# update count, swap deleted with last
			# NOTE: this is simple and fine
			# BUT I don't like that the components relative position are not kept
			count = self._count[entity]-1
			self._count[entity] = count
			last = head + count
			deleted = component._idx
			if deleted != last:
				for arr in self._dense.values(): 
					arr[deleted] = arr[last]
			self.entities_contained[last] = ComponentStorage.NONE
			if count > 1:
				return False
		else:
			# swap-pop last into head
			last = self._size - 1
			self._size = last # decrement size
			if head != last:
				for arr in self._dense.values(): 
					arr[head] = arr[last]
				moved = self.entities_contained[last]
				self._sparse[moved] = head
			self.entities_contained[last] = ComponentStorage.NONE
		
		self._sparse[entity] = ComponentStorage.NONE
		return True

	def get(self, entity: Entity) -> ComponentProxy:
		
		head = self._sparse[entity]
		
		if self.mult_comp:
			count = self._count[entity]
			return [ ComponentProxy(self, entity, idx) for idx in range(head, head + count) ]
		
		if head == ComponentStorage.NONE: return None
		return ComponentProxy(self, entity, head)

	def _get_rows(self, entities:np.ndarray|None=None) -> np.ndarray:
		if entities is None:
			ent = self.entities_contained
			idx = np.arange(ent.size)[ent != ComponentStorage.NONE]
		else:
			ent = np.atleast_1d(entities).astype(int)
			if self.mult_comp:
				ent = np.unique(ent)
				
				startIdx = self._sparse[ent]
				counts = np.vectorize(self._count.__getitem__, otypes=(np.uint16,))(ent)
				
				base_idx = np.repeat(startIdx, counts)
				total = counts.sum()
				single_offsets = np.arange(total) # [0, 1, 2, 3, ...] size total
				cum_counts = np.cumsum(counts) - counts # [0, c0, c0+c1, ...] size counts
				large_offsets = np.repeat(cum_counts, counts) # [[0] * c0, [c0] * c1, [c0+c1] * c2, ...] size total
				offsets = single_offsets - large_offsets
				
				idx = base_idx + offsets
				idx = idx.astype(int)
			else:
				idx = self._sparse[ent]
				idx = idx[idx!=ComponentStorage.NONE]
		return idx
	
	def get_vector(self, entities:np.ndarray|None=None) -> np.ndarray:
		""" returns component fields for given entities """
		return LazyDict( dict( tuple( (f, self._dense[f]) for f in self.fields) ), self._get_rows(entities))
	
	def get_full_vector(self, entities:np.ndarray|None=None) -> np.ndarray:
		rows = self._get_rows(entities)
		return np.stack( tuple(self._dense[f][rows] for f in self.fields), axis=1)

	def set_vector(self, entities: np.ndarray, **value_arrays: dict) -> None:
		"""
		Overwrite the rows at `entities` for component fields in dict
		(except the internal 'entity' column) with the columns of `vector`.
		"""
		rows = self._get_rows(entities)
		for field in value_arrays.keys():
			self._dense[field][rows] = value_arrays[field]
	
	def set_full_vector(self, entities: np.ndarray, vector : np.ndarray) -> None:
		self.set_vector(entities, **dict(zip(self.fields, vector)))

	def query(self,
			  condition: Callable,
			  entities: np.ndarray|None = None) -> np.ndarray:
		"""
		Return entity ids for which `condition` holds.

		- `condition` is a vectorized predicate with keyword args: lambda x, y: ...
		- `entities` is a susbset of entities we want apply the match on
		- entity is optionally injected as a param named `entity`
		  ex: pos.query(lambda x, entity: ...)
		"""

		if self._size == 0:
			return np.empty(0, dtype=int)

		idx = self._get_rows(entities)
		ent = self.entities_contained[idx]

		if idx.size == 0:
			return np.empty(0, dtype=int)

		ns = {field: self._dense[field][idx] for field in inspect_signature(condition).parameters.keys()}

		def raiseError():
			raise ValueError("Condition must accept all component properties as params and return a boolean array of the same length as the evaluated rows.")

		try:
			mask = condition(**ns)
		except:
			raiseError()

		mask = np.asarray(mask, dtype=bool)
		if mask.shape[0] != idx.shape[0]:
			raiseError()

		matched = ent[mask]
		if matched.size == 0:
			return np.empty(0, dtype=int)

		return matched.astype(int)



class ECS:
	"""
	ECS with:
	  - compact component arrays for vectorized ops
	  - preallocated component storage to avoid frequent reallocation
	  - per-entity bitmask for fast component queries
	  - entity ID recycling via free list

	NOTE: if you need to tag entities, just use a Set of entity ids
	(don't use an empty component, component types are costly/limited)
	"""
	def __init__(self):
		self._stores: Dict[Type, ComponentStorage] = {}
		self._comp_bits: Dict[Type, int] = {}
		self._next_bit = 0
		self.entity_masks = np.zeros((0,), dtype=np.uint64)
		self._free_entities: List[Entity] = []
		self._next_entity_id = 0

	@property
	def count(self) -> int:
		return self._next_entity_id - len(self._free_entities)
	
	@property
	def entity_masks_size(self) -> int:
		return self.entity_masks.shape[0]
	
	def _grow_entity_mask(self, entity : Entity):
		entity_masks_size = self.entity_masks_size
		new_cap = max(entity_masks_size * 2, entity+32)
		new_entity_masks = np.full((new_cap,), 0, dtype=int)
		new_entity_masks[:entity_masks_size] = self.entity_masks[:entity_masks_size]
		self.entity_masks = new_entity_masks
	
	def create_entities(self, n: int) -> List[Entity]:
		"""
		Create `n` new entity IDs
		NOTE: re-issuing dead entity ids to keep sparse array sizes low
			  however potential bug : a re-issued entity may be mistaken for the previous dead entity
		"""
		out : List[Entity] = []
		len_free = len(self._free_entities)
		if len_free:
			if len_free <= n:
				out = self._free_entities
				self._free_entities = []
				n -= len_free
			elif n == 1:
				return [ self._free_entities.pop() ]
			else:
				out = self._free_entities[-n:]
				self._free_entities = self._free_entities[:-n]
				return out
		
		current_id = self._next_entity_id
		self._next_entity_id += n
		if self.entity_masks_size <= self._next_entity_id:
			self._grow_entity_mask(self._next_entity_id)
		out.extend( range(current_id, self._next_entity_id) )
		return list(map( Entity, out)) # dunno why but it wont be Entity type otherwise
		
	def create_entity(self, *components) -> Entity:
		[entity] = self.create_entities(1)
		self.add_component(entity, *components)
		return entity
	
	def delete_entity(self, entity: Entity) -> None:
		if entity in self._free_entities: return
		for comp_cls in list(self._comp_bits):
			if self.has_component(entity, comp_cls):
				self.remove_component(entity, comp_cls)
		self._free_entities.append(entity)
	
	def register(self, *component_types, allow_same_type_components_per_entity : bool = False, capacity=128):
		for cls in component_types:
			if cls not in self._comp_bits:
				self._comp_bits[cls] = 1 << self._next_bit
				self._next_bit += 1
			if cls not in self._stores:
				self._stores[cls] = ComponentStorage(cls, mult_comp = allow_same_type_components_per_entity, capacity=capacity)
	
	def add_component(self, entity: Entity, *components) -> None:
		bits = self.entity_masks[entity]
		for comp in components:
			cls = type(comp)
			# if this throws you pbbly did not register the component class
			bit = self._comp_bits[cls]
			bits |= bit
			self._stores[cls]._add(entity, comp)
		self.entity_masks[entity] = bits

	def remove_component(self, component: ComponentProxy) -> None:
		if entity >= entity_masks_size: return
		store = self._stores.get(comp_cls)
		# _remove returns True if there are no components of that type left in entity
		if store and store._remove(entity):
			bit = self._comp_bits.get(comp_cls, 0)
			self.entity_masks[entity] &= ~bit

	def get_store(self, comp_cls: Type) -> ComponentStorage:
		return self._stores.get(comp_cls)

	def get_vectors(self, *args):
		"""
		returns numeric blocks of N component types associated with entities

		Usage:
			ecs.get_vectors(C1, C2, …, entities)

		- C1…Cn are component classes
		- entities: 1D array of entity IDs
		"""
		*comp_clss, entities = args
		es = np.atleast_1d(entities).astype(int)
		return [self._stores.get(cls).get_vector(es) for cls in comp_clss]

	def where(self, *comp_clss, exclude=None) -> np.ndarray:
		"""
		Usage: ecs.where(C1, C2, ...) # → all entities with those comps
		"""
		
		mask = 0
		for cls in comp_clss:
			mask |= self._comp_bits.get(cls, 0)
		if mask == 0: return []

		criteria = self.entity_masks & mask == mask
		
		if exclude:
			exc_mask = 0
			for cls in exclude:
				exc_mask |= self._comp_bits.get(cls, 0)
			criteria &= self.entity_masks & exc_mask == 0

		ents = np.nonzero(criteria)[0]
		
		return ents



class ComponentProxy:
	"""
	Lazy proxy for one component instance in a structured‐array storage.
	Wraps ComponentStorage, holds the entity ID and the record‐index in its _dense array.
	"""

	def __init__(self, store: ComponentStorage, entity: Entity, dense_index: int):
		self._store = store
		self._idx = dense_index
		self._entity = entity

	def __getattr__(self, name: str):
		return self._store._dense[name][self._idx]

	def __setattr__(self, name: str, value: Any):
		if name.startswith('_'):
			object.__setattr__(self, name, value)
		else:
			self._store._dense[name][self._idx] = value

	def __repr__(self):
		store = self._store
		idx = self._idx
		vals = {field: store._dense[field][idx] for field in store.fields }
		return f"<{store.component_cls.__name__}Proxy e={self._entity} {vals}>"

	def build(self) -> Any:
		"""
		Reconstruct a full component instance (dataclass) from the proxy.
		"""
		store = self._store
		kwargs = {}
		for field in store.fields:
			kwargs[field] = store._dense[field][self._idx]
		return store.component_cls(**kwargs)


class LazyDict(dict):
	"""
	Lazy dictionary for component field vectors in a structured‐array storage.
	Returned by get_vector method from ComponentStorage.
	"""

	def __init__(self, dense:dict, rows:np.ndarray):
		self._dense = dense
		self._rows = rows

	def __getitem__(self, field):
		if field in self:
			return dict.__getitem__(self, field)

		item = self._dense[field][self._rows] 
		dict.__setitem__(self, field, item)
		return item

	def __len__(self):
		return len(self._dense)

	def keys(self):
		return self._dense.keys()

	def __iter__(self):
		yield from [ (field, self.__getitem__(field)) for field in self.keys() ]

	def __repr__(self):
		return '{' + ','.join( [f'"{k}":{v}' for k,v in self] ) +  '}'


# utility function for constructing multi-field vectors
# usage : array_dict, name0, name1, etc
# returns : stacked ndarray of fields named 
def vectorized(*args) -> np.ndarray:
	arrays = tuple(args[0][name] for name in args[1:])
	return np.stack(arrays, axis=1)