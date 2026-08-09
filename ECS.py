from _collections_abc import dict_keys, dict_items
from inspect import signature as inspect_signature
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from common import higher_pow2, Any, Type, Sequence, Iterator, Iterable, List, Dict, Tuple, Callable
from enum import IntFlag, auto

# NOTES:
# * if we wanted to support a really big number of entities, we could use a paged array for the sparse
# it would add an additional indirection but avoid allocating a lot of empty memory


# useful when we'll instanciate scenes, so we can translate entity ids stored in components
#class Entity(int): pass # sadly mypyc disallows inheriting builtins
#class Entity(np.uint64): pass # compiles but segfaults
Entity = np.uint64
Index = np.uint32
Generation = np.uint32
NONE = np.iinfo(Index).max
INDEX_MAX = np.iinfo(Index).max
INDEX_BITS = int(INDEX_MAX).bit_count()
GENERATION_INCREMENT = 1 << INDEX_BITS

TypeVar = Type[Any]
Array = NDArray[Any]
component = dataclass

def entity_index(entity: Array) -> Array:
	return (np.asarray(entity) & INDEX_MAX).astype(Index) 

def entity_index_1(entity: Entity) -> Index:
	return Index(entity & INDEX_MAX) 

def entity_generation(entity: Entity | Array) -> Generation | Array:
	return (np.asarray(entity) >> INDEX_BITS).astype(Generation)


class ComponentStorage:
	"""
	Stores one component type using:
	  - structured numpy array (_dense) for all fields
	  - numpy arrays for sparse (entity -> head index)
	"""

	def __init__(self,
				 component_cls: TypeVar,
				 capacity: Index = Index(128)):
		self.component_cls = component_cls
		self.mult_comp = False

		self._capacity	: Index = Index(capacity)
		self._size		: Index = Index(0)
		self._dense 	: Dict[str,Array] = {}

		self._sparse : Array  = np.full(self._capacity, NONE, dtype=Index)
		self._dense['entity'] = np.full(self._capacity, NONE, dtype=Entity)

		# Inspect dataclass annotations to build structured dtype
		ann = getattr(component_cls, "__annotations__", {})
		self.fields = list(ann.keys())

		for field, typ in ann.items():
			dtype = (
				np.float64 if typ is float else
				# issubclass is for IntFlag
				int if issubclass(typ, int) else
				type if isinstance(typ, np.dtype) else
				object
			) if isinstance(typ, type) else object
			#print(field, typ, dtype)
			self._dense[field] = np.zeros(self._capacity, dtype=dtype)

	@property
	def sparse_size(self) -> Index:
		return Index(len(self._sparse))

	@property
	def _entities_contained(self) -> Array:
		return self._dense['entity']

	@property
	def _entities_contained_capped(self) -> Array:
		return self._entities_contained[:self._size]

	def _grow_sparse(self, ent_index: Index) -> None:
		size = self.sparse_size
		new_cap = Index(higher_pow2(ent_index))
		sp = np.full(new_cap, NONE, dtype=Index)
		sp[:size] = self._sparse
		self._sparse = sp

	def _grow_dense(self, needed: Index = Index(0)) -> None:
		size = self._size
		new_cap = Index(higher_pow2(needed))
		for field_name, arr in self._dense.items():
			new_arr = np.zeros(new_cap, dtype=arr.dtype)
			new_arr[:size] = arr[:size]
			self._dense[field_name] = new_arr
		self._entities_contained[size:] = NONE
		self._capacity = new_cap

	def _set_dense_index(self, dense_idx:Index, ent_index:Index) -> None:
		# ensure sparse space
		if ent_index >= self.sparse_size: self._grow_sparse(ent_index)
		self._sparse[ent_index] = dense_idx

	def _add_at_dense_end(self, entity:Entity, component:Any) -> None:
		idx = self._size
		new_size = idx + 1
		# ensure dense space
		if new_size >= self._capacity: self._grow_dense(new_size)
		self._size = new_size
		# write record at idx
		self._set_at_index(idx, entity, component)

	def _add(self, entity:Entity, component:Any) -> None:
		idx = self._size
		self._add_at_dense_end(entity, component)
		self._set_dense_index(idx, entity_index_1(entity))

	def _set_at_index(self, idx:Index, entity:Entity, component:Any) -> None:
		self._entities_contained[idx] = entity
		for field in self.fields:
			self._dense[field][idx] = getattr(component, field)

	# NOTE: bool return tells ecs to update entity_masks for this entity
	def _remove(self, proxy:ComponentProxy) -> bool:
		entity = proxy._entity
		deleted = proxy._idx
		return self._remove_impl(entity, deleted)

	def _remove_impl(self, entity:Entity, deleted:Index) -> bool:
		head = deleted
		# swap-pop last into head
		last = self._size - 1
		if head != last:
			for arr in self._dense.values(): 
				arr[head] = arr[last]
			moved = self._entities_contained[last]
			self._sparse[moved] = head
		self._entities_contained[last] = NONE
		self._sparse[entity] = NONE
		self._size = last # decrement size
		return True

	def _remove_entity(self, entity:Entity) -> None:
		head = self._sparse[entity]
		if head != NONE:
			self._remove_impl(entity, head)

	def get(self, entity: Entity) -> Tuple[ComponentProxy, ...]|None:
		raise Exception(f"Called 'get' on ComponentStorage ({self.component_cls}, entity {entity})")

	def get_1(self, entity: Entity) -> ComponentProxy|None:
		head = self._sparse[entity]
		if head == NONE: return None
		return ComponentProxy(self, entity, head)

	def _get_rows(self, entities:Array|None=None) -> Array:
		if entities is None:
			ent = self._entities_contained[:self._size]
			idx = np.arange(ent.size)
			return idx
		else:
			ent = np.atleast_1d(entities).astype(Entity)
			# NOTE: we are exploiting the fact that both Entity and indices are ints here
			if ent.size == 0: return ent
			return self._get_rows_impl(ent)

	def _get_rows_impl(self, ent: Array) -> Array:
		ent_idx = entity_index(ent)
		idx = self._sparse[ent_idx]

		not_none = idx != NONE
		idx = idx[not_none]
		idx = idx[self._entities_contained[idx] == ent[not_none]]

		return idx.astype(Index)
	
	def get_vector(self, entities:Array|None=None) -> LazyDict:
		""" returns component fields for given entities """
		return LazyDict( self._dense, self._get_rows(entities) )
	
	def get_full_vector(self, entities:Array|None=None) -> Array:
		rows = self._get_rows(entities)
		return np.stack( tuple(self._dense[f][rows] for f in self.fields), axis=1)

	def set_vector(self, entities: Array, **value_arrays: dict_items[str,Array]) -> None:
		"""
		Overwrite the rows at `entities` for component fields in dict
		(except the internal 'entity' column) with the columns of `vector`.
		"""
		rows = self._get_rows(entities)
		for field in value_arrays.keys():
			self._dense[field][rows] = value_arrays[field]
	
	def set_full_vector(self, entities: Array, vector : Array) -> None:
		self.set_vector(entities, **dict(zip(self.fields, vector)))

	def query(self,
			  condition: Callable[..., Array],
			  entities: Array|None = None) -> Array:
		"""
		Return entity ids for which `condition` holds.

		- `condition` is a vectorized predicate with keyword args: lambda x, y: ...
		- `entities` is a susbset of entities we want apply the match on
		- entity is optionally injected as a param named `entity`
		  ex: pos.query(lambda x, entity: ...)
		"""

		if self._size == 0:
			return np.empty(0, dtype=Entity)

		idx = self._get_rows(entities)
		ent = self._entities_contained[idx]

		if idx.size == 0:
			return np.empty(0, dtype=Entity)

		arguments = {
			field: self._dense[field][idx]
			for field in inspect_signature(condition).parameters.keys()
		}

		mask = np.asarray(condition(**arguments), dtype=bool)
		matched = ent[mask]

		return np.atleast_1d(matched.astype(Entity))


class MultiComponentStorage(ComponentStorage):
	"""
	component storage that supports multiple components of the same type per entity
	slightly less performant when you add and remove components randomly
	"""

	def __init__(self,
			 component_cls: TypeVar,
			 capacity: Index = Index(128)):

		super().__init__(component_cls, capacity)
		self.mult_comp = True
		self._count : Dict[Entity, int] = {}

	# conequence of the policy used to support multiple components per entity
	def _defragment(self, needed:Index) -> None:
		new_cap = Index( higher_pow2(needed) )
		dense_idx = np.flatnonzero(self._entities_contained != NONE)
		new_size = Index(len(dense_idx))
		for field_name, arr in self._dense.items():
			new_arr = np.zeros(new_cap, dtype=arr.dtype)
			new_arr[:new_size] = arr[dense_idx]
			self._dense[field_name] = new_arr
		self._entities_contained[new_size:] = NONE
		valid = self._entities_contained != NONE
		sparse_idx = np.flatnonzero(np.r_[True, self._entities_contained[1:] != self._entities_contained[:-1]] & valid)
		self._sparse[self._entities_contained[sparse_idx]] = sparse_idx
		self._size = new_size
		self._capacity = new_cap

	def _add(self, entity:Entity, component:Any) -> None:
		idx = self._size
		count = self._count.get(entity, 0)

		ent_index = entity_index_1(entity)
		if count == 0:
			self._add_at_dense_end(entity, component)
			self._set_dense_index(idx, ent_index)
			self._count[entity] = 1
			return

		head = self._sparse[ent_index]
		last = head+count
		newCount = count+1

		self._count[entity] = newCount

		if last == idx:
			self._add_at_dense_end(entity, component)
			return

		if self._entities_contained[last] == NONE:
			self._set_at_index(last, entity, component)
			return

		needed = idx + newCount
		newLast = idx + count
		if needed > self._capacity:
			real_size = Index( sum( self._count.values() ) )

			if real_size > needed * 0.5:
				self._grow_dense(needed)

			else:
				self._defragment(needed)
				self._add(entity, component)
				return

		# move to end
		self._size = needed
		new_block = slice(idx, newLast)
		old_block = slice(head, last)
		for arr in self._dense.values():
 			arr[new_block] = arr[old_block]
		self._entities_contained[old_block] = NONE
		self._set_at_index(newLast, entity, component)
		self._set_dense_index(idx, ent_index)

	def _remove_entity(self, entity:Entity) -> None:
		rows = self._get_rows(np.array([entity]))
		if rows.size != 0:
			self._sparse[entity] = NONE
			self._entities_contained[rows] = NONE
			del self._count[entity]

	def get_1(self, entity: Entity) -> ComponentProxy|None:
		raise Exception(f"Called 'get_1' on MultiStorage ({self.component_cls}, entity {entity})")

	def get(self, entity: Entity) -> Tuple[ComponentProxy, ...]|None:
		head = Index( self._sparse[entity] )
		if head == NONE: return None
		count = self._count[entity]
		return tuple( ComponentProxy(self, entity, idx) for idx in np.arange(head, head + count, dtype=Index) )

	def _get_rows_impl(self, ent: Array) -> Array:
		ent = np.unique(ent)
		startIdx = self._sparse[entity_index(ent)]

		not_none = startIdx != NONE
		ent, startIdx = ent[not_none], startIdx[not_none]

		generation_mask = self._entities_contained[startIdx] == ent
		ent, startIdx = ent[generation_mask], startIdx[generation_mask]

		counts = np.vectorize(self._count.get, otypes=(np.uint16,))(ent, 0)

		if ent.size == 1:
			return np.arange(startIdx[0], startIdx[0] + counts[0], dtype=Index)
		
		total = counts.sum()
		base_idx = np.repeat(startIdx, counts) # [start0, start0, start1, start1, ...] size total 
		single_offsets = np.arange(total) # [0, 1, 2, 3, ...] size total
		cum_counts = np.r_[0, np.cumsum(counts)[:-1]] # [0, c0, c0+c1, ...] size counts
		large_offsets = np.repeat(cum_counts, counts) # [[0] * c0, [c0] * c1, [c0+c1] * c2, ...] size total
		offsets = single_offsets - large_offsets
		
		return base_idx + offsets.astype(Index)

	def _remove_impl(self, entity:Entity, deleted:Index) -> bool:
		head = self._sparse[entity]
		if head == NONE: return True
		# update count, swap deleted with last
		# NOTE: this is simple and fine BUT I don't like that the components relative position are not kept
		# alternative is displace all the components on the right of the deleted one
		count = self._count[entity]-1
		last = head + count
		if deleted != last:
			for arr in self._dense.values():
				arr[deleted] = arr[last]
		self._entities_contained[last] = NONE
		if count > 0:
			self._count[entity] = count
			return False
		self._sparse[entity] = NONE
		del self._count[entity]
		return True



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
	def __init__(self:ECS):
		self._stores: Dict[TypeVar, ComponentStorage] = {}
		self._comp_bits: Dict[TypeVar, int] = {}
		self._next_bit = 0
		self.entity_masks = np.zeros((0,), dtype=int)
		self._free_entities: List[Entity] = []
		self._next_entity_id : Entity = Entity(0)

	@property
	def count(self) -> Index:
		return Index( self._next_entity_id - len(self._free_entities) )
	
	@property
	def entity_masks_size(self) -> Index:
		return Index(len(self.entity_masks))
	
	def _grow_entity_mask(self, entity:Entity) -> None:
		masks_size = self.entity_masks_size
		new_cap = higher_pow2(entity+1)
		new_entity_masks = np.zeros(new_cap, dtype=Index)
		new_entity_masks[:masks_size] = self.entity_masks[:masks_size]
		self.entity_masks = new_entity_masks
	
	def create_entities(self, quantity: int) -> List[Entity]:
		"""
		Create `quantity` new entity IDs
		NOTE: re-issuing dead entity ids to keep sparse array sizes low
			  however potential bug : a re-issued entity may be mistaken for the previous dead entity
		"""
		len_free = len(self._free_entities)
		if len_free <= quantity:
			res = np.asarray(self._free_entities, dtype=Entity)
			self._free_entities = []
			quantity -= len_free
		else:
			res = np.asarray(self._free_entities[-quantity:], dtype=Entity)
			self._free_entities = self._free_entities[:-quantity]
			res = res + GENERATION_INCREMENT
			return list(res)
		
		current_id = self._next_entity_id
		self._next_entity_id += quantity
		if self.entity_masks_size <= self._next_entity_id:
			self._grow_entity_mask(self._next_entity_id)
		#if quantity == 1: return [ current_id ]
		res = np.append(res, np.arange(current_id, self._next_entity_id, dtype=Entity) )
		return list(res)
		
	def create_entity(self, *components:Any) -> Entity:
		[entity] = self.create_entities(1)
		self.add_component(entity, *components)
		return entity
	
	def delete_entity(self, entity: Entity) -> None:
		if entity in self._free_entities: return
		entity_bits = self.entity_masks[entity]
		if entity_bits:
			for comp_cls, comp_bit in self._comp_bits.items():
				if comp_bit & entity_bits:
					store = self._stores[comp_cls]
					store._remove_entity(entity)
		self.entity_masks[entity] = 0
		self._free_entities.append(entity)
	
	def register(self, *component_types:TypeVar, allow_same_type_components_per_entity:bool= False, capacity:Index=Index(128)) -> None:
		for cls in component_types:
			if cls not in self._comp_bits:
				self._comp_bits[cls] = 1 << self._next_bit
				self._next_bit += 1
			if cls not in self._stores:
				storage = MultiComponentStorage if allow_same_type_components_per_entity else ComponentStorage
				self._stores[cls] = storage(cls, capacity=capacity)
	
	def add_component(self, entity:Entity, *components:Any) -> None:
		bits = self.entity_masks[entity]
		for comp in components:
			cls : TypeVar = type(comp)
			# if this throws you pbbly did not register the component class
			bit = self._comp_bits[cls]
			bits |= bit
			self._stores[cls]._add(entity, comp)
		self.entity_masks[entity] = bits

	def remove_component(self, component: ComponentProxy) -> None:
		entity = component._entity
		store = component._store
		if entity >= self.entity_masks_size: return
		# _remove returns True if there are no components of that type left in entity
		if store and store._remove(component) and (bit := self._comp_bits.get(store.component_cls, 0))!=0:
			self.entity_masks[entity] &= ~bit

	def get_store(self, comp_cls: TypeVar) -> ComponentStorage|MultiComponentStorage|None:
		return self._stores.get(comp_cls)

	def get_vectors(self, *args:Any) -> List[LazyDict]:
		"""
		returns numeric blocks of N component types associated with entities

		Usage:
			ecs.get_vectors(C1, C2, …, entities)

		- C1…Cn are component classes
		- entities: 1D array of entity IDs
		"""
		*comp_clss, entities = args
		es = np.atleast_1d(entities).astype(int)
		return [self._stores[cls].get_vector(es) for cls in comp_clss if cls in self._stores]

	def where(self, *comp_clss:TypeVar, exclude:List[TypeVar]|None=None) -> Array:
		"""
		Usage: ecs.where(C1, C2, ...) # → all entities with those comps
		"""
		
		mask = 0
		for cls in comp_clss:
			mask |= self._comp_bits.get(cls, 0)
		if mask == 0: return np.array([])

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

	def __init__(self, store: ComponentStorage, entity: Entity, dense_index: Index):
		self._store = store
		self._idx = dense_index
		self._entity = entity

	def __getattr__(self, name: str) -> Any:
		return self._store._dense[name][self._idx]

	def __setattr__(self, name: str, value: Any) -> None:
		if name.startswith('_'):
			object.__setattr__(self, name, value)
		else:
			self._store._dense[name][self._idx] = value

	def __repr__(self) -> str:
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


class LazyDict:
	"""
	Lazy dictionary for component field vectors in a structured‐array storage.
	Returned by get_vector method from ComponentStorage.
	"""

	def __init__(self, dense:Dict[str,Array], rows:Array):
		self._dense = dense
		self._rows = rows
		self._cache : Dict[str,Array]= {}

	def __getitem__(self, field:str) -> Any:
		if field in self._cache.keys():
			return self._cache[field]

		item = self._dense[field][self._rows] 
		self._cache[field] = item
		return item

	def __setitem__(self, field:str, value:Any) -> Any:
		self._cache[field] = value

	def __len__(self) -> int:
		return len(self._dense)

	def keys(self) -> dict_keys[str,Any]:
		return self._dense.keys()

	def items(self) -> Iterator[tuple[str, Any]]:
		for field in self.keys(): yield (field, self.__getitem__(field))

	def __repr__(self) -> str:
		return '{' + ','.join( (f'"{k}":{v}' for k,v in self.items()) ) + '}'


# utility function for constructing multi-field vectors
# usage : array_dict, name0, name1, etc
# returns : stacked ndarray of fields named 
def vectorized(*args:Any) -> Array:
	arrays = tuple(args[0][name] for name in args[1:])
	return np.stack(arrays, axis=1)