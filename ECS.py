from _collections_abc import dict_keys, dict_items
from inspect import signature as inspect_signature
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from common import higher_pow2, Any, Type, Sequence, Iterator, Iterable, List, Dict, Tuple, Callable
from enum import IntFlag, auto


# useful when we'll instanciate scenes, so we can translate entity ids stored in components
#class Entity(int): pass # sadly mypyc disallows inheriting builtins
#class Entity(np.uint64): pass # compiles but segfaults
Entity = np.uint64
Index =  np.uint32
NONE = Index(np.iinfo(Index).max)
INDEX_MAX = np.iinfo(Index).max
INDEX_BITS = int(INDEX_MAX).bit_count()

SPARSE_PAGE_BITS = 10
SPARSE_PAGE_SIZE = 1 << SPARSE_PAGE_BITS

TypeVar = Type[Any]
Array = NDArray[Any]
component = dataclass


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
		self._dense 	: Dict[str,Array]    = {}

		self._sparse : Dict[Entity, Array]    = {} # array of indices
		self._sparse_count: Dict[Entity, int] = {} # array of used slots
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
	def _entities_contained(self) -> Array:
		return self._dense['entity']

	@property
	def _entities_contained_capped(self) -> Array:
		return self._entities_contained[:self._size]

	def _sparse_location(self, entity: Entity):
		return divmod(entity, SPARSE_PAGE_SIZE)

	def _grow_dense(self, needed: Index = Index(0)) -> None:
		size = self._size
		new_cap = Index(higher_pow2(needed))
		for field_name, arr in self._dense.items():
			new_arr = np.zeros(new_cap, dtype=arr.dtype)
			new_arr[:size] = arr[:size]
			self._dense[field_name] = new_arr
		self._entities_contained[size:] = NONE
		self._capacity = new_cap

	def _get_dense_index(self, entity: Entity) -> Index:
		page_idx, offset = self._sparse_location(entity)
		page = self._sparse.get(page_idx)
		return Index(NONE if page is None else page[offset])

	def _set_dense_index_in_sparse(self, dense_idx:Index, entity:Entity) -> None:
		page_idx, offset = self._sparse_location(entity)
		page = self._sparse.get(page_idx)

		if page is None:
			page = np.full(SPARSE_PAGE_SIZE, NONE, dtype=Index)
			self._sparse[page_idx] = page
			self._sparse_count[page_idx] = 0

		if page[offset] == NONE:
			self._sparse_count[page_idx] += 1

		page[offset] = dense_idx

	def _clear_dense_index_in_sparse(self, entity:Entity) -> None:
		page_idx, offset = self._sparse_location(entity)
		page = self._sparse.get(page_idx)

		if page is None or page[offset] == NONE:
			return

		page[offset] = NONE
		count = self._sparse_count[page_idx] - 1

		if count:
			self._sparse_count[page_idx] = count
		else:
			del self._sparse[page_idx]
			del self._sparse_count[page_idx]

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
		self._set_dense_index_in_sparse(idx, entity)

	def _set_at_index(self, idx:Index, entity:Entity, component:Any) -> None:
		self._entities_contained[idx] = entity
		for field in self.fields:
			self._dense[field][idx] = getattr(component, field)

	# NOTE: bool return tells ecs to update entity_masks for this entity
	def _remove(self, proxy:ComponentProxy) -> bool:
		entity = proxy._entity
		deleted = proxy._idx
		return self._remove_impl(entity, deleted)

	def _remove_impl(self, entity: Entity, deleted: Index) -> bool:
		head = deleted
		last = self._size - 1

		if head != last:
			for arr in self._dense.values(): arr[head] = arr[last]
			moved = self._entities_contained[last]
			self._set_dense_index_in_sparse(head, moved)

		self._entities_contained[last] = NONE
		self._clear_dense_index_in_sparse(entity)
		self._size = last
		return True

	def _remove_entity(self, entity:Entity) -> None:
		head = self._get_dense_index(entity)
		if head != NONE: self._remove_impl(entity, head)

	def get(self, entity: Entity) -> Tuple[ComponentProxy, ...]|None:
		raise Exception(f"Called 'get' on ComponentStorage ({self.component_cls}, entity {entity})")

	def get_1(self, entity: Entity) -> ComponentProxy | None:
		head = self._get_dense_index(entity)
		if head == NONE: return None
		return ComponentProxy(self, entity, head)

	def _get_rows(self, entities: Array | None = None) -> Array:
		if entities is None:
			return np.arange(self._size, dtype=Index)

		ent = np.atleast_1d(entities).astype(Entity)
		return self._get_rows_impl(ent)

	def _get_rows_impl(self, ent: Array) -> Array:
		page_idx = ent >> SPARSE_PAGE_BITS
		offset = ent % SPARSE_PAGE_SIZE

		idx = np.full(ent.size, NONE, dtype=Index)

		for page_id in np.unique(page_idx):
			page = self._sparse.get(page_id)
			if page is not None:
				mask = page_idx == page_id
				idx[mask] = page[offset[mask]]

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

	def _get_dense_indices(self, entities:Array) -> Array:
		page_idx = entities // SPARSE_PAGE_SIZE
		offset = entities % SPARSE_PAGE_SIZE
		out = np.full(entities.size, NONE, dtype=Entity)

		for page_id in np.unique(page_idx):
			page = self._sparse.get(page_id)
			if page is not None:
				mask = page_idx == page_id
				out[mask] = page[offset[mask]]

		return out

	def _defragment(self, needed:Index) -> None:
		new_cap = Index(higher_pow2(needed))
		dense_idx = np.flatnonzero(self._entities_contained != NONE)
		new_size = Index(len(dense_idx))

		for field_name, arr in self._dense.items():
			new_arr = np.zeros(new_cap, dtype=arr.dtype)
			new_arr[:new_size] = arr[dense_idx]
			self._dense[field_name] = new_arr

		self._entities_contained[new_size:] = NONE

		valid = self._entities_contained[:new_size]
		sparse_idx = np.flatnonzero(
			np.r_[True, valid[1:] != valid[:-1]]
		)

		self._sparse.clear()
		self._sparse_count.clear()
		for idx in sparse_idx:
			self._set_dense_index_in_sparse( Index(idx), valid[idx] )

		self._size = new_size
		self._capacity = new_cap


	def _add(self, entity:Entity, component:Any) -> None:
		idx = self._size
		count = self._count.get(entity, 0)

		if count == 0:
			self._add_at_dense_end(entity, component)
			self._set_dense_index_in_sparse(idx, entity)
			self._count[entity] = 1
			return

		head = self._get_dense_index(entity)
		last = head + count
		newCount = count + 1

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
			real_size = Index(sum(self._count.values()))

			if real_size > needed * 0.5:
				self._grow_dense(needed)
			else:
				self._defragment(needed)
				self._add(entity, component)
				return

		self._size = needed
		new_block = slice(idx, newLast)
		old_block = slice(head, last)

		for arr in self._dense.values():
			arr[new_block] = arr[old_block]

		self._entities_contained[old_block] = NONE
		self._set_at_index(newLast, entity, component)
		self._set_dense_index_in_sparse(idx, entity)


	def _remove_entity(self, entity:Entity) -> None:
		rows = self._get_rows(np.array([entity]))
		if rows.size:
			self._set_dense_index_in_sparse(NONE, entity)
			self._entities_contained[rows] = NONE
			del self._count[entity]


	def get(self, entity:Entity) -> Tuple[ComponentProxy, ...] | None:
		head = self._get_dense_index(entity)
		if head == NONE: return None
		count = self._count[entity]
		return tuple( ComponentProxy(self, entity, idx) for idx in np.arange(head, head + count, dtype=Index) )


	def _get_rows_impl(self, ent:Array) -> Array:
		ent = np.unique(ent)
		startIdx = self._get_dense_indices(ent)

		not_none = startIdx != NONE
		ent, startIdx = ent[not_none], startIdx[not_none]

		generation_mask = self._entities_contained[startIdx] == ent
		ent, startIdx = ent[generation_mask], startIdx[generation_mask]

		counts = np.fromiter(
			(self._count[e] for e in ent),
			dtype=np.uint16,
			count=ent.size,
		)

		if ent.size == 1:
			return np.arange(
				startIdx[0],
				startIdx[0] + counts[0],
				dtype=Index,
			)

		total = counts.sum()
		base_idx = np.repeat(startIdx, counts)
		offsets = np.arange(total) - np.repeat(
			np.r_[0, np.cumsum(counts)[:-1]],
			counts,
		)

		return base_idx + offsets.astype(Index)


	def _remove_impl(self, entity:Entity, deleted:Index) -> bool:
		head = self._get_dense_index(entity)

		if head == NONE: return True

		count = self._count[entity] - 1
		last = head + count

		if deleted != last:
			for arr in self._dense.values():
				arr[deleted] = arr[last]

		self._entities_contained[last] = NONE

		if count:
			self._count[entity] = count
			return False

		self._clear_dense_index_in_sparse(entity)
		del self._count[entity]
		return True



class ECS:
	"""
	ECS with:
	- compact component arrays for vectorized ops
	- preallocated component storage
	- paged per-entity bitmasks for fast component queries
	- monotonically increasing 64-bit entity IDs

	NOTE: if you need to tag entities, consider using a Set of entity ids.
	"""

	def __init__(self:ECS):
		self._stores: Dict[TypeVar, ComponentStorage] = {}
		self._comp_bits: Dict[TypeVar, int] = {}

		# bit 0 means that the entity exists
		self._next_bit = 1

		self.entity_masks: Dict[Index, Array] = {}
		self._entity_page_count: Dict[Index, int] = {}

		self._next_entity_id: Entity = Entity(0)
		self._entity_count: Index = Index(0)

	@property
	def count(self) -> Index:
		return self._entity_count

	def _entity_mask_location(self, entity:Entity) -> tuple[Index, Index]:
		page, offset = divmod(int(entity), SPARSE_PAGE_SIZE)
		return Index(page), Index(offset)

	def _get_entity_mask(self, entity:Entity) -> int:
		page_idx, offset = self._entity_mask_location(entity)
		page = self.entity_masks.get(page_idx)
		return 0 if page is None else int(page[offset])

	def _set_entity_mask(self, entity:Entity, mask:int) -> None:
		page_idx, offset = self._entity_mask_location(entity)
		page = self.entity_masks.get(page_idx)

		if page is None:
			page = np.zeros(SPARSE_PAGE_SIZE, dtype=np.uint64)
			self.entity_masks[page_idx] = page
			self._entity_page_count[page_idx] = 0

		if page[offset] == 0:
			self._entity_page_count[page_idx] += 1

		page[offset] = mask

	def _clear_entity_mask(self, entity:Entity) -> None:
		page_idx, offset = self._entity_mask_location(entity)
		page = self.entity_masks.get(page_idx)

		if page is None or page[offset] == 0:
			return

		page[offset] = 0
		count = self._entity_page_count[page_idx] - 1

		if count:
			self._entity_page_count[page_idx] = count
		else:
			del self.entity_masks[page_idx]
			del self._entity_page_count[page_idx]

	def create_entities(self, quantity:int) -> List[Entity]:
		start = self._next_entity_id
		end = Entity(int(start) + quantity)

		entities = np.arange(start, end, dtype=Entity)
		self._next_entity_id = end
		self._entity_count = Index(int(self._entity_count) + quantity)

		for entity in entities:
			self._set_entity_mask(entity, 1) # alive bit is (1 << 0) = 1

		return list(entities)

	def create_entity(self, *components:Any) -> Entity:
		[entity] = self.create_entities(1)
		self.add_component(entity, *components)
		return entity

	def delete_entity(self, entity:Entity) -> None:
		entity_bits = self._get_entity_mask(entity)
		if not entity_bits:
			return

		for comp_cls, comp_bit in self._comp_bits.items():
			if comp_bit & entity_bits:
				self._stores[comp_cls]._remove_entity(entity)

		self._clear_entity_mask(entity)
		self._entity_count -= 1

	def register(
		self,
		*component_types:TypeVar,
		allow_same_type_components_per_entity:bool=False,
		capacity:Index=Index(128),
	) -> None:
		for cls in component_types:
			if cls not in self._comp_bits:
				self._comp_bits[cls] = 1 << self._next_bit
				self._next_bit += 1

			if cls not in self._stores:
				storage = (
					MultiComponentStorage
					if allow_same_type_components_per_entity
					else ComponentStorage
				)
				self._stores[cls] = storage(cls, capacity=capacity)

	def add_component(self, entity:Entity, *components:Any) -> None:
		bits = self._get_entity_mask(entity)
		if not bits:
			return

		for comp in components:
			cls: TypeVar = type(comp)
			bit = self._comp_bits[cls]
			bits |= bit
			self._stores[cls]._add(entity, comp)

		self._set_entity_mask(entity, bits)

	def remove_component(self, component:ComponentProxy) -> None:
		entity = component._entity
		store = component._store

		bits = self._get_entity_mask(entity)
		if not bits:
			return

		if (
			store
			and store._remove(component)
			and (bit := self._comp_bits.get(store.component_cls, 0))
		):
			self._set_entity_mask(entity, bits & ~bit)

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

	def _entities_matching_mask(self, mask:int, exc_mask:int=0) -> Array:
		found = []

		for page_idx, page in self.entity_masks.items():
			criteria = ((page & mask) == mask) & ((page & exc_mask) == 0)
			idx = np.flatnonzero(criteria)

			if idx.size: found.append(page_idx * SPARSE_PAGE_SIZE + idx)

		return ( np.concatenate(found).astype(Entity) if found else np.empty(0, dtype=Entity) )

	
	def where(self, *comp_clss:TypeVar, exclude=()) -> Array:
		"""
		Usage: ecs.where(C1, C2, ...) # → all entities with those comps
		"""
		mask = 1
		for cls in comp_clss:
			mask |= self._comp_bits[cls]

		exc_mask = 0
		for cls in exclude:
			exc_mask |= self._comp_bits.get(cls, 0)

		return self._entities_matching_mask(mask, exc_mask)


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