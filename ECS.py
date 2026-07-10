from _collections_abc import dict_keys, dict_items
from inspect import signature as inspect_signature
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from common import higher_pow2, Any, Type, Sequence, Iterator, Iterable, List, Dict, Tuple, Callable
from enum import IntFlag, auto

# NOTES:
# * referencing components or entities is not easy bc reused Entity Ids and slots can create collisions
# a solution would be generational ids : reserve some bits from Entity Id for the 'version' then check on access ?
# but let's not fix problems we dont have yet.
# * if we wanted to support a really big number of entities, we could use a paged array for the sparse
# it would add an additional indirection but avoid allocating a lot of empty memory


# useful when we'll instanciate scenes, so we can translate entity ids stored in components
#class Entity(int): pass # sadly mypyc disallows inheriting builtins
Entity = int|np.int64
NONE = -1

TypeVar = Type[Any]
Array = NDArray[Any]
component = dataclass


class ComponentStorage:
	"""
	Stores one component type using:
	  - one structured NumPy array for dense component data
	  - one NumPy sparse array mapping entity -> dense index
	"""

	def __init__(
		self,
		component_cls: TypeVar,
		capacity: int = 128,
	):
		self.component_cls = component_cls
		self.mult_comp = False

		self._size: int = 0

		ann = getattr(component_cls, "__annotations__", {})
		self.fields = list(ann.keys())

		dense_dtype = np.dtype(
			[("entity", np.int64)] +
			[
				(field,
					 np.float64 if typ is float
				else np.bool_ if type is bool # I recommend IntFlag instead of bool
				else (
					int if issubclass(typ, int)
					else typ if isinstance(typ, np.dtype)
					else object
				) if typ is type
				else object
				) for field, typ in ann.items()
			]
		)

		self._dense: Array = np.zeros(capacity, dtype=dense_dtype)
		self._dense["entity"] = NONE

		self._sparse: Array = np.full(capacity, NONE, dtype=np.int64)

	@property
	def sparse_size(self) -> int:
		return int(self._sparse.shape[0])

	@property
	def capacity(self) -> int:
		return int(self._dense.shape[0])

	@property
	def _entities_contained(self) -> Array:
		return self._dense["entity"][:self._size+1]

	def _grow_sparse(self, entity: int) -> None:
		old_size = self.sparse_size
		new_capacity = higher_pow2(entity + 1)

		new_sparse = np.full(new_capacity, NONE, dtype=self._sparse.dtype)
		new_sparse[:old_size] = self._sparse

		self._sparse = new_sparse

	def _grow_dense(self, needed: int = 0) -> None:
		new_capacity = higher_pow2(needed)

		new_dense = np.zeros(new_capacity, dtype=self._dense.dtype)
		new_dense["entity"] = NONE
		new_dense[:self._size] = self._dense[:self._size]

		self._dense = new_dense

	def _set_entity(self, idx: int, entity: Entity) -> None:
		if entity >= self.sparse_size:
			self._grow_sparse(int(entity))

		self._sparse[entity] = idx

	def _simpleAdd(self, entity: Entity, component: Any) -> None:
		idx = self._size
		new_size = idx + 1

		if new_size > self.capacity:
			self._grow_dense(new_size)

		self._size = new_size
		self._set(idx, entity, component)

	def _add(self, entity: Entity, component: Any) -> None:
		idx = self._size
		self._simpleAdd(entity, component)
		self._set_entity(idx, entity)

	def _set(
		self,
		idx: int | np.integer,
		entity: Entity,
		component: Any,
	) -> None:
		self._dense["entity"][idx] = entity

		for field in self.fields:
			self._dense[field][idx] = getattr(component, field)

	# bool return tells ECS to update entity_masks for this entity.
	def _remove(self, proxy: ComponentProxy) -> bool:
		return self._remove_impl(proxy._entity, proxy._idx)

	def _remove_impl(
		self,
		entity: Entity,
		deleted: int | np.integer,
	) -> bool:
		last = self._size - 1
		self._size = last

		if deleted != last:
			# Structured arrays allow moving the complete record at once.
			self._dense[deleted] = self._dense[last]

			moved_entity = self._dense["entity"][deleted]
			self._sparse[moved_entity] = deleted

		# Clear the unused record.
		self._dense[last] = np.zeros((), dtype=self._dense.dtype)
		self._dense["entity"][last] = NONE
		self._sparse[entity] = NONE

		return True

	def _remove_entity(self, entity: Entity) -> None:
		if entity >= self.sparse_size:
			return

		head = self._sparse[entity]

		if head != NONE:
			self._remove_impl(entity, head)

	def get(self, entity: Entity) -> Tuple[ComponentProxy, ...] | None:
		raise Exception(
			f"Called 'get' on ComponentStorage "
			f"({self.component_cls}, entity {entity})"
		)

	def get_1(self, entity: Entity) -> ComponentProxy | None:
		if entity >= self.sparse_size:
			return None

		head = self._sparse[entity]

		if head == NONE:
			return None

		return ComponentProxy(self, entity, head)

	def _get_rows(self, entities: Array | None = None) -> Array:
		if entities is None:
			ent = self._entities_contained[:self._size]
			idx = np.arange(self._size)

			if self.mult_comp:
				idx = idx[ent != NONE]

			return idx

		ent = np.atleast_1d(entities).astype(int, copy=False)

		if ent.size == 0:
			return ent

		return self._get_rows_impl(ent)

	def _get_rows_impl(self, ent: Array) -> Array:
		# Avoid indexing sparse with entities outside its bounds.
		valid = (ent >= 0) & (ent < self.sparse_size)

		if not np.any(valid):
			return np.empty(0, dtype=int)

		idx = self._sparse[ent[valid]]
		idx = idx[idx != NONE]

		return np.atleast_1d(idx).astype(int, copy=False)

	def get_vector(self, entities: Array | None = None) -> LazyDict:
		"""
		Return component fields for the selected entities.

		A structured array supports field access through:
		    self._dense["field"]
		"""
		return LazyDict(self._dense, self._get_rows(entities))

	def get_full_vector(self, entities: Array | None = None) -> Array:
		rows = self._get_rows(entities)

		return np.stack(
			[self._dense[field][rows] for field in self.fields],
			axis=1,
		)

	def set_vector(
		self,
		entities: Array,
		**value_arrays: Array,
	) -> None:
		rows = self._get_rows(entities)

		for field, values in value_arrays.items():
			if field == "entity":
				continue

			if field not in self.fields:
				raise KeyError(
					f"{field!r} is not a field of {self.component_cls}"
				)

			self._dense[field][rows] = values

	def set_full_vector(
		self,
		entities: Array,
		vector: Array,
	) -> None:
		vector = np.asarray(vector)

		if vector.ndim != 2:
			raise ValueError("vector must be a two-dimensional array")

		if vector.shape[0] != len(self.fields):
			raise ValueError(
				f"Expected {len(self.fields)} columns, "
				f"got {vector.shape[0]}"
			)

		self.set_vector(
			entities,
			**{
				field: vector[column,:]
				for column, field in enumerate(self.fields)
			},
		)

	def query(
		self,
		condition: Callable[..., Array],
		entities: Array | None = None,
	) -> Array:
		"""
		Return entity IDs for which `condition` holds.

		`condition` is a vectorized predicate receiving fields as keyword
		arguments. The internal `entity` field may also be requested:

		    position.query(lambda x, entity: x > entity)
		"""
		if self._size == 0:
			return np.empty(0, dtype=int)

		idx = self._get_rows(entities)

		if idx.size == 0:
			return np.empty(0, dtype=int)

		ent = self._dense["entity"][idx]

		arguments = {
			field: self._dense[field][idx]
			for field in inspect_signature(condition).parameters
		}

		mask = np.asarray(condition(**arguments), dtype=bool)

		if mask.ndim == 0:
			mask = np.full(idx.size, mask, dtype=bool)

		matched = ent[mask]
		return np.atleast_1d(matched).astype(int, copy=False)


class MultiComponentStorage(ComponentStorage):
	"""
	component storage that supports multiple components of the same type per entity
	slightly less performant when you add and remove components randomly
	"""

	def __init__(self,
			 component_cls: TypeVar,
			 capacity: int = 128):

		super().__init__(component_cls, capacity)
		self.mult_comp = True
		self._count : Dict[Entity, int] = {}

	# conequence of the policy used to support multiple components per entity
	def _defragment(self, real_size:int, needed:int) -> None:
		new_cap = max(real_size * 2, needed + real_size - self._size + 32)
		dense_idx = np.flatnonzero(self._entities_contained != NONE)
		new_size = len(dense_idx)

		new_arr = np.zeros(new_cap, dtype=self._dense.dtype)
		new_arr[:new_size] = self._dense[dense_idx]
		self._dense = new_arr
		
		self._entities_contained[new_size:] = NONE
		valid = self._entities_contained != NONE
		sparse_idx = np.flatnonzero(np.r_[True, self._entities_contained[1:] != self._entities_contained[:-1]] & valid)
		self._sparse[self._entities_contained[sparse_idx]] = sparse_idx
		self._size = new_size

	def _add(self, entity:Entity, component:Any) -> None:
		idx = self._size
		count = self._count.get(entity, 0)

		if count == 0:
			self._simpleAdd(entity, component)
			self._set_entity(idx, entity)
			self._count[entity] = 1
			return

		head = self._sparse[entity]
		last = head+count
		newCount = count+1
		needed = idx + newCount
		newLast = idx + count

		if needed >= self.capacity:
			real_size = sum( self._count.values() )

			if real_size > needed * 0.5:
				self._grow_dense(needed)

			else:
				self._defragment(real_size, needed)
				self._add(entity, component)
				return

		self._count[entity] = newCount

		if self._entities_contained[last] == NONE:
			if last == idx:
				self._simpleAdd(entity, component)
			else:
				self._set(last, entity, component)
			return

		# move to end
		self._size = needed
		new_block = slice(idx, newLast)
		old_block = slice(head, last)
		self._dense[new_block] = self._dense[old_block]
		self._entities_contained[old_block] = NONE
		self._set(newLast, entity, component)
		self._set_entity(idx, entity)

	def _remove_entity(self, entity:Entity) -> None:
		rows = self._get_rows(np.array([entity]))
		if rows.size != 0:
			self._sparse[entity] = NONE
			self._entities_contained[rows] = NONE
			del self._count[entity]

	def get_1(self, entity: Entity) -> ComponentProxy|None:
		raise Exception(f"Called 'get_1' on MultiStorage ({self.component_cls}, entity {entity})")

	def get(self, entity: Entity) -> Tuple[ComponentProxy, ...]|None:
		head = int(self._sparse[entity])
		if head == NONE: return None
		count = self._count[entity]
		return tuple( ComponentProxy(self, entity, idx) for idx in range(head, head + count) )

	def _get_rows_impl(self, ent:Array) -> Array:
		ent = np.unique(ent) 
		
		startIdx = self._sparse[ent]
		counts = np.vectorize(self._count.get, otypes=(np.uint16,))(ent,0)

		if ent.size == 1:
			return np.arange(startIdx[0], startIdx[0] + counts[0])
		
		total = counts.sum()
		base_idx = np.repeat(startIdx, counts) # [start0, start0, start1, start1, ...] size total
		single_offsets = np.arange(total) # [0, 1, 2, 3, ...] size total
		cum_counts = np.r_[0, np.cumsum(counts)[:-1]] # [0, c0, c0+c1, ...] size counts
		large_offsets = np.repeat(cum_counts, counts) # [[0] * c0, [c0] * c1, [c0+c1] * c2, ...] size total
		offsets = single_offsets - large_offsets
		
		idx = base_idx + offsets
		return np.atleast_1d(idx).astype(int)

	def _remove_impl(self, entity:Entity, deleted:int|np.integer) -> bool:
		head = self._sparse[entity]
		if head == NONE: return True
		# update count, swap deleted with last
		# NOTE: this is simple and fine BUT I don't like that the components relative position are not kept
		# alternative is displace all the components on the right of the deleted one
		count = self._count[entity]-1
		last = head + count
		if deleted != last:
			self._dense[deleted] = self._dense[last]
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
		self.entity_masks = np.zeros((0,), dtype=np.uint64)
		self._free_entities: List[Entity] = []
		self._next_entity_id : int = 0

	@property
	def count(self) -> int:
		return self._next_entity_id - len(self._free_entities)
	
	@property
	def entity_masks_size(self) -> int:
		return self.entity_masks.shape[0]
	
	def _grow_entity_mask(self, entity:int) -> None:
		entity_masks_size = self.entity_masks_size
		new_cap = max(entity_masks_size * 2, entity+32)
		new_entity_masks = np.zeros(new_cap, dtype=int)
		new_entity_masks[:entity_masks_size] = self.entity_masks[:entity_masks_size]
		self.entity_masks = new_entity_masks
	
	def create_entities(self, quantity: int) -> List[Entity]:
		"""
		Create `quantity` new entity IDs
		NOTE: re-issuing dead entity ids to keep sparse array sizes low
			  however potential bug : a re-issued entity may be mistaken for the previous dead entity
		"""
		out : List[Entity] = []
		len_free = len(self._free_entities)
		if len_free:
			if len_free <= quantity:
				out = self._free_entities
				self._free_entities = []
				quantity -= len_free
			elif quantity == 1:
				return [ self._free_entities.pop() ]
			else:
				out = self._free_entities[-quantity:]
				self._free_entities = self._free_entities[:-quantity]
				return out
		
		current_id = self._next_entity_id
		self._next_entity_id += quantity
		if self.entity_masks_size <= self._next_entity_id:
			self._grow_entity_mask(self._next_entity_id)
		#if quantity == 1: return [ current_id ]
		out.extend( range(current_id, self._next_entity_id) )
		return out
		
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
	
	def register(self, *component_types:TypeVar, allow_same_type_components_per_entity:bool= False, capacity:int=128) -> None:
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

	def __init__(self, store: ComponentStorage, entity: Entity, dense_index: int|np.int64):
		self._store = store
		self._idx = int(dense_index)
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

	def __init__(self, dense:Array, rows:Array):
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

	def keys(self) -> tuple[str, ...]:
		return self._dense.dtype.names or tuple()

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