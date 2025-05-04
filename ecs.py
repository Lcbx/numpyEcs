import numpy as np
import numbers
import inspect
from typing import Type, Dict, Callable, Any, List, Tuple
from dataclasses import dataclass

component = dataclass

ComponentProxy = lambda s,e: None

class ComponentStorage:
    """
    Stores one component type using:
      - preallocated 2D numpy array for all float fields (_nums)
      - parallel Python lists for non-numeric fields (_objs)
      - numpy arrays for sparse (entity -> index) and dense (index -> entity)
    """
    def __init__(self, component_cls: Type, initial_capacity: int = 128):
        self.component_cls = component_cls
        ann = getattr(component_cls, '__annotations__', {})
        # Only float annotations are treated as numeric
        self._num_fields = [n for n,t in ann.items() if t is float]
        self._obj_fields = [n for n in ann if n not in self._num_fields]
        # Preallocate buffers
        self._capacity = initial_capacity
        self._size = 0
        self._nums = np.zeros((self._capacity, len(self._num_fields)), dtype=float)
        self._objs: Dict[str, List[Any]] = {f: [] for f in self._obj_fields}
        self._dense = np.zeros((self._capacity,), dtype=int)
        self.sparse = np.full((self._capacity,), -1, dtype=int)

    @property
    def dense(self) -> np.ndarray:
        return self._dense[:self._size]
    
    @property
    def sparse_size(self) -> int:
        return self.sparse.shape[0]
        
    def _grow_sparse(self, entity : int):
        sparse_size = self.sparse_size
        new_cap = max(sparse_size * 2, entity+32)
        new_sparse = np.full((new_cap,), -1, dtype=int)
        new_sparse[:sparse_size] = self.sparse[:sparse_size]
        self.sparse = new_sparse

    def _grow_dense(self):
        new_cap = self._capacity * 2
        # expand numeric buffer
        new_nums = np.zeros((new_cap, len(self._num_fields)), dtype=float)
        new_nums[:self._size] = self._nums[:self._size]
        self._nums = new_nums
        # expand dense buffer
        new_dense = np.zeros((new_cap,), dtype=int)
        new_dense[:self._size] = self._dense[:self._size]
        self._dense = new_dense
        self._capacity = new_cap

    def add(self, entity: int, component: Any) -> None:
        # grow if needed
        if entity >= self.sparse_size:
            self._grow_sparse(entity)
        if self._size >= self._capacity:
            self._grow_dense()
        idx = self._size
        # write sparse/dense
        self.sparse[entity] = idx
        self._dense[idx] = entity
        # store numeric fields
        row = [getattr(component, f) for f in self._num_fields]
        self._nums[idx] = np.array(row, dtype=float)
        # store object fields
        for f in self._obj_fields:
            self._objs[f].append(getattr(component, f))
        self._size += 1

    def remove(self, entity: int) -> None:
        if entity >= self.sparse_size or self.sparse[entity] == -1:
            return
        idx = int(self.sparse[entity])
        last = self._size - 1
        last_ent = int(self._dense[last])
        # swap sparse/dense
        self._dense[idx] = last_ent
        self.sparse[last_ent] = idx
        self.sparse[entity] = -1
        # swap numeric row
        self._nums[idx] = self._nums[last]
        # swap object lists
        for f, lst in self._objs.items():
            lst[idx] = lst[last]
            lst.pop()
        self._size -= 1

    def get(self, entity: int) -> ComponentProxy:
        return ComponentProxy(self, entity)

class ECS:
    """
    Fully optimized ECS with:
      - float-only numeric arrays for vectorized ops
      - preallocated component storage to avoid frequent reallocation
      - per-entity bitmask for fast component queries
      - entity ID recycling via free list
    """
    def __init__(self):
        self._stores: Dict[Type, ComponentStorage] = {}
        self._comp_bits: Dict[Type, int] = {}
        self._next_bit = 0
        self.entity_masks = np.zeros((0,), dtype=np.uint64)
        self._free_entities: List[int] = []
        self._next_entity_id = 0
    
    @property
    def entity_masks_size(self) -> int:
        return self.entity_masks.shape[0]
    
    def _grow_entity_mask(self, entity : int):
        entity_masks_size = self.entity_masks_size
        new_cap = max(entity_masks_size * 2, entity+32)
        new_entity_masks = np.full((new_cap,), 0, dtype=int)
        new_entity_masks[:entity_masks_size] = self.entity_masks[:entity_masks_size]
        self.entity_masks = new_entity_masks

    def create_entities(self, n: int) -> np.ndarray:
        """
        Create `n` new entity IDs, reusing deleted ones when possible.
        """
        out = []
        # reuse free entities first
        while self._free_entities and len(out) < n:
            out.append(self._free_entities.pop())
        # then allocate new IDs
        while len(out) < n:
            eid = self._next_entity_id
            self._next_entity_id += 1
            out.append(eid)
        return out
        
    def create_entity(self, *components) -> int:
        [entity] = self.create_entities(1)
        for comp in components:
            self.add_component(entity, comp)
        return entity
    
    def delete_entity(self, entity: int) -> None:
        for comp_cls in list(self._comp_bits):
            if self.has_component(entity, comp_cls):
                self.remove_component(entity, comp_cls)
        # add to free list
        self._free_entities.append(entity)

    def add_component(self, entity: int, component: Any) -> None:
        cls = type(component)
        if cls not in self._comp_bits:
            self._comp_bits[cls] = 1 << self._next_bit
            self._next_bit += 1
        bit = self._comp_bits[cls]
        self._grow_entity_mask(entity)
        self.entity_masks[entity] |= bit
        if cls not in self._stores:
            self._stores[cls] = ComponentStorage(cls)
        self._stores[cls].add(entity, component)

    def remove_component(self, entity: int, comp_cls: Type) -> None:
        bit = self._comp_bits.get(comp_cls, 0)
        if entity < entity_masks_size:
            self.entity_masks[entity] &= ~bit
        store = self._stores.get(comp_cls)
        if store:
            store.remove(entity)

    def get_component(self, entity: int, comp_cls: Type) -> ComponentProxy:
        store = self._stores.get(comp_cls)
        return ComponentProxy(store, entity)

    def get_vector(self, comp_cls: Type, entities: np.ndarray) -> np.ndarray:
        """ returns numeric block of component type comp_cls associated with entities """
        store = self._stores[comp_cls]
        es = np.atleast_1d(entities).astype(int)
        idx = store.sparse[es]
        return store._nums[idx]

    def get_vectors(self, *args):
        """
        returns numeric blocks of N component types associated with entities

        Usage:
            ecs.get_blocks(C1, C2, …, entities)

        - C1…Cn are component classes
        - entities: 1D array of entity IDs
        """
        *comp_clss, entities = args
        es = np.atleast_1d(entities).astype(int)
        return [self.get_vector(C, es) for C in comp_clss]
    
    def set_vector(self, comp_cls: Type, entities: np.ndarray, vectors: np.ndarray) -> None:
        """
        Overwrite the numeric rows of `comp_cls` at `entities` with `vectors`.
        """
        store = self._stores[comp_cls]
        es = np.atleast_1d(entities).astype(int)
        idx = store.sparse[es]
        store._nums[idx] = vectors


    def where(self, *args) -> np.ndarray:
        """
        Usage:
          ecs.where(C1, C2, ..., predicate_fn)   # filter by predicate
          ecs.where(C1, C2, ...)                 # no predicate → all entities with those comps

        If the last argument is a callable, it’s used as a filter on the unpacked components;
        otherwise every entity that has all of the listed component types is returned.
        """

        # detect whether the last arg is a predicate or a component class
        last = args[-1]
        if callable(last) and not inspect.isclass(last) and len(args) >= 2:
            *comp_clss, predicate = args
        else:
            comp_clss = args
            predicate = None
        
        mask = 0
        for cls in comp_clss:
            mask |= self._comp_bits.get(cls, 0)
        if mask == 0:
            return []
        
        ents = np.nonzero((self.entity_masks & mask) == mask)[0]
        
        if predicate is None:
            return ents

        # apply predicate
        flags = [predicate(*[self.get_component(e, C).build() for C in comp_clss]) for e in ents]
        return ents[np.array(flags, dtype=bool)]    
        
class ComponentProxy:
    """
    Lazy proxy for one component instance on one entity.
    Wraps the ComponentStorage for fast field access.
    """
    def __init__(self, store, entity: int):
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_entity", entity)

    def __getattr__(self, name):
        store = self._store
        idx = int(store.sparse[self._entity])
        if name in store._num_fields:
            j = store._num_fields.index(name)
            return store._nums[idx, j]
        elif name in store._obj_fields:
            return store._objs[name][idx]

    def __setattr__(self, name, value):
        store = self._store
        idx = int(store.sparse[self._entity])
        if name in store._num_fields:
            j = store._num_fields.index(name)
            store._nums[idx, j] = float(value)
        elif name in store._obj_fields:
            store._objs[name][idx] = value

    def __repr__(self):
        store = self._store
        idx = int(store.sparse[self._entity])
        data = {
            f: getattr(self, f)
            for f in store._num_fields + store._obj_fields
        }
        return f"<{store.component_cls.__name__}Proxy e={self._entity} {data}>"
    
    def build(self) -> Any:
        """
        Reconstruct and return a full instance of the component.
        """
        store = self._store
        idx = int(store.sparse[self._entity])
        kwargs: Dict[str, Any] = {}
        # numeric fields
        for j, f in enumerate(store._num_fields):
            kwargs[f] = store._nums[idx, j]
        # object fields
        for f in store._obj_fields:
            kwargs[f] = store._objs[f][idx]
        return store.component_cls(**kwargs)