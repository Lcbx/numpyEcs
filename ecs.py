import numpy as np
import inspect
from typing import Type, Dict, Callable, Any, List, Tuple
from dataclasses import dataclass

Entity = int
component = dataclass

# forward declaration
ComponentProxy = lambda s,e: None

class ComponentStorage:
    """
    Stores one component type using:
      - preallocated 2D numpy array for all float fields (_nums)
      - parallel Python lists for non-numeric fields (_objs)
      - numpy arrays for _sparse (entity -> index) and dense (index -> entity)
    """
    
    # can either be in _sparse (dense index pointing to nowhere)
    #  or in dense to indeicate an empty spot (when multi-component flag is set)
    NONE = -1
    
    def __init__(self, component_cls: Type, capacity: int = 128, mult_comp: bool = False):
        self.mult_comp = mult_comp
        self.component_cls = component_cls
        ann = getattr(component_cls, '__annotations__', {})
        # Only float annotations are treated as numeric
        self._num_fields = [n for n,t in ann.items() if t is float]
        self._obj_fields = [n for n in ann if n not in self._num_fields]
        # Preallocate buffers
        self._capacity = capacity
        self._size = 0
        self._nums = np.zeros((self._capacity, len(self._num_fields)), dtype=float)
        self._objs: Dict[str, List[Any]] = {f: [] for f in self._obj_fields}
        self._dense = np.zeros((self._capacity,), dtype=int)
        self._sparse = np.full((self._capacity,), ComponentStorage.NONE, dtype=int)

    @property
    def dense(self) -> np.ndarray:
        return self._dense[:self._size]
    
    @property
    def _sparse_size(self) -> int:
        return self._sparse.shape[0]
        
    def _grow__sparse(self, entity : Entity):
        _sparse_size = self._sparse_size
        new_cap = max(_sparse_size * 2, entity+32)
        new__sparse = np.full((new_cap,), ComponentStorage.NONE, dtype=int)
        new__sparse[:_sparse_size] = self._sparse[:_sparse_size]
        self._sparse = new__sparse
    
    def _grow_dense(self, needed_size:int=0):
        
        # expand
        new_cap = max(self._size * 2, needed_size+32)
        new_nums = np.zeros((new_cap, len(self._num_fields)), dtype=float)
        new_dense = np.zeros((new_cap,), dtype=int)
        
        if self.mult_comp:
           # Rebuild compacted
           valid = np.nonzero(self._dense[:self._size] != ComponentStorage.NONE)[0]
           self._size = valid.size
           new_nums[:self._size] = self._nums[valid]
           new_dense[:self._size] = self._dense[valid]
           for f, lst in self._objs.items():
               self._objs[f] = [lst[i] for i in valid]
           for new_idx, ent in enumerate(self._dense):
               if self._dense[new_idx-1] != ent:
                   self._sparse[ent] = new_idx
        else:
            new_nums[:self._size] = self._nums[:self._size]
            new_dense[:self._size] = self._dense[:self._size]
        
        self._nums = new_nums
        self._dense = new_dense
        self._capacity = new_cap
    
    def _add(self, entity: Entity, component: Any) -> None:
        if entity >= self._sparse_size:
            self._grow__sparse(entity)
        if self._size >= self._capacity:
            self._grow_dense()
        
        head = self._sparse[entity]
        if not self.mult_comp or head == ComponentStorage.NONE:
            idx = self._size
            self._sparse[entity] = idx
            self._size += 1
        else:
            idx = head
            while (idx < self._size and self._dense[idx] == entity): idx += 1
            
            if self._dense[idx] == ComponentStorage.NONE: pass
            elif idx == self._size:
                self._size += 1
            else:
                self._relocate_to_end(entity, idx - head)
                idx = self._size
                self._size += 1
        
        # write the new component into slot `idx`
        self._set(idx, entity, component)
    
    def _set(self, denseIndex:int, entity:Entity, component : Any) -> None:
        self._dense[denseIndex] = entity
        compType = type(component)
        if compType == self.component_cls:
            for j, f in enumerate(self._num_fields):
                self._nums[denseIndex, j] = float(getattr(component, f))
            for f in self._obj_fields:
                value = getattr(component, f)
                if denseIndex < len(self._objs[f]):
                    self._objs[f][denseIndex] = value
                else:
                    self._objs[f].append(value)
            return
        elif isinstance(component, ComponentProxy):
            proxy = component
            src_store = proxy._store
            src_idx   = proxy._denseIndex
            self._nums[denseIndex, :] = src_store._nums[src_idx, :]
            for f in self._obj_fields:
                value = src_store._objs[f][src_idx]
                if denseIndex < len(self._objs[f]):
                    self._objs[f][denseIndex] = value
                else:
                    self._objs[f].append(value)
            return
        raise ValueError(f"Component type mismatch in _set : {compType} instead of {self.component_cls}")
     
    def _relocate_to_end(self, entity: Entity, length: int):
        """
        Move the contiguous block of `length` slots for `entity` to the end
        of our dense/_nums buffers, leaving holes behind.
        Used in multi-component add.
        """
        
        #proxies = self.get(entity)
        #print(f"relocate entity {entity}, length {length}, size {self._size}, proxies {proxies}")
        needed = self._size + length
        if needed > self._capacity: self._grow_dense(needed)
        
        head = self._sparse[entity]
        tail = head + length
        
        proxies = self.get(entity)
        
        # NOTE: if before _grow_dense(), this would delete the data
        self._dense[head:tail] = ComponentStorage.NONE
        self._sparse[head:tail] = ComponentStorage.NONE
        
        old_size = self._size
        self._sparse[entity] = old_size
        self._size += length
        for i, comp in enumerate(proxies):
            self._set(old_size + i, entity, comp)

    # TODO: multi-components are handled with ComponentProxies ;
    # we sould use those for component deletions
    def _remove(self, entity: Entity, which:int=0) -> None:
        if entity >= self._sparse_size:
            return
        
        idx = int(self._sparse[entity])
        
        if self.mult_comp:
            comps = self.get(entity)
            idx = comps[which]._denseIndex
            self._dense[idx] = ComponentStorage.NONE
            if which == 0:
                while self._dense[idx] == ComponentStorage.NONE: idx +=1
                self._sparse[entity] = idx
        else:
            self._size -= 1
            self._sparse[entity] = ComponentStorage.NONE
            
            last = self._size
            if last == idx: return
            last_ent = int(self._dense[last])
            self._set(idx, last_ent, self.get(last_ent))
            

    def get(self, entity: Entity) -> Any:
        """
        If mult_comp=False: return a single proxy or None.
        If mult_comp=True: return a list of proxies (possibly empty).
        """
        head = self._sparse[entity]
        if head == ComponentStorage.NONE:
            return [] if self.mult_comp else None
        
        if not self.mult_comp: return ComponentProxy(self, entity, head)
        
        #print(f"get entity {entity}, index {head}, dense {self._dense[head]}, size {self._size}")
        
        proxies = []
        while head < self._size and self._dense[head] == entity:
            proxies.append(ComponentProxy(self, entity, head))
            head += 1
            while self._dense[head] == ComponentStorage.NONE: head +=1
        
        return proxies


    def get_vector(self, entities: np.ndarray = None) -> np.ndarray:
        """ returns numeric block of component associated with entities """
        if entities is None: return self._nums
        es = np.atleast_1d(entities).astype(int)
        idx = self._sparse[es]
        return self._nums[idx]


    def set_vector(self, entities: np.ndarray, vector: np.ndarray) -> None:
        """ Overwrites the numeric rows at `entities` with `vectors` """
        es = np.atleast_1d(entities).astype(int)
        idx = self._sparse[es]
        self._nums[idx] = vector


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
        self._free_entities: List[Entity] = []
        self._next_entity_id = 0
    
    @property
    def entity_masks_size(self) -> int:
        return self.entity_masks.shape[0]
    
    def _grow_entity_mask(self, entity : Entity):
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
        len_free = len(self._free_entities)
        if len_free:
            if len_free <= n:
                out = self._free_entities
                self._free_entities = []
                n -= len_free
            elif n == 1:
                return self._free_entities.pop()
            else:
                out = self._free_entities[-n:]
                self._free_entities[:-n]
                return out
        
        current_id = self._next_entity_id
        self._next_entity_id += n
        if self.entity_masks_size <= self._next_entity_id:
            self._grow_entity_mask(self._next_entity_id)
        out.extend( range(current_id, self._next_entity_id) )
        return out
        
    def create_entity(self, *components) -> int:
        [entity] = self.create_entities(1)
        self.add_component(entity, *components)
        return entity
    
    def delete_entity(self, entity: Entity) -> None:
        if entity in self._free_entities: return
        for comp_cls in list(self._comp_bits):
            if self.has_component(entity, comp_cls):
                self.remove_component(entity, comp_cls)
        self._free_entities.append(entity)
    
    def register(self, *component_types, allow_same_type_components_per_entity : bool = False, initial_capacity=128):
        for cls in component_types:
            if cls not in self._comp_bits:
                self._comp_bits[cls] = 1 << self._next_bit
                self._next_bit += 1
            if cls not in self._stores:
                self._stores[cls] = ComponentStorage(cls,
                    mult_comp=allow_same_type_components_per_entity,
                    capacity=128)
    
    def allow_multiple_components_per_entity(self, comp_cls: Type) -> None:
        store = self._stores.get(comp_cls)
        store.mult_comp = True
    
    def add_component(self, entity: Entity, *components) -> None:
        bits = self.entity_masks[entity]
        for comp in components:
            cls = type(comp)
            bit = self._comp_bits[cls]
            bits |= bit
            self._stores[cls]._add(entity, comp)
        self.entity_masks[entity] = bits

    def remove_component(self, entity: Entity, comp_cls: Type) -> None:
        bit = self._comp_bits.get(comp_cls, 0)
        if entity < entity_masks_size:
            self.entity_masks[entity] &= ~bit
        store = self._stores.get(comp_cls)
        if store:
            store._remove(entity)

    def get_store(self, comp_cls: Type) -> ComponentStorage:
        return self._stores.get(comp_cls)

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
        return [self._stores.get(cls).get_vector(es) for cls in comp_clss]

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
        
        if predicate is None: return ents

        # apply predicate
        stores = tuple([self.get_store(cls) for cls in comp_clss])
        flags = [predicate(*[s.get(e) for s in stores]) for e in ents]
        return ents[np.array(flags, dtype=bool)]    


class ComponentProxy:
    """
    Lazy proxy for one component instance.
    Wraps ComponentStorage and stores the dense‐array index directly.
    """
    def __init__(self, store : ComponentStorage, entity : Entity, denseIndex: int):
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_denseIndex", denseIndex)

    def __getattr__(self, name):
        store = self._store
        idx   = self._denseIndex
        if name in store._num_fields:
            j = store._num_fields.index(name)
            return store._nums[idx, j]
        if name in store._obj_fields:
            return store._objs[name][idx]
        raise AttributeError(f"{store.component_cls.__name__} has no field {name}")

    def __setattr__(self, name, value):
        store = self._store
        idx   = self._denseIndex
        if name in store._num_fields:
            j = store._num_fields.index(name)
            store._nums[idx, j] = float(value)
            return
        if name in store._obj_fields:
            store._objs[name][idx] = value
            return
        raise AttributeError(f"{store.component_cls.__name__} has no field {name}")

    def __repr__(self):
        store = self._store
        idx   = self._denseIndex
        data = {f: getattr(self, f)
                for f in store._num_fields + store._obj_fields}
        return f"<{store.component_cls.__name__}Proxy e={self._entity} {data}>"

    def build(self) -> Any:
        """
        Reconstruct and return a full instance of the component.
        """
        store = self._store
        idx   = self._denseIndex
        kwargs = {f: store._nums[idx, j]
                  for j, f in enumerate(store._num_fields)}
        for f in store._obj_fields:
            kwargs[f] = store._objs[f][idx]
        return store.component_cls(**kwargs)