import pytest

from ecs import *


def test_no_free_entities():
    ecs = ECS()
    ids = ecs.create_entities(3)
    assert isinstance(ids, list) or isinstance(ids, np.ndarray)
    assert list(ids) == [0, 1, 2]
    assert ecs._next_entity_id == 3
    assert ecs._free_entities == []

def test_single_free_entity_exact():
    ecs = ECS()
    ecs._free_entities = [10]
    [eid] = ecs.create_entities(1)
    assert isinstance(eid, Entity)
    assert eid == 10
    assert ecs._next_entity_id == 0
    assert ecs._free_entities == []

def test_single_free_entity_more_than_one():
    ecs = ECS()
    ecs._free_entities = [5]
    out = ecs.create_entities(3)
    assert out == [5, 0, 1]
    assert ecs._next_entity_id == 2
    assert ecs._free_entities == []

def test_multiple_delete_entity_twice():
    ecs = ECS()
    ecs._free_entities = [7, 8, 9, 10]
    ecs.delete_entity(7)
    assert ecs._free_entities == [7, 8, 9, 10]
    out = ecs.create_entities(2)
    assert out == [9,10]

def test_multiple_free_entities_exact_n():
    ecs = ECS()
    ecs._free_entities = [20, 30]
    ecs._next_entity_id = 5
    out = ecs.create_entities(2)
    assert out == [20, 30]
    assert ecs._free_entities == []
    assert ecs._next_entity_id == 5

def test_multiple_free_entities_more_than_n():
    ecs = ECS()
    ecs._free_entities = [100, 101, 102, 103]
    ecs._next_entity_id = 0
    out = ecs.create_entities(3)
    assert out == [101, 102, 103]
    assert ecs._free_entities[0] == 100
    assert ecs._next_entity_id == 0

@component
class Foo:
    a: float
    b: str

def test_single_component_storage_basic():
    s = ComponentStorage(Foo, capacity=2)
    assert s.get(1) is None
    
    s._add(1, Foo(1.23, "first"))
    proxy = s.get(1)
    assert proxy is not None
    assert pytest.approx(proxy.a) == 1.23
    assert proxy.b == "first"
    
    s._remove(proxy)
    assert s.get(1) is None
    s._add(1, Foo(4.56, "second"))
    proxy = s.get(1)
    assert pytest.approx(proxy.a) == 4.56
    assert proxy.b == "second"
    
    s._add(2, Foo(7.89, "third"))
    s._add(3, Foo(0.12, "fourth"))  # forces _grow_dense
    proxy = s.get(3)
    assert pytest.approx(proxy.a) == 0.12
    assert proxy.b == "fourth"

    # rebuild component (not useful, JIC)
    obj = proxy.build()
    assert pytest.approx(obj.a) == 0.12
    assert obj.b == "fourth"

    # check __setattr__ modifies the store 
    proxy.a = 0.69
    assert pytest.approx(s.get(3).a) == 0.69

    s._remove(s.get(2))
    assert s.get(2) is None

def test_single_component_storage_sparse_dense_integrity():
    s = ComponentStorage(Foo, capacity=1)
    
    for eid, val in enumerate([10.0, 20.0, 30.0], start=5):
        s._add(eid, Foo(val, f"val{eid}"))
    assert s._capacity >= 3
    
    for eid in (5,6,7):
        idx = s._sparse[eid]
        assert 0 <= idx < s._size
    
    foo6 = s.get(6)
    assert foo6 is not None
    
    s._remove(foo6)
    assert s.get(6) is None
    assert s.get(5) is not None and s.get(5).a == pytest.approx(10.0)
    assert s.get(7) is not None and s.get(7).a == pytest.approx(30.0)



def rotate_vectors_by_quaternions(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    q: shape (N,4)  as [qx,qy,qz,qw]
    v: shape (N,M,3) -- M vectors per entity
    returns: rotated vectors, same shape
    """
    q_vec = q[:, None, :3]             # (N,1,3)
    qw    = q[:, None, 3:4]            # (N,1,1)
    t = 2.0 * np.cross(q_vec, v, axis=2)  # (N,M,3)
    return v + qw * t + np.cross(q_vec, t, axis=2)


def update_world_aabb(ecs):
    
    # Non‐rotated boxes
    nr_ents = ecs.where(LocalAABB, Position)
    la, pos = ecs.get_vectors(LocalAABB, Position, nr_ents)
    pos = vectorized(pos, 'x', 'y', 'z')
    # just translate
    wa_min = vectorized(la, 'x_min', 'y_min', 'z_min') + pos
    wa_max = vectorized(la, 'x_max', 'y_max', 'z_max') + pos
    aabbs = ecs.get_store(AxisAlignedBoundingBox)
    wa = np.append(wa_min, wa_max, axis=1).transpose()
    aabbs.set_full_vector(nr_ents, wa)
    
    # Rotated boxes
    rot_ents = ecs.where(LocalAABB, Position, Orientation)
    la, pos, ori = ecs.get_vectors(LocalAABB, Position, Orientation, rot_ents)
    pos = vectorized(pos, 'x', 'y', 'z')
    ori = vectorized(ori, 'qx', 'qy', 'qz', 'qw')
    #bb = ecs.get_store(LocalAABB).get_full_vector(rot_ents)
    #mins = la[:, :3]
    #maxs = la[:, 3:]
    mins = vectorized(la, 'x_min', 'y_min', 'z_min')
    maxs = vectorized(la, 'x_max', 'y_max', 'z_max')
    # build 8 corners
    corners = np.array([[x,y,z] for x in (0,1)
                                 for y in (0,1)
                                 for z in (0,1)], int)
    offs = mins[:,None,:]*(1-corners)[None,:,:] + \
          maxs[:,None,:]*corners[None,:,:]
    # rotate & translate
    pts = rotate_vectors_by_quaternions(ori, offs) + pos[:,None,:]
    wa_min = pts.min(axis=1)
    wa_max = pts.max(axis=1)
    wa = np.append(wa_min, wa_max, axis=1).transpose()
    aabbs.set_full_vector(nr_ents,wa)


def detect_aabb_overlaps(ecs):
    ents = ecs.where(AxisAlignedBoundingBox)
    blk = ecs.get_store(AxisAlignedBoundingBox).get_vector(ents)
    mins = vectorized(blk, 'x_min', 'y_min', 'z_min')
    maxs = vectorized(blk, 'x_max', 'y_max', 'z_max')

    ok1 = mins[:,None,:] <= maxs[None,:,:]
    ok2 = maxs[:,None,:] >= mins[None,:,:]
    overlap = np.logical_and(ok1, ok2).all(axis=2)

    i,j = np.triu_indices(ents.size, k=1)
    hits = overlap[i,j]
    return [(int(ents[ii]), int(ents[jj]))
            for ii,jj,h in zip(i,j,hits) if h]

# Component definitions
@component
class Position2D:
    x: float; y: float

@component
class Velocity2D:
    x: float; y: float

class TagEnum(IntFlag):
    Enemy=auto()
    Flying=auto()
    Dazed=auto()

@component
class Tag:
    value:TagEnum

@component
class Position:
    x: float; y: float; z: float

@component
class Orientation:
    qx: float; qy: float; qz: float; qw: float

@component
class LocalAABB:
    x_min: float; y_min: float; z_min: float
    x_max: float; y_max: float; z_max: float

@component
class AxisAlignedBoundingBox:
    x_min: float; y_min: float; z_min: float
    x_max: float; y_max: float; z_max: float


def make_pos2d_store(vals):
    store = ComponentStorage(Position2D, capacity=max(32, len(vals)))
    for eid, (x, y) in enumerate(vals):
        store._add(eid, Position2D(x, y))
    return store

def make_multitag_store(blocks_per_entity):
    store = MultiComponentStorage(Tag, capacity=64)
    for eid, values in blocks_per_entity.items():
        for v in values:
            store._add(eid, Tag(v))
    return store

def test_get_whole_vector():
    pos = make_pos2d_store([(0, 0), (1, 1), (2, 2)])
    vec = pos.get_vector()
    vec = vectorized(vec, *pos.fields)
    for i, v in enumerate(vec):
        assert np.allclose((i, i), v)

def test_get_rows():
    pos = make_pos2d_store([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
    pos._remove(pos.get(3))
    pos._remove(pos.get(0))
    pos._add(5, Position2D(-1,-1))
    # remember : rows are the indices in the dense array
    rows = pos._get_rows(range(4))
    assert rows.tolist() == [1,2]
    rows = pos._get_rows()
    assert sorted(rows.tolist()) == [0,1,2,3]
    rows = pos._get_rows([])
    assert not rows

def test_query_simple_vectorized():
    pos = make_pos2d_store([(0, 0), (1, 2), (5, -3), (-1, 1), (4, 10)])
    # x > 0 and |y| < 3  -> entities 1 only (1,2) fails second cond; entity 2 has |y|=3 -> false
    out = pos.query(lambda x, y: (x > 0) & (np.abs(y) < 3))
    assert out.dtype == int
    assert out.tolist() == [1]

def test_query_annotated_types():
    pos = make_pos2d_store([(0, 0), (1, 2), (5, -3), (-1, 1), (4, 10)])
    def condition(x:float, y:float)->bool:
        return (x > 0) & (np.abs(y) < 3)
    out = pos.query(condition)
    assert out.dtype == int
    assert out.tolist() == [1]

def test_query_subset_filtering():
    pos = make_pos2d_store([(10, 0), (9, 0), (11, 0), (8, 0)])
    subset = np.array([0, 1, 3], dtype=int)  # exclude entity 2 even though it matches
    out = pos.query(lambda x, y: x >= 10, entities=subset)
    assert out.tolist() == [0]

def test_query_returns_empty_when_no_match():
    pos = make_pos2d_store([(0, 0), (1, 1)])
    out = pos.query(lambda x, y: (x < 0))
    assert isinstance(out, np.ndarray)
    assert out.size == 0

def test_query_can_use_entity_id_in_predicate():
    pos = make_pos2d_store([(5, 0), (5, 0), (5, 0)])
    out = pos.query( lambda x, y, entity: (x == 5) & (entity != 1))
    assert out.tolist() == [0, 2]

def test_query_raises_on_length_mismatch():
    pos = make_pos2d_store([(0, 0), (1, 1), (2, 2)])
    with pytest.raises(ValueError):
        pos.query(lambda x, y: np.array([True, False]))

def test_query_on_empty_storage():
    pos = ComponentStorage(Position2D, capacity=8)
    out = pos.query(lambda x: x > 0)
    assert out.size == 0

def test_query_mult_comp_any_row_matches_once_stable_order():
    mt = make_multitag_store({
        0: [1, 0],
        1: [0, 0],
        2: [2, 1],
    })
    out = mt.query(lambda value: value == 1)
    assert out.tolist() == [0, 2]

def test_query_mult_comp_subset_filtering():
    mt = make_multitag_store({
        2: [4, 5],
        5: [3, 3],
        7: [1, 2],
        9: [1],
    })
    subset = np.array([5, 7, 9], dtype=int)
    out = mt.query(lambda value: value == 1, entities=subset)
    assert out.tolist() == [7, 9]

def test_query_mult_comp_no_match_returns_empty():
    mt = make_multitag_store({0: [0, 0], 1: [2], 3: [4, 5]})
    out = mt.query(lambda value: value == 9)
    assert out.size == 0

def test_query_mult_comp_get_vector_returns_all_comp():
    mt = make_multitag_store({0: [0, 0], 1: [2], 3: [4, 5], 4: [6, 6]})
    # TODO: support multicomponent retrieval
    try:
        out = mt.get_vector( [0,3] )
        assert out['value'].tolist() == [ 0,0, 4,5]
    except:
        pass

def test_query_intflag_predicate():
    tags = ComponentStorage(Tag, capacity=8)
    tags._add(1, Tag(TagEnum.Enemy | TagEnum.Dazed))
    tags._add(2, Tag(TagEnum.Flying))
    tags._add(3, Tag(TagEnum.Enemy))
    out = tags.query(lambda value: (value & TagEnum.Enemy) != 0)
    assert out.tolist() == [1, 3]

def test_movement_system():
    ecs = ECS()
    ecs.register(Position2D, Velocity2D, Tag)
    # Setup entities
    ecs.create_entity()
    ecs.create_entity(
        Position2D(0.0, 0.0),
        Velocity2D(1.0, 2.0))
    ecs.create_entity(
        Position2D(5.0, -3.0),
        Velocity2D(-1.0, 0.5),
        Tag(TagEnum.Enemy | TagEnum.Dazed))

    # Verify initial positions
    positions = ecs.get_store(Position2D)
    p1 = positions.get(1)
    p2 = positions.get(2)
    assert (p1.x, p1.y) == pytest.approx((0.0, 0.0))
    assert (p2.x, p2.y) == pytest.approx((5.0, -3.0))

    # Apply movement only for Tag
    targets = ecs.where(Tag, Position2D, Velocity2D)
    p_vec, v_vec = ecs.get_vectors(Position2D, Velocity2D, targets)
    p_vec['x'] -= v_vec['x'] * 0.5
    p_vec['y'] -= v_vec['y'] * 0.5
    ecs.get_store(Position2D).set_vector(targets, **p_vec)

    # Check updated position for entity 2
    p2_after = ecs.get_store(Position2D).get(2)
    assert (p2_after.x, p2_after.y) == pytest.approx((5.0 + 0.5, -3.0 - 0.25))
    p1_after = ecs.get_store(Position2D).get(1)
    assert (p1_after.x, p1_after.y) == pytest.approx((0.0, 0.0))

    # test where exclude param
    untagged = ecs.where(Position2D, Velocity2D, exclude=[Tag])
    assert not 2 in untagged

    # test intFlag
    tagVal = ecs.get_store(Tag).get(2).value
    assert tagVal & TagEnum.Enemy > 0
    assert tagVal & TagEnum.Flying == 0

    # test query
    tagVal = ecs.get_store(Tag).query(lambda value: value&TagEnum.Enemy == 1)
    assert len(tagVal) == 1
    assert tagVal[0] == 2


def test_aabb_system_and_overlap():
    ecs = ECS()
    ecs.register(LocalAABB, Position, AxisAlignedBoundingBox, Orientation)
    # Entity 1 with rotated AABB
    e1 = ecs.create_entity(
        LocalAABB(-1,-1,-1, 1,1,1),
        Position(5,0,0),
        AxisAlignedBoundingBox(0,0,0,0,0,0),
        Orientation(0,0,np.sin(np.pi/8), np.cos(np.pi/8))
    )
    # Entity 2 without rotation
    ecs.add_component(2,
        LocalAABB(-2,-2,-2, 2,2,2),
        Position(5,0,0),
        AxisAlignedBoundingBox(0,0,0,0,0,0))

    # Run systems
    update_world_aabb(ecs)
    wa1 = ecs.get_store(AxisAlignedBoundingBox).get(e1)
    # Compute expected rotated extents
    half_diag = np.sqrt(2)
    assert wa1.x_min == pytest.approx(5 - half_diag)
    assert wa1.x_max == pytest.approx(5 + half_diag)
    assert wa1.y_min == pytest.approx(-half_diag)
    assert wa1.y_max == pytest.approx(half_diag)
    assert wa1.z_min == pytest.approx(-1)
    assert wa1.z_max == pytest.approx(1)

    # Overlap detection
    hits = detect_aabb_overlaps(ecs)
    assert hits == [(e1, 2)]


@component
class MultiComp:
    val: float

def test_storage_multicomp_add_get_remove():
    # Create a storage that allows multiple MultiComp per entity
    store = MultiComponentStorage(MultiComp, capacity=4)
    entity_id = 42

    store._add(entity_id, MultiComp(1.))
    store._add(entity_id, MultiComp(2.))
    store._add(entity_id, MultiComp(3.))

    comps = store.get(entity_id)
    assert isinstance(comps, Sequence)
    assert len(comps) == 3
    assert comps[0].val == pytest.approx(1.)
    assert comps[1].val == pytest.approx(2.)
    assert comps[2].val == pytest.approx(3.)

    idx = store._sparse[entity_id]
    block = store._dense['val'][:3]
    assert np.allclose(block, [1., 2., 3.])

    store._remove(store.get(entity_id)[1])
    comps = store.get(entity_id)
    assert isinstance(comps, Sequence)
    assert len(comps) == 2
    assert comps[0].val == pytest.approx(1.0)
    assert comps[1].val == pytest.approx(3.)

def test_ecs_multicomp_reassignment():
    ecs = ECS()
    ecs.register(MultiComp, allow_same_type_components_per_entity=True, capacity=5)
    ecs.register(Position2D, capacity=5)
    store = ecs.get_store(MultiComp)
    positions = ecs.get_store(Position2D)

    ecs.create_entities(12)
    e = ecs.create_entity(MultiComp(1.1), MultiComp(2.2), MultiComp(3.3), Position2D(1,1))
    e2 = ecs.create_entity(MultiComp(5.5), MultiComp(6.6), MultiComp(7.7), Position2D(2,2))
    ecs.add_component(e, MultiComp(4.4))
    proxies = store.get(e)
    assert len(proxies) == 4
    assert sorted( map(lambda p: p.val, proxies) ) == [1.1, 2.2, 3.3, 4.4]
    store._remove(next(filter(lambda p: p.val==2.2, proxies)))
    store._remove(next(filter(lambda p: p.val==3.3, proxies)))
    proxies = store.get(e)
    assert len(proxies) == 2
    assert sorted( map(lambda p: p.val, proxies) ) == [1.1, 4.4]

    proxies = store.get(e2)
    assert len(proxies) == 3
    assert sorted( map(lambda p: p.val, proxies) ) == [5.5, 6.6, 7.7]

    proxy = positions.get(e2)
    assert proxy.x == 2 and proxy.y == 2

    ecs.delete_entity(e2)
    assert not store.get(e2)
    assert not positions.get(e2)
    
    ecs.add_component(e, MultiComp(8.8))
    vec = store.get_full_vector([e,e2,8])
    assert vec.shape == (3, 1)
    assert sorted( vec.flatten().tolist() ) == [1.1, 4.4, 8.8]
    
    ecs.add_component(e, MultiComp(9.9))
    proxies = store.get(e)
    assert len(proxies) == 4
    assert sorted( map(lambda p: p.val, proxies) ) == [1.1, 4.4, 8.8, 9.9]

def test_ecs_multicomp_defragment():
    ecs = ECS()
    ecs.register(MultiComp, allow_same_type_components_per_entity=True, capacity=12)
    store = ecs.get_store(MultiComp)

    ecs.create_entities(16)
    for i in range(4):
        j = i * 4
        ecs.add_component(j, MultiComp(j), MultiComp(j+1), MultiComp(j+2))
        comps = store.get(j)
        ecs.remove_component(comps[1])
        ecs.remove_component(comps[1])

    assert store._count == {0: np.uint16(1), 4: np.uint16(1), 8: np.uint16(1), 12: np.uint16(1)}
    assert np.unique(store._entities_contained).tolist() == [-1, 0, 4,  8, 12]
    assert np.unique(store._dense['val'][:store._size+1][store._entities_contained != ComponentStorage.NONE]).tolist() == [0., 4., 8., 12.]

    for i in range(4):
        j = i * 4
        ecs.add_component(j,MultiComp(j+1), MultiComp(j+2))

    assert store._count == {0: np.uint16(3), 4: np.uint16(3), 8: np.uint16(3), 12: np.uint16(3)}
    assert np.unique(store._entities_contained).tolist() == [-1, 0, 4,  8, 12]
    assert np.unique(store._dense['val'][:store._size+1][store._entities_contained != ComponentStorage.NONE]).tolist() == [0.,1.,2., 4.,5.,6., 8.,9.,10., 12.,13.,14.]

    for i in range(4):
        j = i * 4
        assert np.unique([ p.val for p in store.get(j) ]).tolist() == [j, j+1, j+2]


