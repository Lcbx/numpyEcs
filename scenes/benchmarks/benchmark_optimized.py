from scenes.benchmarks.benchmark_loop import *
import scenes.benchmarks.benchmark_loop as loop
from pyray import *
from ecs import *

@component
class Position:
    x: float; y: float; z: float

@component
class Velocity:
    x: float; y: float; z: float

@component
class Mesh:
    color : Color
    
@component
class BoundingBox:
    x_min: float; y_min: float; z_min: float
    x_max: float; y_max: float; z_max: float

world = ECS()
world.register(Position, Velocity, Mesh, BoundingBox)

ground = world.create_entity(
    Position(0,-0.51,0),
    Mesh(LIGHTGRAY),
    BoundingBox(
        -25,0,-25,
        25,1,25
    )
)

def add_cubes():
    for e in world.create_entities(250):
        world.add_component(e,
            Position(
                get_random_value(-SPACE_SIZE, SPACE_SIZE),
                get_random_value(0, 25),
                get_random_value(-SPACE_SIZE, SPACE_SIZE) ),
            Velocity(
                get_random_value(-4, 4),
                0,
                get_random_value(-4, 4) ),
            BoundingBox(
                get_random_value(-CUBE_MAX, 0), get_random_value(-CUBE_MAX, 0), get_random_value(-CUBE_MAX, 0),
                get_random_value(1, CUBE_MAX), get_random_value(1, CUBE_MAX), get_random_value(1, CUBE_MAX) ),
            Mesh(rnd_color()),
        )

positions = world.get_store(Position)
velocities = world.get_store(Velocity)
meshes = world.get_store(Mesh)
bboxes = world.get_store(BoundingBox)

def draw():
    ents = world.where(Position, Mesh, BoundingBox)
    pos_vec, mesh_vec, bb_vec, = (positions.get_vector(ents), meshes.get_vector(ents), bboxes.get_vector(ents))
    pos_vec = vectorized( pos_vec, *positions.fields )
    bmins = vectorized( bb_vec, 'x_min', 'y_min', 'z_min')
    bmaxs = vectorized( bb_vec, 'x_max', 'y_max', 'z_max')
    sizes = bmaxs - bmins
    centers = pos_vec + (bmaxs + bmins) * 0.5
    for center, size, color in zip(centers, sizes, mesh_vec['color']):
        draw_cube(
            tuple(center),
            size[0], # x
            size[1], # y
            size[2], # z
            color
        )

def update(frameTime):
    pv = world.where(Position, Velocity)
    p_vec, v_vec = (positions.get_vector(pv), velocities.get_vector(pv))
    p_vec = vectorized( p_vec, *positions.fields )
    v_vec = vectorized( v_vec, *velocities.fields )
    p_vec += v_vec * frameTime
    positions.set_vector(pv, x=p_vec[:,0], y=p_vec[:,1], z=p_vec[:,2] )

def cubes_len():
    return world.count

loop.add_cubes = add_cubes
loop.draw = draw
loop.update = update
loop.cubes_len = cubes_len
loop.run()

