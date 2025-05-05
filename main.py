from tests import *
import argparse
import sys

parser = argparse.ArgumentParser(
    prog='python game engine',
    description='WIP python game engine with custom ecs',
    epilog='link: https://github.com/Lcbx/numpyEcs')
parser.add_argument('-t', '--tests', action='store_true', help='launches unit tests')
args = parser.parse_args()



if args.tests:
    #import tests
    #tests.test_no_free_entities()
    ret = pytest.main("-q tests.py".split())
    sys.exit(ret)


from pyray import *
from ecs import *

@component
class Position:
    x: float; y: float; z: float

@component
class Velocity:
    x: float; y: float; z: float

# @component
# class Orientation:
    # qx: float; qy: float; qz: float; qw: float

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
        25.5,0,25
    )
)

SPACE_SIZE = 30
rnd_uint8 = lambda : get_random_value(0, 255)
rnd_color = lambda : Color(rnd_uint8(),rnd_uint8(),rnd_uint8(),255)
for e in world.create_entities(25):
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
            get_random_value(-10, 0), get_random_value(-10, 0), get_random_value(-10, 0),
            get_random_value(1, 10), get_random_value(1, 10), get_random_value(1, 10) ),
        Mesh(rnd_color()),
    )


camera = Camera3D(
    Vector3(30, 70,-30),
    Vector3(0,0,0),
    Vector3(0,1,0),
    60.0,
    CAMERA_PERSPECTIVE
)


init_window(800, 450, "Hello")
set_target_fps(30)

while not window_should_close():
    
    frameTime = get_frame_time()
    
    pv = world.where(Position, Velocity)
    p_vec, v_vec = world.get_vectors(Position, Velocity, pv)
    p_vec += v_vec * frameTime
    world.set_vector(Position, pv, p_vec)
    
    begin_drawing()
    begin_mode_3d(camera)
    clear_background(WHITE)
    
    # NOTE: poor performance
    for e in world.where(Position, Mesh, BoundingBox):
        pos = world.get_component(e, Position)
        mesh = world.get_component(e, Mesh)
        bbSize = world.get_component(e, BoundingBox)
        # NOTE: this isn't the best representation of a bounding box
        draw_cube(Vector3(pos.x,pos.y,pos.z),
            bbSize.x_max - bbSize.x_min,
            bbSize.y_max - bbSize.y_min,
            bbSize.z_max - bbSize.z_min,
            mesh.color)
    
    end_mode_3d()
    draw_text(f"{get_fps()}", 10, 10, 20, LIGHTGRAY)
    end_drawing()
close_window()

