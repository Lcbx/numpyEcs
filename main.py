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
    #tests.test_single_free_entity_exact()
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
        25,1,25
    )
)

rnd_uint8 = lambda : get_random_value(0, 255)
rnd_color = lambda : Color(rnd_uint8(),rnd_uint8(),rnd_uint8(),255)

SPACE_SIZE = 30
CUBE_MAX = 7
for e in world.create_entities(10):
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


camera = Camera3D(
    Vector3(30, 70,-25),
    Vector3(0,0,-25),
    Vector3(0,1,0),
    60.0,
    CAMERA_PERSPECTIVE
)


init_window(800, 450, "Hello")
set_target_fps(60)

while not window_should_close():
    
    frameTime = get_frame_time()

    # I added a get_store methode to ECS class
    positions = world.get_store(Position)
    meshes = world.get_store(Mesh)
    bboxes = world.get_store(BoundingBox)

    pv = world.where(Position, Velocity)
    p_vec, v_vec = world.get_vectors(Position, Velocity, pv)
    p_vec += v_vec * frameTime
    positions.set_vector(pv, p_vec)
    
    begin_drawing()
    begin_mode_3d(camera)
    clear_background(WHITE)
    

    ents = world.where(Position, Mesh, BoundingBox)
    pos_arr, mesh_ar, bb_arr, = world.get_vectors(Position, Mesh, BoundingBox, ents)

    for pos, bb, mesh in zip(pos_arr, bb_arr, mesh_ar):
        bmin = bb[:3]
        bmax = bb[3:]
        center = (bmax + bmin) * 0.5
        draw_cube(
            tuple(pos + center),
            bmax[0] - bmin[0], # x
            bmax[1] - bmin[1], # y
            bmax[2] - bmin[2], # z
            mesh[meshes.color_id] # have to do this for object fields
        )
    
    end_mode_3d()
    draw_text(f"{get_fps()}", 10, 10, 20, LIGHTGRAY)
    end_drawing()
close_window()

