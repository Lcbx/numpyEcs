import raylib as rl
from pyray import Vector2, Vector3, Color, Camera3D, rl_load_texture, RenderTexture
from ecs import *
import shader_util


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

positions = world.get_store(Position)
velocities = world.get_store(Velocity)
meshes = world.get_store(Mesh)
bboxes = world.get_store(BoundingBox)

ground = world.create_entity(
    Position(0,-0.51,0),
    Mesh(rl.LIGHTGRAY),
    BoundingBox(
        -25,0,-25,
        25,1,25
    )
)

rnd_uint8 = lambda : rl.GetRandomValue(0, 255)
rnd_color = lambda : Color(rnd_uint8(),rnd_uint8(),rnd_uint8(),255)

SPACE_SIZE = 30
CUBE_MAX = 7

for e in world.create_entities(15):
    world.add_component(e,
        Position(
            rl.GetRandomValue(-SPACE_SIZE, SPACE_SIZE),
            rl.GetRandomValue(0, 20),
            rl.GetRandomValue(-SPACE_SIZE, SPACE_SIZE) ),
        Velocity(
            rl.GetRandomValue(-4, 4),
            0,
            rl.GetRandomValue(-4, 4) ),
        BoundingBox(
            rl.GetRandomValue(-CUBE_MAX, 0), rl.GetRandomValue(-CUBE_MAX, 0), rl.GetRandomValue(-CUBE_MAX, 0),
            rl.GetRandomValue(1, CUBE_MAX),  rl.GetRandomValue(1, CUBE_MAX),  rl.GetRandomValue(1, CUBE_MAX) ),
        Mesh(rnd_color()),
    )

camera_nearFar = (0.1, 1000.0)
camera = Camera3D(
    Vector3(30, 70,-25),
    Vector3(0,0,-25),
    Vector3(0,1,0),
    60.0,
    rl.CAMERA_PERSPECTIVE
)


WINDOW_SIZE = Vector2(800, 450) 
rl.InitWindow(int(WINDOW_SIZE.x), int(WINDOW_SIZE.y), b"Hello")
rl.SetTargetFPS(60)


def load_shaders():
    global sceneShader
    newShader = rl.LoadShader(b"scenes/lightmap.vs", b"scenes/lightmap.fs")
    if newShader.id > 0: sceneShader = newShader

sceneShader = None
load_shaders()


light_nearFar = (5,70)
light_camera = Camera3D(
    Vector3(-20, 30, 5),
    Vector3(0,0,0),
    Vector3(0,1,0),
    90.0,
    rl.CAMERA_ORTHOGRAPHIC
)
shadowmap = shader_util.shadow_buffer(1024,1024)


def run():
    while not rl.WindowShouldClose():
        
        frameTime = rl.GetFrameTime()

        if rl.IsKeyPressed(rl.KEY_R): load_shaders()

        pv = world.where(Position, Velocity)
        p_vec, v_vec = (positions.get_vector(pv), velocities.get_vector(pv))
        p_vec += v_vec * frameTime

        # bounce when out of bounds
        mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
        mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
        v_vec[mask_x, 0] *= -1
        v_vec[mask_z, 2] *= -1

        positions.set_vector(pv, p_vec)
        velocities.set_vector(pv, v_vec)
        
        rl.BeginTextureMode(shadowmap)
        rl.rlSetClipPlanes(light_nearFar[0], light_nearFar[1])
        rl.BeginMode3D(light_camera)
        rl.ClearBackground(rl.WHITE)
        rl.rlSetCullFace(rl.RL_CULL_FACE_FRONT)
        
        lightVP = rl.MatrixMultiply(rl.rlGetMatrixModelview(), rl.rlGetMatrixProjection())
    
        draw_scene()
        
        rl.rlSetCullFace(rl.RL_CULL_FACE_BACK)
        rl.EndMode3D()
        rl.EndTextureMode()


        rl.BeginDrawing()
        rl.rlSetClipPlanes(camera_nearFar[0], camera_nearFar[1])
        rl.BeginMode3D(camera)
        rl.ClearBackground(rl.WHITE)

        rl.BeginShaderMode(sceneShader)
        
        lightDir = rl.Vector3Normalize(rl.Vector3Subtract(light_camera.position, light_camera.target))
        shader_util.SetShaderValue(rl.GetShaderLocation(sceneShader,b"lightDir"),lightDir)
        rl.SetShaderValueMatrix(sceneShader,rl.GetShaderLocation(sceneShader,b"lightVP"),lightVP)
        rl.SetShaderValueTexture(sceneShader,rl.GetShaderLocation(sceneShader,b"texture_shadowmap"),shadowmap.depth)
        
        draw_scene()

        rl.EndShaderMode()
        rl.EndMode3D()
        
        draw_shadowmap()
        rl.DrawText(f"fps {rl.GetFPS()} cubes {world.count} ".encode('utf-8'), 10, 10, 20, rl.LIGHTGRAY)
        
        rl.EndDrawing()
    rl.CloseWindow()


def draw_shadowmap():
    display_size = WINDOW_SIZE.x / 5.0
    display_scale = display_size / float(shadowmap.depth.width)
    rl.DrawTextureEx(shadowmap.texture, Vector2(WINDOW_SIZE.x - display_size, 0.0), 0.0, display_scale, rl.RAYWHITE)
    rl.DrawTextureEx(shadowmap.depth, Vector2(WINDOW_SIZE.x - display_size, display_size), 0.0, display_scale, rl.RAYWHITE)

def draw_scene():
    ents = world.where(Position, Mesh, BoundingBox)
    pos_vec, mesh_vec, bb_vec, = (positions.get_vector(ents), meshes.get_vector(ents), bboxes.get_vector(ents))
    bmins = bb_vec[:,:3] # entity (int), bounding box (6 floats)
    bmaxs = bb_vec[:,3:]
    sizes = bmaxs - bmins
    centers = pos_vec + (bmaxs + bmins) * 0.5

    for center, size, mesh in zip(centers, sizes, mesh_vec):
        rl.DrawCube(
            tuple(center),
            size[0], # x
            size[1], # y
            size[2], # z
            mesh[meshes.color_id]
        )


run()