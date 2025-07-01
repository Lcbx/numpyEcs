import raylib as rl
from pyray import Vector2, Vector3, Color, Camera3D, rl_load_texture, RenderTexture
from ecs import *
import shader_util as su

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


def load_shaders():
    global shadowMeshShader
    global shadowBlurShader
    global sceneShader

    try:
        newShader = su.BetterShader('scenes/shadowmesh.shader');
        if newShader.valid(): shadowMeshShader = newShader
        else: raise Exception('shadowmesh.shader')

        newShader = su.BetterShader('scenes/lightmap.compute');
        if newShader.valid(): shadowBlurShader = newShader
        else: raise Exception('lightmap.compute')

        newShader = su.BetterShader('scenes/lightmap.shader')
        if newShader.valid(): sceneShader = newShader
        else: raise Exception('lightmap.shader')

    except:
        print(newShader.functions[newShader._vertex_body])
        print('______________________')
        print(newShader.functions[newShader._fragment_body])


WINDOW_SIZE = Vector2(800, 500)
#rl.SetConfigFlags(rl.FLAG_MSAA_4X_HINT) #|rl.FLAG_WINDOW_RESIZABLE)
rl.InitWindow(int(WINDOW_SIZE.x), int(WINDOW_SIZE.y), b"Hello")
rl.SetTargetFPS(60)


sceneShader = None
shadowMeshShader = None
shadowBlurShader = None
load_shaders()

camera_nearFar = (0.1, 1000.0)
camera = Camera3D(
    Vector3(-20, 70,25),
    Vector3(0,10,0),
    Vector3(0,1,0),
    60.0,
    rl.CAMERA_PERSPECTIVE
)

light_nearFar = (5,100)
light_camera = Camera3D(
    Vector3(30, 30, 25),
    Vector3(0,0,-20),
    Vector3(0,1,0),
    90.0,
    rl.CAMERA_ORTHOGRAPHIC
)

unused_camera = None

SM_SIZE = 2048
SHADOW_FORMAT = rl.PIXELFORMAT_UNCOMPRESSED_R32G32B32
shadowmap = su.create_render_buffer(SM_SIZE,SM_SIZE,colorFormat=SHADOW_FORMAT, depth_map=True)
shadowmap_blurbuffer = su.create_render_buffer(SM_SIZE,SM_SIZE,colorFormat=SHADOW_FORMAT)

# model
model_root = b'C:/Users/User/Desktop/mixamo_toon_girl/'
model = rl.LoadModel(model_root + b'mixamo_toon_girl.glb')
#model_albedo = rl.LoadTexture(model_root + b'mixamo_toon_girl_Ch29_1001_Diffuse.png')
#su.SetMaterialTexture(model.materials[0], rl.MATERIAL_MAP_DIFFUSE, model_albedo)
#for i in range(model.materialCount):
#    model.materials[i].shader = shader.shader

#anims = su.LoadModelAnimations(model_root + b'mixamo_toon_girl.glb')
#animFrameCounter = 0


def run():
    global camera, unused_camera
    while not rl.WindowShouldClose():

        frameTime = rl.GetFrameTime()
        
        inputs()
        update(frameTime)

        # write shadowmap

        rl.BeginTextureMode(shadowmap)
        rl.rlSetClipPlanes(light_nearFar[0], light_nearFar[1])
        rl.BeginMode3D(light_camera)
        rl.ClearBackground(rl.WHITE)
        rl.rlSetCullFace(rl.RL_CULL_FACE_FRONT)
        
        lightDir = rl.Vector3Normalize(rl.Vector3Subtract(light_camera.position, light_camera.target))
        lightVP = rl.MatrixMultiply(rl.rlGetMatrixModelview(), rl.rlGetMatrixProjection())

        with shadowMeshShader:    
            shadowMeshShader.bias = rl.Vector3Scale(lightDir, 0.00001)
            draw_scene(shadowMeshShader,randomize_color=True)

        rl.rlSetCullFace(rl.RL_CULL_FACE_BACK)
        rl.EndMode3D()
        rl.EndTextureMode()

        # blur passes
        read_buffer = shadowmap
        write_buffer = shadowmap_blurbuffer
        step = 16.0/float(SM_SIZE)
        last = 1.0 /float(SM_SIZE)
        with shadowBlurShader:
            while step > last:
                step /= 2.0
                rl.BeginTextureMode(write_buffer)
                shadowBlurShader.stepSize = step    
                # screen-wide rectangle, y-flipped due to default OpenGL coordinates
                rl.DrawTextureRec(read_buffer.texture,
                    (0, 0, shadowmap.texture.width, -shadowmap.texture.height), (0, 0), rl.WHITE);
                rl.EndTextureMode()
                read_buffer, write_buffer = write_buffer, read_buffer
        

        rl.BeginDrawing()
        rl.rlSetClipPlanes(camera_nearFar[0], camera_nearFar[1])
        rl.BeginMode3D(camera)
        rl.ClearBackground(rl.WHITE)

        with sceneShader:
            sceneShader.lightDir = lightDir
            sceneShader.lightVP  = lightVP
            sceneShader.shadowDepthMap = shadowmap.depth
            sceneShader.shadowPenumbraMap = read_buffer.texture
            draw_scene(sceneShader)

        rl.EndMode3D()
        
        draw_shadowmap()
        rl.DrawText(f"fps {rl.GetFPS()} cubes {world.count} ".encode('utf-8'), 10, 10, 20, rl.LIGHTGRAY)
        
        rl.EndDrawing()
    rl.CloseWindow()


def draw_shadowmap():
    display_size = WINDOW_SIZE.x / 5.0
    display_scale = display_size / float(shadowmap.depth.width)
    rotation = 0
    rl.DrawTextureEx(shadowmap.texture, Vector2(WINDOW_SIZE.x - display_size, 0.0), rotation, display_scale, rl.RAYWHITE)
    rl.DrawTextureEx(shadowmap_blurbuffer.texture, Vector2(WINDOW_SIZE.x - display_size, display_size), rotation, display_scale, rl.RAYWHITE)
    rl.DrawTextureEx(shadowmap.depth, Vector2(WINDOW_SIZE.x - display_size, 2 * display_size), rotation, display_scale, rl.RAYWHITE)

def inputs():
    global camera
    global unused_camera

    if rl.IsKeyPressed(rl.KEY_R): load_shaders()
    if rl.IsKeyPressed(rl.KEY_P):
        light_camera.projection = (rl.CAMERA_ORTHOGRAPHIC
            if light_camera.projection == rl.CAMERA_PERSPECTIVE
            else rl.CAMERA_PERSPECTIVE
        )
    if rl.IsKeyPressed(rl.KEY_L):
        if unused_camera:
            camera = unused_camera
            unused_camera = None
        else:
            unused_camera = camera
            camera = light_camera

    scrollspeed = 3.0
    mw = scrollspeed * rl.GetMouseWheelMove()
    if mw != 0 and ( camera.position.y > scrollspeed + 0.5 or mw > 0.0):
        pos = np.array([camera.position.x, camera.position.y, camera.position.z])
        tar = np.array([camera.target.x, camera.target.y, camera.target.z])
        pos -= mw * tar * 0.1;
        new_pos = pos + (tar - pos) / np.linalg.norm(pos + 0.1) * mw
        camera.position = Vector3(new_pos[0], new_pos[1], new_pos[2])

def update(frameTime):

    #global animFrameCounter
    #rl.UpdateModelAnimation(model, anims[0], animFrameCounter)
    #animFrameCounter += 1
    #if animFrameCounter >= anims[0].frameCount: animFrameCounter = 0

    pv = world.where(Position, Velocity)
    p_vec, v_vec = (positions.get_vector(pv), velocities.get_vector(pv))
    p_vec += v_vec * frameTime

    # bounce when out of bounds
    mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
    mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
    v_vec[mask_x, 0] *= -1
    v_vec[mask_z, 2] *= -1
    # if further than boundary, it would get stuck alternating direction each frame
    p_vec[mask_x, 0] = np.sign(p_vec[mask_x, 0]) * 0.99 * SPACE_SIZE
    p_vec[mask_z, 2] = np.sign(p_vec[mask_z, 2]) * 0.99 * SPACE_SIZE

    positions.set_vector(pv, p_vec)
    velocities.set_vector(pv, v_vec)


def draw_scene(shader:su.BetterShader, randomize_color=False):
    global model
    with shader:
        for i in range(model.materialCount):
            model.materials[i].shader = shader.shader
        # model, position, rotation axis, rotation (deg), scale, tint
        rl.DrawModelEx(model, Vector3(0,0.5,0), Vector3(1,0,0), -90.0, Vector3(800,800,800), rl.WHITE)

        ents = world.where(Position, Mesh, BoundingBox)
        pos_vec, mesh_vec, bb_vec, = (positions.get_vector(ents), meshes.get_vector(ents), bboxes.get_vector(ents))
        bmins = bb_vec[:,:3] # entity (int), bounding box (6 floats)
        bmaxs = bb_vec[:,3:]
        sizes = bmaxs - bmins
        centers = pos_vec + (bmaxs + bmins) * 0.5
        meshIds = ents / np.max(ents)

        for meshId, center, size, mesh in zip(meshIds, centers, sizes, mesh_vec):
            rl.DrawCube(
                tuple(center),
                size[0], # x
                size[1], # y
                size[2], # z
                (int(255 * meshId),255,255,255) if randomize_color else mesh[meshes.color_id]
            )


run()