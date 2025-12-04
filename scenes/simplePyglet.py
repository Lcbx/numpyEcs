import math
import time
import ctypes
import numpy as np

from pyglet.app import run
from pyglet.window import  Window, key, mouse
import pyglet.gl as gl
from pyglet.graphics import Batch
import pyglet.graphics.shader as pygletShader
from shader_util import BetterShaderSource, PassthroughShaderSource
pygletShader.ShaderSource = PassthroughShaderSource

from pygltflib import GLTF2, BufferView, Accessor
from pyrr import Matrix44, Vector3

# ----------------------------
# Config
# ----------------------------
WINDOW_W, WINDOW_H = 1800, 900
TITLE = "pyglet + ShaderProgram + glTF"
MODEL_PATH = 'scenes/resources/rooftop_utility_pole.glb'


def build_shader_program():
    source = BetterShaderSource('scenes/shaders/pyglet.shader')
    vert = pygletShader.Shader(source.vertex_glsl, 'vertex')
    frag = pygletShader.Shader(source.fragment_glsl, 'fragment')
    return pygletShader.ShaderProgram(vert, frag)

# ----------------------------
# glTF loader (no GL calls)
# ----------------------------
def _get_data_from_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    acc: Accessor = gltf.accessors[accessor_index]
    bv: BufferView = gltf.bufferViews[acc.bufferView]
    buf = gltf.buffers[bv.buffer]

    # load buffer data
    if buf.uri is None:  # GLB
        bin_chunk = gltf.binary_blob()  # bytes
    else:
        # external .bin (not the case for .glb)
        raise NotImplementedError("External buffers not handled")

    # slice underlying bytes for this view
    b = bv.byteOffset or 0
    e = b + (bv.byteLength or 0)
    view_bytes = memoryview(bin_chunk)[b:e]

    # stride/offset into accessor
    comp_type = acc.componentType
    type_num_comps = {
        "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16
    }[acc.type]

    np_dtype = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32
    }[comp_type]

    stride = bv.byteStride or (np.dtype(np_dtype).itemsize * type_num_comps)
    count = acc.count
    offset = acc.byteOffset or 0

    # read tightly into numpy (use frombuffer then stride)
    arr = np.frombuffer(view_bytes, dtype=np_dtype, count=count*type_num_comps, offset=offset)
    if stride != np.dtype(np_dtype).itemsize * type_num_comps:
        raise NotImplementedError("Interleaved views not handled")
    #    # handle interleaved views (rare in simple exports)
    #    # fall back to manual gathering
    #    rec = np.empty((count, type_num_comps), dtype=np_dtype)
    #    base = offset
    #    for i in range(count):
    #        start = base + i*stride
    #        rec[i] = np.frombuffer(view_bytes, dtype=np_dtype, count=type_num_comps, offset=start)
    #    return rec
    #else:
    return arr.reshape(count, type_num_comps)


def load_gltf_first_mesh(program, batch, glb_path: str):
    gltf = GLTF2().load(glb_path)

    # take first mesh, first primitive
    mesh = gltf.meshes[0]
    prim = mesh.primitives[0]

    pos = _get_data_from_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
    nor = _get_data_from_accessor(gltf, prim.attributes.NORMAL).astype(np.float32) if prim.attributes.NORMAL is not None else np.zeros_like(pos)
    uv  = _get_data_from_accessor(gltf, prim.attributes.TEXCOORD_0).astype(np.float32) if prim.attributes.TEXCOORD_0 is not None else np.zeros((pos.shape[0],2), dtype=np.float32)
    idx = _get_data_from_accessor(gltf, prim.indices)
    if idx.dtype != np.uint32:
        idx = idx.astype(np.uint32)
    

    pos_flat = pos.reshape(-1).astype('f')
    nor_flat = nor.reshape(-1).astype('f')
    uv_flat  = uv.reshape(-1).astype('f')
    indices = idx.astype(int).reshape(-1)

    vertex_count = pos.shape[0]

    vlist = program.vertex_list_indexed(
        vertex_count,
        gl.GL_TRIANGLES,
        indices,
        batch=batch,
        aPos=('f', pos_flat),
        aNormal=('f', nor_flat),
        aUV=('f', uv_flat),
    )
    return vlist

# ----------------------------
# Camera
# ----------------------------
class Camera:
    def __init__(self, position, target, up, fovy_deg, near=0.1, far=1000.0):
        self.position = Vector3(position)
        self.target = Vector3(target)
        self.up = Vector3(up)
        self.fovy_deg = fovy_deg
        self.near = near
        self.far = far

    def view(self):
        return Matrix44.look_at(self.position, self.target, self.up, dtype=np.float32)

    def proj(self, aspect):
        return Matrix44.perspective_projection(
            self.fovy_deg, aspect, self.near, self.far, dtype=np.float32
        )

camera_dist = 30.0
camera = Camera(
    position=(-20.0, 70.0, 25.0),
    target=(0.0, 10.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fovy_deg=60.0,
    near=0.1, far=1000.0
)
orbit = True

start_t = time.time()
frame_start = 0.0
fps_frames = 0

# ----------------------------
# Application
# ----------------------------
def main():
    global camera_dist, camera, orbit

    #config = pyglet.gl.Config(
    #    double_buffer=True,
    #    major_version=4,
    #    minor_version=3,
    #    samples = 1,
    #)
    window = Window(
        width=WINDOW_W,
        height=WINDOW_H,
        caption=TITLE,
        #config=config,
        resizable=False,
        vsync=False
    )
    #window.vsync = False

    # Minimal explicit GL state (everything else via pyglet abstractions)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClearColor(0.15, 0.16, 0.19, 1.0)

    # Shader program & batch
    program = build_shader_program()
    batch = Batch()

    # Model transform
    scale = 5.0
    model_mat = Matrix44.from_scale([scale, scale, scale], dtype=np.float32)

    # Light direction (same as original)
    l = np.array([30.0, 30.0, 25.0]) - np.array([0.0, 0.0, -20.0])
    l = l / np.linalg.norm(l)
    light_dir = l.astype(np.float32)

    # Create vertex list in a batch (this hides VBO/VAO/EBO stuff)
    mesh_vlist = load_gltf_first_mesh(program, batch, MODEL_PATH)

    # ---------------- Events ----------------
    @window.event
    def on_resize(width, height):
        # Keep viewport in sync; this is one of the few explicit GL calls
        gl.glViewport(0, 0, width, height)

    @window.event
    def on_key_press(symbol, modifiers):
        global orbit
        if symbol == key.O:
            orbit = not orbit

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        global camera_dist, camera
        scrollspeed = 3.0
        yoff = scroll_y
        mw = scrollspeed * yoff
        if mw != 0 and (camera.position.y > scrollspeed + 0.5 or mw > 0.0):
            tar = np.array([camera.target.x, camera.target.y, camera.target.z], dtype=np.float32)
            cam = np.array([camera.position.x, camera.position.y, camera.position.z], dtype=np.float32)
            camera_dist -= mw * 0.3
            cam[1] += (tar[1] - cam[1]) / (abs(cam[1]) + 0.1) * mw
            camera.position = Vector3(cam)

    @window.event
    def on_draw():
        global frame_start, fps_frames, camera

        # per-frame timing
        now = time.time()
        elapsed = now - start_t

        if orbit:
            cam_ang = elapsed * 0.5
            camera.position = Vector3((
                math.cos(cam_ang) * camera_dist,
                camera.position.y,
                math.sin(cam_ang) * camera_dist
            ))

        # Clears color + depth using pyglet helper
        window.clear()

        aspect = window.width / window.height
        view = camera.view()
        proj = camera.proj(aspect)

        # Bind shader, set uniforms the pyglet way, then draw batch
        program.bind()

        # ShaderProgram accepts numpy arrays / sequences for uniforms.
        program['uModel'] = model_mat.reshape(-1).astype('f')
        program['uView']  = view.reshape(-1).astype('f')
        program['uProj']  = proj.reshape(-1).astype('f')
        program['uLightDir'] = light_dir

        batch.draw()
        pygletShader.ShaderProgram.unbind()

        # simple FPS to console
        fps_frames += 1
        if now - frame_start >= 1.0:
            print(f"fps {fps_frames}")
            frame_start = time.time()
            fps_frames = 0

    @window.event
    def on_close():
        mesh_vlist.delete()
        program.delete()
        return Window.on_close(window)

    # refresh rate, 1/60 = 60hz, 0 -> afap
    run(0) 

main()
