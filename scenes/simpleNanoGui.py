import sys, ctypes, math, time
import numpy as np
from OpenGL.GL import *
from pygltflib import GLTF2, Accessor, BufferView
from pyrr import Matrix44, Vector3
import nanogui
from nanogui import Screen, Window, GroupLayout, Label, Canvas
from nanogui import glfw as nglfw   # for key enums

# ----------------------------
# Config
# ----------------------------
WINDOW_W, WINDOW_H = 1800, 900
TITLE = "NanoGUI + OpenGL 4.3"
MODEL_PATH = 'scenes/resources/rooftop_utility_pole.glb'

# ----------------------------
# Minimal shader (unchanged)
# ----------------------------
VERT_SRC = r"""
#version 430 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uLightDir;

out vec3 vNormalWS;
out vec3 vLightDirWS;
out vec2 vUV;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vNormalWS = mat3(uModel) * aNormal;
    vLightDirWS = normalize(uLightDir);
    vUV = aUV;
    gl_Position = uProj * uView * worldPos;
}
"""

FRAG_SRC = r"""
#version 430 core
in vec3 vNormalWS;
in vec3 vLightDirWS;
in vec2 vUV;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormalWS);
    float NdotL = max(dot(N, vLightDirWS), 0.0);
    vec3 base = vec3(0.82, 0.71, 0.55);
    vec3 color = base * (0.2 + 0.8 * NdotL);
    FragColor = vec4(color, 1.0);
}
"""

def compile_shader(src, stype):
    sid = glCreateShader(stype)
    glShaderSource(sid, src)
    glCompileShader(sid)
    ok = glGetShaderiv(sid, GL_COMPILE_STATUS)
    if not ok:
        log = glGetShaderInfoLog(sid).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return sid

def link_program(vs, fs):
    pid = glCreateProgram()
    glAttachShader(pid, vs)
    glAttachShader(pid, fs)
    glLinkProgram(pid)
    ok = glGetProgramiv(pid, GL_LINK_STATUS)
    if not ok:
        log = glGetProgramInfoLog(pid).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    glDetachShader(pid, vs); glDetachShader(pid, fs)
    glDeleteShader(vs); glDeleteShader(fs)
    return pid

def build_default_program():
    vs = compile_shader(VERT_SRC, GL_VERTEX_SHADER)
    fs = compile_shader(FRAG_SRC, GL_FRAGMENT_SHADER)
    return link_program(vs, fs)

# ----------------------------
# glTF loader (positions, normals, uvs, indices)
# ----------------------------
def _get_data_from_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    acc: Accessor = gltf.accessors[accessor_index]
    bv: BufferView = gltf.bufferViews[acc.bufferView]
    buf = gltf.buffers[bv.buffer]

    if buf.uri is None:  # GLB
        bin_chunk = gltf.binary_blob()
    else:
        raise NotImplementedError("External buffers not handled in this snippet")

    b = bv.byteOffset or 0
    e = b + (bv.byteLength or 0)
    view_bytes = memoryview(bin_chunk)[b:e]

    comp_type = acc.componentType
    ncomps = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}[acc.type]
    np_dtype = {
        5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32
    }[comp_type]

    stride = bv.byteStride or (np.dtype(np_dtype).itemsize * ncomps)
    count  = acc.count
    offset = acc.byteOffset or 0

    arr = np.frombuffer(view_bytes, dtype=np_dtype, count=count*ncomps, offset=offset)
    if stride != np.dtype(np_dtype).itemsize * ncomps:
        rec = np.empty((count, ncomps), dtype=np_dtype)
        base = offset
        for i in range(count):
            start = base + i*stride
            rec[i] = np.frombuffer(view_bytes, dtype=np_dtype, count=ncomps, offset=start)
        return rec
    else:
        return arr.reshape(count, ncomps)

class GLMesh:
    def __init__(self, vao, vbo, nbo, tbo, ebo, index_count):
        self.vao = vao; self.vbo = vbo; self.nbo = nbo; self.tbo = tbo; self.ebo = ebo
        self.index_count = index_count
    def destroy(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo]); glDeleteBuffers(1, [self.nbo])
        glDeleteBuffers(1, [self.tbo]); glDeleteBuffers(1, [self.ebo])

def load_gltf_first_mesh(glb_path: str) -> GLMesh:
    gltf = GLTF2().load(glb_path)
    mesh = gltf.meshes[0]; prim = mesh.primitives[0]

    pos = _get_data_from_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
    nor = _get_data_from_accessor(gltf, prim.attributes.NORMAL).astype(np.float32) if prim.attributes.NORMAL is not None else np.zeros_like(pos)
    uv  = _get_data_from_accessor(gltf, prim.attributes.TEXCOORD_0).astype(np.float32) if prim.attributes.TEXCOORD_0 is not None else np.zeros((pos.shape[0],2), dtype=np.float32)
    idx = _get_data_from_accessor(gltf, prim.indices)
    if idx.dtype != np.uint32: idx = idx.astype(np.uint32)

    vao = glGenVertexArrays(1); glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, pos.nbytes, pos, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    nbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, nbo)
    glBufferData(GL_ARRAY_BUFFER, nor.nbytes, nor, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    tbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tbo)
    glBufferData(GL_ARRAY_BUFFER, uv.nbytes, uv, GL_STATIC_DRAW)
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)

    glBindVertexArray(0)
    return GLMesh(vao, vbo, nbo, tbo, ebo, index_count=idx.size)

# ----------------------------
# Camera
# ----------------------------
class Camera:
    def __init__(self, position, target, up, fovy_deg, near=0.1, far=1000.0):
        self.position = Vector3(position)
        self.target   = Vector3(target)
        self.up       = Vector3(up)
        self.fovy_deg = fovy_deg
        self.near, self.far = near, far
    def view(self):
        return Matrix44.look_at(self.position, self.target, self.up, dtype=np.float32)
    def proj(self, aspect):
        return Matrix44.perspective_projection(self.fovy_deg, aspect, self.near, self.far, dtype=np.float32)

# ----------------------------
# GLCanvas that renders the scene
# ----------------------------
class MyCanvas(Canvas):
    def __init__(self, parent, camera):
        super().__init__(parent)
        self.camera = camera
        self._init_done = False

        # scene state
        self.scale = 5.0
        l = np.array([30.0, 30.0, 25.0]) - np.array([0.0, 0.0, -20.0])
        self.light_dir = (l / np.linalg.norm(l)).astype(np.float32)

        # will be filled on first draw when GL is surely ready
        self.program = None
        self.loc_uModel = self.loc_uView = self.loc_uProj = self.loc_uLight = -1
        self.model_mat = Matrix44.from_scale([self.scale, self.scale, self.scale], dtype=np.float32)
        self.mesh = None

    def _lazy_init(self):
        # GL state
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.15, 0.16, 0.19, 1.0)

        # program + uniforms
        self.program = build_default_program()
        self.loc_uModel = glGetUniformLocation(self.program, "uModel")
        self.loc_uView  = glGetUniformLocation(self.program, "uView")
        self.loc_uProj  = glGetUniformLocation(self.program, "uProj")
        self.loc_uLight = glGetUniformLocation(self.program, "uLightDir")

        # geometry
        self.mesh = load_gltf_first_mesh(MODEL_PATH)

        # sanity
        print("GL_VERSION:", glGetString(GL_VERSION).decode())
        self._init_done = True

    def draw(self, ctx):
        if not self._init_done:
            self._lazy_init()

        # viewport from canvas size
        w, h = self.size()
        glViewport(0, 0, int(w), int(h))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = max(1.0, float(w) / float(max(1, h)))
        view = self.camera.view()
        proj = self.camera.proj(aspect)

        glUseProgram(self.program)
        glUniformMatrix4fv(self.loc_uModel, 1, GL_FALSE, self.model_mat)
        glUniformMatrix4fv(self.loc_uView,  1, GL_FALSE, view)
        glUniformMatrix4fv(self.loc_uProj,  1, GL_FALSE, proj)
        glUniform3fv(self.loc_uLight, 1, self.light_dir)

        glBindVertexArray(self.mesh.vao)
        glDrawElements(GL_TRIANGLES, self.mesh.index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def cleanup(self):
        if self.mesh:
            self.mesh.destroy()
            self.mesh = None
        if self.program:
            glDeleteProgram(self.program)
            self.program = None

# ----------------------------
# NanoGUI Screen with input + FPS
# ----------------------------
class App(Screen):
    def __init__(self):
        super(App, self).__init__(
            (WINDOW_W, WINDOW_H), TITLE,
            resizable=True,
            maximized=False,
            fullscreen=False,
            #depth_buffer=True,
            #stencil_buffer=False,
            #float_buffer=False,
            gl_major=4,
            gl_minor=1
        )

        # camera (mirrors your original)
        self.camera_dist = 30.0
        self.camera = Camera(
            position=(-20.0, 70.0, 25.0),
            target=(0.0, 10.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fovy_deg=60.0, near=0.1, far=1000.0
        )
        self.orbit = True

        window = nanogui.Window(self, "Canvas widget demo")
        window.set_position((15, 15))
        window.set_layout(nanogui.GroupLayout())

        self.canvas = MyCanvas(self, self.camera)
        self.canvas.set_size((WINDOW_W, WINDOW_H))

        self.stats = Window(self, "Stats")
        self.stats.set_position((15, 15))
        self.stats.set_layout(nanogui.GroupLayout())

        self.fps_label = Label(self.canvas, "FPS: 0")

        # see https://github.com/mitsuba-renderer/nanogui/issues/184
        #self.perform_layout()

        # fps tracking
        self._accum = 0.0
        self._frames = 0
        self._last = time.time()

        # start time for orbit
        self.start_t = time.time()

    # called every frame before widgets are drawn
    def draw_contents(self):
        now = time.time()
        dt = now - self._last
        self._last = now

        # orbit update
        if self.orbit:
            elapsed = now - self.start_t
            cam_ang = elapsed * 0.5
            self.camera.position = Vector3((
                math.cos(cam_ang) * self.camera_dist,
                self.camera.position.y,
                math.sin(cam_ang) * self.camera_dist
            ))

        # fps
        self._accum += dt; self._frames += 1
        if self._accum >= 1.0:
            self.fps_label.set_caption(f"FPS: {self._frames}")
            self._accum = 0.0; self._frames = 0

    # keyboard: toggle orbit on 'O'
    def keyboard_event(self, key, scancode, action, modifiers):
        if action == nglfw.PRESS:
            if key == nglfw.KEY_O:
                self.orbit = not self.orbit
                return True
        # fall back to default handler
        return super().keyboard_event(key, scancode, action, modifiers)

    # scroll: zoom and adjust camera height (matches your logic)
    def scroll_event(self, p, rel):
        # rel is a Vector2f, rel.y is the vertical wheel delta
        mw = 3.0 * rel.y
        cam = np.array([self.camera.position.x, self.camera.position.y, self.camera.position.z], dtype=np.float32)
        tar = np.array([self.camera.target.x,   self.camera.target.y,   self.camera.target.z],   dtype=np.float32)
        if mw != 0 and (self.camera.position.y > 3.0 + 0.5 or mw > 0.0):
            self.camera_dist -= mw * 0.3
            cam[1] += (tar[1] - cam[1]) / (abs(cam[1]) + 0.1) * mw
            self.camera.position = Vector3(cam.tolist())
        return True  # we handled it

    def dispose(self):
        # make sure GL resources are freed
        self.canvas.cleanup()


nanogui.init()
app = App()
app.draw_all()
app.set_visible(True)
nanogui.run()
app.dispose()
nanogui.shutdown()
