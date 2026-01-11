from ecs import *
from shader_util import *
from random import randint 

@component
class Position:
	x: float; y: float; z: float

@component
class Velocity:
	x: float; y: float; z: float
	
@component
class BoundingBox:
	x_min: float; y_min: float; z_min: float
	x_max: float; y_max: float; z_max: float

world = ECS()
world.register(Position, Velocity, BoundingBox)

positions = world.get_store(Position)
velocities = world.get_store(Velocity)
bboxes = world.get_store(BoundingBox)

SPACE_SIZE = 180
CUBE_MAX_SIDE = 7

for e in world.create_entities(200):
	world.add_component(e,
		Position(
			randint(-SPACE_SIZE, SPACE_SIZE),
			randint(0, 20),
			randint(-SPACE_SIZE, SPACE_SIZE) ),
		Velocity(
			randint(-4, 4),
			0,
			randint(-4, 4) ),
		BoundingBox(
			randint(-CUBE_MAX_SIDE, 0), randint(-CUBE_MAX_SIDE, 0), randint(-CUBE_MAX_SIDE, 0),
			randint(1, CUBE_MAX_SIDE),  randint(1, CUBE_MAX_SIDE),  randint(1, CUBE_MAX_SIDE) )
	)


def load_shaders():
	global sceneShader, prepassShader

	shaders_dir = 'scenes/shaders/'
	sceneShader = build_shader_program(shaders_dir + 'lightmap.shader')
	prepassShader = build_shader_program(shaders_dir + 'prepass.shader')

WINDOW_w, WINDOW_h = 1800, 900
window = RenderContext.InitWindow(WINDOW_w, WINDOW_h, "Hello")


sceneShader : Program = None
prepassShader : Program = None
load_shaders()


camera_dist = 30
camera = Camera(
	position=(-20.0, 70.0, 25.0),
	target=(0.0, 10.0, 0.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=60.0,
	near=0.1, far=1000.0
)

light_camera = Camera(
	position=(30, 30, 25),
	target=(0,0,-20),
	up=(0,1,0),
	fovy_deg=90.0,
	near=10,far=300,
	perspective=False
)

unused_camera = None

# TODO : handle resolution changes (rebuild buffers)
# None is for color format, means we dont actually draw into it
prepass_buffer = create_frame_buffer(WINDOW_w, WINDOW_h, None, depth_map=True)

SM_SIZE = 2048
shadow_buffer = create_frame_buffer(SM_SIZE,SM_SIZE,None, depth_map=True)


# FOR DEBUG
sceneShader = build_shader_program('scenes/shaders/pyglet.shader')

# model
model_root = 'scenes/resources/'
scale = 5.0
mesh = load_gltf_meshes(sceneShader, model_root + 'rooftop_utility_pole.glb')[0]
model_mat = Mat4.from_translation( (0, 15, 5) ) * Mat4.from_scale( (scale, scale, scale) )
mesh['uTint'] = (0.2, 0.5, 0.2, 1.0)
mesh['uModel'] = model_mat
vMesh = mesh.draw(transform=model_mat) # use default batch instead

heightmap = load_gltf_meshes(sceneShader, model_root + 'heightmap_mesh.glb')
for h in heightmap:
    h['uTint'] = (0.82, 0.71, 0.55, 1.0)
    h.draw()

EnableDepth()
EnableCullFace()
setClearColor(0.15, 0.16, 0.19, 1.0)

@window.event
def on_draw():
	lightDir = np.array(light_camera.position, dtype=np.float32) - np.array(light_camera.target, dtype=np.float32)
	if any( x != 0. for x in lightDir): lightDir /= np.linalg.norm(lightDir)

	lightView = light_camera.view()
	lightProj = light_camera.projection(1) # width/height = SM_SIZE / SM_SIZE = 1

	with RenderContext(shader=sceneShader, camera=camera):
		ClearBuffers()
		sceneShader['uLightDir'] = lightDir
		#sceneShader['lightVP'] = lightView @ lightProj
		#sceneShader['invProj'] = RenderContext.projection.inverse
		#sceneShader['viewDepthMap'] = prepass_buffer.depth
		#sceneShader['shadowDepthMap'] = shadow_buffer.depth

		vCubes = Cubes(sceneShader,
			((2,0,0), (-1,0,0)),	# positions
			((3,1,2), (1,1,1)),	    # sizes
            (0.12, 0.31, 0.65, 1.0) # color
		).draw()
	vCubes.delete() # delete previous cubes or we add more each frame 


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


run(0)

def run():
	while not rl.WindowShouldClose():

		frameTime = rl.GetFrameTime()
		
		with WatchTimer('update'):
			inputs()
			update(frameTime)
		
		with WatchTimer('total draw'):
			
			with WatchTimer('shadow'):

				with RenderContext(shader=prepassShader, texture=shadow_buffer, camera=light_camera, clipPlanes=light_nearFar) as render:
					lightDir = rl.Vector3Normalize(rl.Vector3Subtract(light_camera.position, light_camera.target))
					lightView = rl.rlGetMatrixModelview()
					lightProj = rl.rlGetMatrixProjection()
					lightViewProj = rl.MatrixMultiply(lightView, lightProj)
					SetPolygonOffset(3, 1)
					draw_scene(render)
					DisablePolygonOffset()

			# main camera
			with WatchTimer('main camera'):
				# TODO : don't do frustum culling twice (prepass + main pass)
				
				# z prepass
				with WatchTimer('prepass'):
					with RenderContext(shader=prepassShader, texture=prepass_buffer, clipPlanes=camera_nearFar, camera=camera) as render:
						ClearBuffers()
						view = rl.rlGetMatrixModelview()
						proj = rl.rlGetMatrixProjection()
						SetPolygonOffset(0.1,0.1)
						draw_scene(render)
						DisablePolygonOffset()
				
				#GenTextureMipmaps(prepass_buffer.depth)
				
				# transfer depth to main buffer for early z discard
				with WatchTimer('forward pass'):
					rl.BeginDrawing()
					ClearBuffers()
					#ClearColorBuffer()
					
					# transfer depth to main buffer for early z discard
					TransferDepth(prepass_buffer.id, WINDOW_w, WINDOW_h, 0, WINDOW_w, WINDOW_h)
					
					# trying to pass depth with a shader for msaa
					#with RenderContext(shader=depthTransferShader, camera=camera) as render:
					#	DrawTexture(prepass_buffer.depth, WINDOW_w, WINDOW_h)
					
					with RenderContext(shader=sceneShader, camera=camera) as render:
						sceneShader.invProj = rl.MatrixInvert(proj)
						sceneShader.lightDir = lightDir
						sceneShader.lightVP = lightViewProj
						sceneShader.viewDepthMap = prepass_buffer.depth
						sceneShader.shadowDepthMap = shadow_buffer.depth
						draw_scene(render)
				
				# maybe add toggles for drawing buffers
				#draw_shadow_buffer()
				#draw_prepass()
				#draw_AO()
				#draw_mat_tex(model)
		
			rl.DrawText(f"fps {rl.GetFPS()} cubes {world.count}".encode(), 10, 10, 20, rl.LIGHTGRAY)
			WatchTimer.display(10, 40, 20, rl.LIGHTGRAY)

			# sleeps for vsync / target fps
			rl.EndDrawing()

			# a lot of stuff happening in EndDrawing it seems...
			WatchTimer.capture()


rotation = 0
DFLT_VIEW_RATIO =  1 / 6.0
def draw_shadow_buffer():
	display_size = WINDOW_w * DFLT_VIEW_RATIO
	display_scale = display_size / float(shadow_buffer.depth.width)
	rl.DrawTextureEx(shadow_buffer.texture, (WINDOW_w - display_size, 0.0), rotation, display_scale, rl.RAYWHITE)
	rl.DrawTextureEx(shadow_buffer2.texture, (WINDOW_w - display_size, display_size), rotation, display_scale, rl.RAYWHITE)
	rl.DrawTextureEx(shadow_buffer.depth, (WINDOW_w - display_size, 2 * display_size), rotation, display_scale, rl.RAYWHITE)
def draw_prepass():
	display_size = WINDOW_w * DFLT_VIEW_RATIO
	display_scale = display_size / float(prepass_buffer.texture.width)
	rl.DrawTextureEx(prepass_buffer.texture, (WINDOW_w - display_size, 0.0), rotation, display_scale, rl.RAYWHITE)
	rl.DrawTextureEx(prepass_buffer.depth, (WINDOW_w - display_size, display_size), rotation, display_scale, rl.RAYWHITE)
def draw_mat_tex(model):
	display_size = WINDOW_w * DFLT_VIEW_RATIO
	i = 0
	for i in range(model.materialCount):
		mat = model.materials[i]
		for tex in [mat.maps[i].texture for i in range(12)]: # rl.MAX_MATERIAL_MAPS
			#print(tex.id)
			if tex.id != 0:
				display_scale = display_size / float(tex.width)
				rl.DrawTextureEx(tex, (WINDOW_w - display_size, i * tex.height * display_scale), rotation, display_scale, rl.RAYWHITE)


orbit = True
applyAO = True
def inputs():
	global camera
	global camera_dist
	global unused_camera
	global orbit
	global applyAO

	if rl.IsKeyPressed(rl.KEY_R): load_shaders()
	if rl.IsKeyPressed(rl.KEY_O):
		orbit = not orbit
	if rl.IsKeyPressed(rl.KEY_I):
		applyAO = not applyAO

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
		cam_pos = np.array([camera.position.x, camera.position.y, camera.position.z])
		tar = np.array([camera.target.x, camera.target.y, camera.target.z])
		camera_dist -= mw * 0.3;
		cam_pos[1] += (tar[1] - cam_pos[1]) / np.linalg.norm(cam_pos[1] + 0.1) * mw
		camera.position = Vector3(cam_pos[0], cam_pos[1], cam_pos[2])

def update(frameTime):
	global camera
	
	time = rl.GetTime()
	
	if orbit:
		cam_ang = time * 0.5
		camera.position = Vector3(
			np.cos(cam_ang) * camera_dist,
			camera.position.y,
			np.sin(cam_ang) * camera_dist)

	#global animFrameCounter
	#rl.UpdateModelAnimation(model, anims[0], animFrameCounter)
	#animFrameCounter += 1
	#if animFrameCounter >= anims[0].frameCount: animFrameCounter = 0

	pv = world.where(Position, Velocity)
	p_vec, v_vec = (positions.get_full_vector(pv), velocities.get_full_vector(pv))
	p_vec += v_vec * frameTime

	# bounce when out of bounds
	mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
	mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
	v_vec[mask_x, 0] *= -1
	v_vec[mask_z, 2] *= -1
	# if further than boundary, it would get stuck alternating direction each frame
	p_vec[mask_x, 0] = np.sign(p_vec[mask_x, 0]) * 0.99 * SPACE_SIZE
	p_vec[mask_z, 2] = np.sign(p_vec[mask_z, 2]) * 0.99 * SPACE_SIZE

	positions.set_full_vector(pv, p_vec.transpose())
	velocities.set_full_vector(pv, v_vec.transpose())


def draw_scene(render:RenderContext, randomize_color=False):
	global model
	with render.shader:
		for i in range(model.materialCount):
			model.materials[i].shader = render.shader.shaderStruct
		for i in range(heightmap.materialCount):
			heightmap.materials[i].shader = render.shader.shaderStruct
		for i in range(pole.materialCount):
			pole.materials[i].shader = render.shader.shaderStruct

		
		# model, position, rotation axis, rotation (deg), scale, tint
		scale = 1
		rl.DrawModelEx(model, (0,0,10), (1,0,0), 0.0, (scale,scale,scale), rl.BEIGE)
		rl.DrawModelEx(heightmap, (0,0,0), (1,0,0), 0.0, (scale,scale,scale), rl.BEIGE)
		scale = 5
		# seems like there is per vertex color in that mesh, with all the same bluish color... ?
		rl.DrawModelEx(pole, (0,0,-10), (1,0,0), 0.0, (scale,scale,scale), rl.WHITE) 

		ents = world.where(Position, Mesh, BoundingBox)
		pos_vec, mesh_vec, bb_vec, = (positions.get_full_vector(ents), meshes.get_vector(ents), bboxes.get_full_vector(ents))
		bmins = bb_vec[:,:3]
		bmaxs = bb_vec[:,3:]
		sizes = bmaxs - bmins
		centers = pos_vec + (bmaxs + bmins) * 0.5
		meshIds = ents / np.max(ents)
		
		for meshId, center, size, color in zip(meshIds, centers, sizes, mesh_vec['color']):
			rl.DrawCube(
				tuple(center),
				size[0], # x
				size[1], # y
				size[2], # z
				(int(255 * meshId),255,255,255) if randomize_color else color
			)


run()