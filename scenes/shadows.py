from ecs import *
import shader_util as su
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
	global sceneShader
	global prepassShader
	global depthTransferShader

	try:
		shaders_dir = 'scenes/shaders/'
		sceneShader = build_shader_program(shaders_dir + 'lightmap.shader')
		depthTransferShader = build_shader_program(shaders_dir + 'depthTransfer.shader')
		prepassShader = build_shader_program(shaders_dir + 'prepass.shader')

	except Exception as ex:
		print('--------------> failed compiling', ex.args[0])
		withLineNumbers = lambda text: '\n'.join([f'{i+1:>3}:{l}' for i,l in enumerate(text.splitlines())])
		print(withLineNumbers(newShader.source.vertex_glsl))
		print('______________________')
		print(withLineNumbers(newShader.source.fragment_glsl))

WINDOW_w, WINDOW_h = 1800, 900
InitWindow(WINDOW_w, WINDOW_h, "Hello")


sceneShader : Program = None
depthTransferShader : Program = None
prepassShader : Program = None
load_shaders()

camera_dist = 30
camera_nearFar = (0.1, 1000.0)
camera = Camera(
	(-20, 70, 25),
	(0,10,0),
	(0,1,0),
	60.0
)

light_nearFar = (10,300)
light_camera = Camera(
	(30, 30, 25),
	(0,0,-20),
	(0,1,0),
	90.0,
	False
)

unused_camera = None

# TODO : handle resolution changes (rebuild buffers)
# None is for color format, means we dont actually draw into it
prepass_buffer = su.create_render_buffer(WINDOW_w, WINDOW_h, None, depth_map=True)

SM_SIZE = 2048
shadow_buffer = su.create_render_buffer(SM_SIZE,SM_SIZE,None, depth_map=True)

batch = Batch()

# model
model_root = 'scenes/resources/'
#model = su.rl.LoadModel(model_root + b'turret.obj')
#model_albedo = su.rl.LoadTexture(model_root + b'turret_diffuse.png')
#su.SetMaterialTexture(model.materials[0], su.rl.MATERIAL_MAP_DIFFUSE, model_albedo)
#heightmap = load_gltf_first_mesh(sceneShader, batch, model_root + 'heightmap_mesh.glb')
pole = load_gltf_first_mesh(sceneShader, model_root + 'rooftop_utility_pole.glb')
pole.draw(batch)

#anims = su.LoadModelAnimations(model_root + b'mixamo_toon_gisu.rl.glb')
#animFrameCounter = 0

def run():
	while not su.rl.WindowShouldClose():

		frameTime = su.rl.GetFrameTime()
		
		with su.WatchTimer('update'):
			inputs()
			update(frameTime)
		
		with su.WatchTimer('total draw'):
			
			with su.WatchTimer('shadow'):

				with su.RenderContext(shader=prepassShader, texture=shadow_buffer, camera=light_camera, clipPlanes=light_nearFar) as render:
					su.ClearBuffers()
					lightDir = su.rl.Vector3Normalize(su.rl.Vector3Subtract(light_camera.position, light_camera.target))
					lightView = su.rl.rlGetMatrixModelview()
					lightProj = su.rl.rlGetMatrixProjection()
					lightViewProj = su.rl.MatrixMultiply(lightView, lightProj)
					su.SetPolygonOffset(3, 1)
					draw_scene(render)
					su.DisablePolygonOffset()

			# main camera
			with su.WatchTimer('main camera'):
				# TODO : don't do frustum culling twice (prepass + main pass)
				
				# z prepass
				with su.WatchTimer('prepass'):
					with su.RenderContext(shader=prepassShader, texture=prepass_buffer, clipPlanes=camera_nearFar, camera=camera) as render:
						su.ClearBuffers()
						view = su.rl.rlGetMatrixModelview()
						proj = su.rl.rlGetMatrixProjection()
						su.SetPolygonOffset(0.1,0.1)
						draw_scene(render)
						su.DisablePolygonOffset()
				
				#su.GenTextureMipmaps(prepass_buffer.depth)
				
				# transfer depth to main buffer for early z discard
				with su.WatchTimer('forward pass'):
					su.rl.BeginDrawing()
					su.ClearBuffers()
					#su.ClearColorBuffer()
					
					# transfer depth to main buffer for early z discard
					su.TransferDepth(prepass_buffer.id, WINDOW_w, WINDOW_h, 0, WINDOW_w, WINDOW_h)
					
					# trying to pass depth with a shader for msaa
					#with su.RenderContext(shader=depthTransferShader, camera=camera) as render:
					#	su.DrawTexture(prepass_buffer.depth, WINDOW_w, WINDOW_h)
					
					with su.RenderContext(shader=sceneShader, camera=camera) as render:	
						sceneShader.invProj = su.rl.MatrixInvert(proj)
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
		
			su.rl.DrawText(f"fps {su.rl.GetFPS()} cubes {world.count}".encode(), 10, 10, 20, su.rl.LIGHTGRAY)
			su.WatchTimer.display(10, 40, 20, su.rl.LIGHTGRAY)

			# sleeps for vsync / target fps
			su.rl.EndDrawing()

			# a lot of stuff happening in EndDrawing it seems...
			su.WatchTimer.capture()


rotation = 0
DFLT_VIEW_RATIO =  1 / 6.0
def draw_shadow_buffer():
	display_size = WINDOW_w * DFLT_VIEW_RATIO
	display_scale = display_size / float(shadow_buffer.depth.width)
	su.rl.DrawTextureEx(shadow_buffer.texture, (WINDOW_w - display_size, 0.0), rotation, display_scale, su.rl.RAYWHITE)
	su.rl.DrawTextureEx(shadow_buffer2.texture, (WINDOW_w - display_size, display_size), rotation, display_scale, su.rl.RAYWHITE)
	su.rl.DrawTextureEx(shadow_buffer.depth, (WINDOW_w - display_size, 2 * display_size), rotation, display_scale, su.rl.RAYWHITE)
def draw_prepass():
	display_size = WINDOW_w * DFLT_VIEW_RATIO
	display_scale = display_size / float(prepass_buffer.texture.width)
	su.rl.DrawTextureEx(prepass_buffer.texture, (WINDOW_w - display_size, 0.0), rotation, display_scale, su.rl.RAYWHITE)
	su.rl.DrawTextureEx(prepass_buffer.depth, (WINDOW_w - display_size, display_size), rotation, display_scale, su.rl.RAYWHITE)
def draw_mat_tex(model):
	display_size = WINDOW_w * DFLT_VIEW_RATIO
	i = 0
	for i in range(model.materialCount):
		mat = model.materials[i]
		for tex in [mat.maps[i].texture for i in range(12)]: # rl.MAX_MATERIAL_MAPS
			#print(tex.id)
			if tex.id != 0:
				display_scale = display_size / float(tex.width)
				su.rl.DrawTextureEx(tex, (WINDOW_w - display_size, i * tex.height * display_scale), rotation, display_scale, su.rl.RAYWHITE)


orbit = True
applyAO = True
def inputs():
	global camera
	global camera_dist
	global unused_camera
	global orbit
	global applyAO

	if su.rl.IsKeyPressed(su.rl.KEY_R): load_shaders()
	if su.rl.IsKeyPressed(su.rl.KEY_O):
		orbit = not orbit
	if su.rl.IsKeyPressed(su.rl.KEY_I):
		applyAO = not applyAO

	if su.rl.IsKeyPressed(su.rl.KEY_P):
		light_camera.projection = (su.rl.CAMERA_ORTHOGRAPHIC
			if light_camera.projection == su.rl.CAMERA_PERSPECTIVE
			else su.rl.CAMERA_PERSPECTIVE
		)
	if su.rl.IsKeyPressed(su.rl.KEY_L):
		if unused_camera:
			camera = unused_camera
			unused_camera = None
		else:
			unused_camera = camera
			camera = light_camera

	scrollspeed = 3.0
	mw = scrollspeed * su.rl.GetMouseWheelMove()
	if mw != 0 and ( camera.position.y > scrollspeed + 0.5 or mw > 0.0):
		cam_pos = np.array([camera.position.x, camera.position.y, camera.position.z])
		tar = np.array([camera.target.x, camera.target.y, camera.target.z])
		camera_dist -= mw * 0.3;
		cam_pos[1] += (tar[1] - cam_pos[1]) / np.linalg.norm(cam_pos[1] + 0.1) * mw
		camera.position = Vector3(cam_pos[0], cam_pos[1], cam_pos[2])

def update(frameTime):
	global camera
	
	time = su.rl.GetTime()
	
	if orbit:
		cam_ang = time * 0.5
		camera.position = Vector3(
			np.cos(cam_ang) * camera_dist,
			camera.position.y,
			np.sin(cam_ang) * camera_dist)

	#global animFrameCounter
	#su.rl.UpdateModelAnimation(model, anims[0], animFrameCounter)
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


def draw_scene(render:su.RenderContext, randomize_color=False):
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
		su.rl.DrawModelEx(model, (0,0,10), (1,0,0), 0.0, (scale,scale,scale), su.rl.BEIGE)
		su.rl.DrawModelEx(heightmap, (0,0,0), (1,0,0), 0.0, (scale,scale,scale), su.rl.BEIGE)
		scale = 5
		# seems like there is per vertex color in that mesh, with all the same bluish color... ?
		su.rl.DrawModelEx(pole, (0,0,-10), (1,0,0), 0.0, (scale,scale,scale), su.rl.WHITE) 

		ents = world.where(Position, Mesh, BoundingBox)
		pos_vec, mesh_vec, bb_vec, = (positions.get_full_vector(ents), meshes.get_vector(ents), bboxes.get_full_vector(ents))
		bmins = bb_vec[:,:3]
		bmaxs = bb_vec[:,3:]
		sizes = bmaxs - bmins
		centers = pos_vec + (bmaxs + bmins) * 0.5
		meshIds = ents / np.max(ents)
		
		for meshId, center, size, color in zip(meshIds, centers, sizes, mesh_vec['color']):
			su.rl.DrawCube(
				tuple(center),
				size[0], # x
				size[1], # y
				size[2], # z
				(int(255 * meshId),255,255,255) if randomize_color else color
			)


run()