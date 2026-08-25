import argparse
import os
import subprocess


BACKENDS = {
	"dx11": ("windows", "s_5_0"),
	"spirv": ("linux", "spirv"),
	"glsl": ("linux", "120"),
	"essl": ("android", "300_es"),
	"metal": ("osx", "metal"),
}

SHADERS = {
	"vs_main.sc": "vertex",
	"fs_main.sc": "fragment",
	"vs_shadow.sc": "vertex",
	"fs_shadow.sc": "fragment",
}

parser = argparse.ArgumentParser()
parser.add_argument("--bgfx", required=True, help="bgfx directory")
parser.add_argument("--backend", choices=BACKENDS, default="dx11")
args = parser.parse_args()

root = os.path.dirname(__file__)
shader_root = os.path.join(root, "shaders")
out_dir = os.path.join(shader_root, args.backend)
os.makedirs(out_dir, exist_ok=True)
platform, profile = BACKENDS[args.backend]

for filename, shader_type in SHADERS.items():
	src = os.path.join(shader_root, filename)
	dst = os.path.join(out_dir, os.path.splitext(filename)[0] + ".bin")
	cmd = [
		args.bgfx + "/.build/win64_vs2022/bin/shadercRelease.exe",
		"-f", src,
		"-o", dst,
		"--type", shader_type,
		"--platform", platform,
		"--profile", profile,
		"--varyingdef", os.path.join(shader_root, "varying.def.sc"),
		"-i", args.bgfx + "/src",
	]
	print(" ".join(cmd))
	subprocess.run(cmd, check=True)
