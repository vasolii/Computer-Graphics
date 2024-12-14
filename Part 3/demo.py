import numpy as np
import matplotlib.pyplot as plt
from functions import render_object
from PIL import Image

#load data from hw3.npy
data = np.load('h3.npy', allow_pickle=True)
data_final = data.item()

verts = data_final['verts']
vertex_colors = data_final['vertex_colors']
face_indices = data_final['face_indices']
face_indices = face_indices.T
uvs = data_final['uvs']
face_uv_indices = data_final['face_uv_indices']
face_uv_indices = face_uv_indices.T
cam_eye = data_final['cam_eye']
cam_up = data_final['cam_up']
cam_lookat = data_final['cam_lookat']
ka = data_final['ka']
kd = data_final['kd']
ks = data_final['ks']
n = data_final['n']
light_positions = np.array(data_final['light_positions'])
light_intensities = np.array(data_final['light_intensities'])
Ia = data_final['Ia']
M = data_final['M']
N = data_final['N']
W = data_final['W']
H = data_final['H']
bg_color = data_final['bg_color']
focal = data_final['focal']
texture_map=Image.open("cat_diff.png")

#gouraud
image_gouraud_all=render_object("gouraud",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, kd, ks, n, light_positions, light_intensities, Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_gouraud_ambient=render_object("gouraud",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, 0, 0, n, light_positions, light_intensities, Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_gouraud_diffusion=render_object("gouraud",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
0, kd, 0, n, light_positions, light_intensities, Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_gouraud_specular=render_object("gouraud",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
0, 0, ks, n, light_positions, light_intensities, Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

#phong
image_phong_all=render_object("phong",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, kd, ks, n, light_positions, light_intensities, Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_phong_ambient=render_object("phong",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, 0, 0, n, light_positions, light_intensities, Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_phong_diffusion=render_object("phong",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
0, kd, 0, n, light_positions, light_intensities, Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_phong_specular=render_object("phong",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
0, 0, ks, n, light_positions, light_intensities, Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

#Different light sources all-gouraud
image_gouraud_all_l1=render_object("gouraud",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, kd, ks, n, [light_positions[0]], [light_intensities[0]], Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_gouraud_all_l2=render_object("gouraud",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, kd, ks, n, [light_positions[1]], [light_intensities[1]], Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_gouraud_all_l3=render_object("gouraud",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, kd, ks, n, [light_positions[2]], [light_intensities[2]], Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

#Different light sources all-phong
image_phong_all_l1=render_object("phong",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, kd, ks, n, [light_positions[0]], [light_intensities[0]], Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_phong_all_l2=render_object("phong",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, kd, ks, n, [light_positions[1]], [light_intensities[1]], Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image_phong_all_l3=render_object("phong",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, kd, ks, n, [light_positions[2]], [light_intensities[2]], Ia,
 uvs=None, face_uv_indices=None, texture_map=None)

image=render_object("texture_map",
focal, cam_eye, cam_lookat, cam_up, bg_color,
M,N,H,W,
verts, vertex_colors, face_indices,
ka, kd, ks, n, light_positions, light_intensities, Ia,
 uvs, face_uv_indices, texture_map)

#Gouraud Shading Techniques
fig, axs = plt.subplots(2, 2)
# Plot for Gouraud-All
axs[0, 0].imshow(image_gouraud_all)
axs[0, 0].set_title("All")

# Plot for Gouraud-Ambient
axs[0, 1].imshow(image_gouraud_ambient)
axs[0, 1].set_title("Ambient")

# Plot for Gouraud-Diffusion
axs[1, 0].imshow(image_gouraud_diffusion)
axs[1, 0].set_title("Diffusion")

# Plot for Gouraud-Specular
axs[1, 1].imshow(image_gouraud_specular)
axs[1, 1].set_title("Specular")
fig.suptitle("Gouraud Shading Techniques")
fig.tight_layout()

#Phong Shading Techniques
fig, axs = plt.subplots(2, 2)
# Plot for Phong-All
axs[0, 0].imshow(image_phong_all)
axs[0, 0].set_title("All")

# Plot for Phong-Ambient
axs[0, 1].imshow(image_phong_ambient)
axs[0, 1].set_title("Ambient")

# Plot for Phong-Diffusion
axs[1, 0].imshow(image_phong_diffusion)
axs[1, 0].set_title("Diffusion")

# Plot for Phong-Specular
axs[1, 1].imshow(image_phong_specular)
axs[1, 1].set_title("Specular")
fig.suptitle("Phong Shading Techniques")
fig.tight_layout()

fig, axs = plt.subplots(2, 2)

# Plot for Gouraud-All
axs[0, 0].imshow(image_gouraud_all)
axs[0, 0].set_title("All light sources")

# Plot for Gouraud-1 ls all
axs[0, 1].imshow(image_gouraud_all_l1)
axs[0, 1].set_title("1st light source")

# Plot for Gouraud-2 ls all
axs[1, 0].imshow(image_gouraud_all_l2)
axs[1, 0].set_title("2nd light source")

# Plot for Gouraud-3 ls all
axs[1, 1].imshow(image_gouraud_all_l3)
axs[1, 1].set_title("3rd light source")

fig.suptitle("Gouraud All Shading Techniques with different light sources")
fig.tight_layout()

fig, axs = plt.subplots(2, 2)
# Plot for phong-All
axs[0, 0].imshow(image_phong_all)
axs[0, 0].set_title("All light sources")

# Plot for phong-1 ls all
axs[0, 1].imshow(image_phong_all_l1)
axs[0, 1].set_title("1st light source")

# Plot for phong-2 ls all
axs[1, 0].imshow(image_gouraud_all_l2)
axs[1, 0].set_title("2nd light source")

# Plot for phong-3 ls all
axs[1, 1].imshow(image_phong_all_l3)
axs[1, 1].set_title("3rd light source")

fig.suptitle("Phong All Shading Techniques with different light sources")
fig.tight_layout()

plt.figure()
plt.imshow(image)
plt.title("All Shading Techniques with all the light sources using phong texture map")

plt.show()


