import numpy as np
from g_shading import g_shading
from functions_part2 import perspective_project
from functions_part2 import lookat
from functions_part2 import rasterize
from light import light

def calculate_normals(verts, faces):
    normal_vector = np.zeros(verts.shape)

    for i in range(faces.shape[0]):
        # Calculate vertices for current triangle
        A = verts[:, faces[i][0]]
        B = verts[:, faces[i][1]]
        C = verts[:, faces[i][2]]

        # edges
        AB = B - A
        AC = C - A
    
        # Calculate the normal for the current triangle
        normal = np.cross(AB, AC)

        # Add the normal to each vertex of the triangle
        normal_vector[:, faces[i][0]] += normal
        normal_vector[:, faces[i][1]] += normal
        normal_vector[:, faces[i][2]] += normal
    
    # Normalize the normals for each vertex
    for j in range(normal_vector.shape[1]):
        normal_length = np.linalg.norm(normal_vector[:, j])
        if normal_length != 0:
            normal_vector[:, j] /= normal_length

    return normal_vector


def shade_gouraud(vertsp, vertsn, vertsc, bcoords, cam_pos, ka, kd, ks, n, lpos, lint, lamb,X):
    final_color=np.zeros((3,3))
    for i in range(3):
        final_color[i]=light(bcoords,vertsn[i],vertsc[i],cam_pos,ka,kd,ks,n,lpos,lint,lamb)
    g_shading(X, vertsp, final_color, "gouraud", bcoords=None, campos=None, ka=None, kd=None, ks=None, n=None, lpos=None, lint=None, lamb=None, vertsn=None,texture_map=None,uv_triangle=None)
    return X

def shade_phong(vertsp, vertsn, vertsc, bcoords, campos, ka, kd, ks, n, lpos, lint, lamb, X):
    g_shading(X, vertsp, vertsc,"phong",bcoords, campos, ka, kd, ks, n, lpos, lint, lamb,vertsn,texture_map=None,uv_triangle=None)
    return X


def render_object(shader,
focal, eye, look, up, bg_color,
M,N,H,W,
verts, vert_colors, faces,
ka, kd, ks, n, lpos, lint, lamb,uvs,face_uv_indices,texture_map):
    
    normals=calculate_normals(verts,faces)

    # render the specified object from the specified camera. 
    R,d=lookat(eye,up,look)
    pts_2d,depth=perspective_project(verts,focal,R,d)
    pixels=rasterize(pts_2d,H,W,M,N)
    pixels=pixels.T

    # Create image
    image = np.ones((M, N, 3)) * bg_color

    depth_triangle = np.zeros(faces.shape[0])
    for i in range(faces.shape[0]):
        depth_triangle[i] = np.mean(depth[faces[i]])
    indices = np.argsort(depth_triangle)[::-1] #vectors in descending order
    faces = faces[indices]
    depth=depth_triangle[indices]
    if shader=="texture_map":
        face_uv_indices=face_uv_indices[indices]
        texture_map = np.array(texture_map)/255

    # Coloring triangles
    for i in range(faces.shape[0]):

        vert_tr=verts.T[faces[i]]
        vertices_triangle = pixels[faces[i]]
        vcolors_triangle = vert_colors.T[faces[i]] 
        normals_triangle=normals.T[faces[i]]

        bcoords=(vert_tr[0]+vert_tr[1]+vert_tr[2])/3 #calculate centroid of the triangle

        if shader=="gouraud":
            image=shade_gouraud(vertices_triangle, normals_triangle, vcolors_triangle, bcoords, eye, ka, kd, ks, n, lpos, lint, lamb,image)
        elif shader=="phong":
            image=shade_phong(vertices_triangle, normals_triangle, vcolors_triangle, bcoords, eye, ka, kd, ks, n, lpos, lint, lamb,image) 
        elif shader=="texture_map":
            uv_triangle = uvs.T[face_uv_indices[i]]
            g_shading(image, vertices_triangle, vcolors_triangle,"texture map",bcoords, eye, ka, kd, ks, n, lpos, lint, lamb,normals_triangle,texture_map,uv_triangle)    
    return image 

