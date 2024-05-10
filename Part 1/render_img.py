import numpy as np
from g_shading import g_shading
from f_shading import f_shading

def render_img(faces, vertices, vcolors, depth, shading):
    # Dhmioyrgia eikonas
    M = 512
    N = 512
    image = np.ones((M, N, 3))

    # Ypologismos bathous gia kathe trigwno kai taksinomhsh trigwnwn
    depth_triangle = np.zeros(faces.shape[0])
    for i in range(faces.shape[0]):
        depth_triangle[i] = np.mean(depth[faces[i]])
    indices = np.argsort(depth_triangle)[::-1] #vectors kata fthinousa taksinomhsh
    faces = faces[indices] 

    # Xrwmatismos trigwnwn
    for i in range(faces.shape[0]):
        vertices_triangle = vertices[faces[i]]
        vcolors_triangle = vcolors[faces[i]]
        if shading == "g":
            g_shading(image, vertices_triangle, vcolors_triangle)
        elif shading == "f":
            f_shading(image, vertices_triangle, vcolors_triangle)
    return image


