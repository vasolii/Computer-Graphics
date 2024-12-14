import numpy as np
from vector_interp import vector_interp

#Perform bilinear interpolation to sample a texture map at given UV coordinates
def bilerp(uv, texture_map):
    M, N, _ = texture_map.shape
    u, v = uv * np.array([N, M])

    # Floor and ceil values
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, N - 1)
    x1 = np.clip(x1, 0, M - 1)
    y0 = np.clip(y0, 0, N- 1)
    y1 = np.clip(y1, 0, M - 1)

    # Get pixel values
    Ia = texture_map[y0, x0]
    Ib = texture_map[y0, x1]
    Ic = texture_map[y1, x0]
    Id = texture_map[y1, x1]

    # Average the four pixel values
    color=(Ia/4 + Ib/4 + Ic/4 + Id/4)

    return color

