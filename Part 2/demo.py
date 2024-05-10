import numpy as np
import matplotlib.pyplot as plt
from functions import render_object
from functions import Transform

#load data from hw2.npy
data = np.load('hw2.npy', allow_pickle=True)
data_final = data.item()

#Placing data in the corresponding tables
v_pos = data_final['v_pos']
v_clr = data_final['v_clr']
t_pos_idx = data_final['t_pos_idx']
eye = data_final['eye']
up = data_final['up']
target = data_final['target']
focal= data_final['focal']
plane_w= data_final['plane_w']
plane_h= data_final['plane_h']
res_w= data_final['res_w']
res_h= data_final['res_h']
theta_0= data_final['theta_0']
rot_axis_0= data_final['rot_axis_0']
t_0 = data_final['t_0'].reshape(-1, 1)
t_1 = data_final['t_1'].reshape(-1, 1) 

"""They are converted to vertical vectors to have the correct dimension as arguments to the .translate() function.
Also, in the hw2.npy file, the number of columns is not set to 1, causing errors afterwards.
 Thus, it is specified here"""

#fish without rotation or translation
image=render_object(v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)
plt.imshow(image)
plt.savefig('0.jpg')

#A)
transform=Transform()
transform.rotate(theta_0,rot_axis_0)
transform_v_pos=transform.transform_pts(v_pos)
A=render_object(transform_v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)
plt.imshow(A)
plt.savefig('1.jpg')

#B)
transform3=Transform()
transform3.translate(t_0)
transform_v_pos3=transform3.transform_pts(transform_v_pos)
B=render_object(transform_v_pos3, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)
plt.imshow(B)
plt.savefig('2.jpg')

#C)
transform4=Transform()
transform4.translate(t_1)
transform_v_pos4=transform4.transform_pts(transform_v_pos3)
C=render_object(transform_v_pos4, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)
plt.imshow(C)
plt.savefig('3.jpg')

#Example with clipping
"""
transform=Transform()
transform.rotate(-theta_0,rot_axis_0)
transform_v_pos=transform.transform_pts(v_pos)

transform.translate(-t_0)
transform_v_pos=transform.transform_pts(transform_v_pos)

transform.translate(-t_1)
transform_v_pos=transform.transform_pts(transform_v_pos)
image_clipping=render_object(transform_v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)
plt.imshow(image_clipping)
plt.show()"""