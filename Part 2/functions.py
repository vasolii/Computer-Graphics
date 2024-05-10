import numpy as np
import math
from g_shading import g_shading
from g_shading import slope
from vector_interp import vector_interp

class Transform:
# Interface for performing affine transformations.
    def __init__(self):
        # Initialize a Transform object.
        self.mat = np.eye(4)

    def rotate(self, theta: float, u: np.ndarray) -> None:
        # rotate the transformation matrix
        R = np.zeros((3, 3)) #Initialize transformation table

        #Shortcuts for useful constants for use in formulas
        c = math.cos(theta)
        s = math.sin(theta)
        ux = u[0]
        uy = u[1]
        uz = u[2]

        #Defining the elements of the matrix R by formula 5.45
        R[0][0]=(1-c)*ux**2+c
        R[0][1]=(1-c)*ux*uy-s*uz
        R[0][2]=(1-c)*ux*uz+s*uy

        R[1][0]=(1-c)*uy*ux+s*uz
        R[1][1]=(1-c)*uy**2+c
        R[1][2]=(1-c)*uy*uz-s*ux

        R[2][0]=(1-c)*uz*ux - s*uy
        R[2][1]=(1-c)*uz*uy+s*ux
        R[2][2]=(1-c)*uz**2+c

        #In homogeneous coordinates
        Rh = np.zeros((4, 4))
        Rh[:3, :3] = R
        Rh[3, 3] = 1

        self.mat=np.dot(Rh,self.mat) #Update of table mat

    def translate(self, t: np.ndarray) -> None:
        # translate the transformation matrix

        #Determination of the matrix Th based on formula 5.37
        Th = np.eye(4)
        Th[0:3,3:4]=t

        self.mat=np.dot(Th,self.mat) #Update of table mat

    def transform_pts(self, pts: np.ndarray) -> np.ndarray:
        # transform the specified points according to our current matrix
        pts_omogeneis=np.ones((4,pts.shape[1]))
        pts_omogeneis[0:3,0:pts.shape[1]+1]=pts
        pts_transformed = np.dot(self.mat,pts_omogeneis)
        return pts_transformed[0:3,:]

def world2view(pts: np.ndarray, R: np.ndarray, c0: np.ndarray) -> np.ndarray:
    pts_homogenous=np.concatenate((pts, np.ones((1, pts.shape[1]))), axis=0) #conversion to homogeneous coordinates

    #Define transformation matrix
    arr = np.zeros((4, 4))
    arr[0:3, 0:3] = R.T
    arr[0:3, 3:4] = -np.dot(R.T, c0)
    arr[3, 3] = 1

    homogenous_coordinates_c=np.dot(arr,pts_homogenous) # formula 6.13

    c_new = homogenous_coordinates_c[:3, :homogenous_coordinates_c.shape[1]] #Return to camera coordinates
    return c_new

def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray) -> tuple [np.ndarray,np.ndarray]:
    ck=target-eye
    ck_norm=np.linalg.norm(ck)
    zc=ck/ck_norm #τυπος 6.6

    t=up-np.dot(up.T,zc)*zc
    yc=t/np.linalg.norm(t) #formula 6.7

    xc=np.cross(zc.T, yc.T) #formula 6.8
    R= np.column_stack((xc.T, yc, zc)) # join arrays as columns, type 6.11
    d=eye #formula 6.12
    parameters=(R,d)
    return parameters

def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray) -> tuple [np.ndarray, np.ndarray]:
    # Project the specified 3d points pts on the image plane, according to a pinhole perspective projection model.
    arr=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
   
    pts_new=world2view(pts,R,t) #transform the coordinates in the camera system 
    final_coordinates=np.zeros((4,pts_new.shape[1])) #initialize array of final homogeneous coordinates of all points of pts_new

    #loop for each point of the pts_new array separately and put in an array
    for i in range(0,pts_new.shape[1]):
        arr_p=np.array([[pts_new[0][i]*focal],[pts_new[1][i]*focal],[pts_new[2][i]],[pts_new[2][i]*focal]])
        arr_p=arr_p/pts_new[2][i] #divide by d=zp
        final_coordinates[0:4,i:i+1]=np.dot(arr,arr_p) #type 6.4 without the assumption that w=1 (in our case w=focal)
    #we are only interested in the x,y coordinates, so placing only those in a tuple   
    pts_2d=(final_coordinates[0:2,:],pts_new[2][:])
    return pts_2d

def rasterize(pts_2d: np.ndarray, plane_w: int, plane_h: int, res_w: int, res_h: int) -> np.ndarray:
    # Rasterize the incoming 2d points from the camera plane to image pixel coordinates
    x_2d=pts_2d[0,:]
    y_2d=pts_2d[1,:]
    pixel=np.zeros((2,x_2d.shape[0]))
    for i in range (0,x_2d.shape[0]): 
        x=x_2d[i]+plane_w/2
        y=y_2d[i]+plane_h/2 #new center the end bottom and left point
        pixel[0,i]=int(x*res_w/plane_w)
        pixel[1,i]=int(y*res_h/plane_h) #find the corresponding pixel
    return pixel      

def render_object(v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target) -> np.ndarray:
    # render the specified object from the specified camera.  
    R,d=lookat(eye,up,target)
    pts_2d,depth=perspective_project(v_pos,focal,R,d)
    pixels=rasterize(pts_2d,plane_w,plane_h,res_w,res_h)
    pixels=pixels.T # to match the dimensions with how the code is implemented 
    # Create image
    image = np.ones((res_h, res_w, 3))

    max_x = res_w
    max_y = res_h

    depth_triangle = np.zeros(t_pos_idx.shape[0])
    for i in range(t_pos_idx.shape[0]):
        depth_triangle[i] = np.mean(depth[t_pos_idx[i]])
    indices = np.argsort(depth_triangle)[::-1] #vectors in descending order
    t_pos_idx = t_pos_idx[indices]
    depth=depth_triangle[indices]
    

    # Coloring triangles
    for i in range(t_pos_idx.shape[0]):
        vertices_triangle = pixels[t_pos_idx[i]]
        vcolors_triangle = v_clr[t_pos_idx[i]]
        if (vertices_triangle[:, 0].max() > max_x):
            clipping(max_x,vertices_triangle,0,vcolors_triangle,image)
        elif (vertices_triangle[:, 1].max() > max_y):
            clipping(max_y,vertices_triangle,1,vcolors_triangle,image)
        else:    
            g_shading(image, vertices_triangle, vcolors_triangle)
    return image    

def clipping(lim,vertices_triangle,option,vcolors_triangle,image): #x=0,y=1 for option

        if option==0:
            other=1
        else: other=0  

        count=count_elements(vertices_triangle[:,option],lim)

        #case we have a two corners except
        if count == 2:
            new_vertices_triangle=np.zeros((3,2))
            new_vcolors_triangle=np.zeros((3,3))

            for i in range(0,3):#finds the vertex that is inside and places it in the new table with its color and coordinates
                if vertices_triangle[i][option]<lim:
                    new_vertices_triangle[0][:]=vertices_triangle[i][:]
                    new_vcolors_triangle[0][:]=vcolors_triangle[i][:]
            k=1 
            for i in range (0,3):# finds the coordinates of the new two vertices and the corresponding colors       
                if vertices_triangle[i][option]>=lim:
                    sl=slope(new_vertices_triangle[0][0],new_vertices_triangle[0][1],vertices_triangle[i][0],vertices_triangle[i][1])
                    if sl<0: #to be the point inside the triangle
                        new_vertices_triangle[k][other]=math.floor(calculate_coordinates(sl,new_vertices_triangle[0][0],new_vertices_triangle[0][1],option,lim))
                    else:
                        new_vertices_triangle[k][other]=math.ceil(calculate_coordinates(sl,new_vertices_triangle[0][0],new_vertices_triangle[0][1],option,lim))                    
                    new_vertices_triangle[k][option]=lim

                    #for coloring   
                    if new_vertices_triangle[0][1]==vertices_triangle[i][1]:
                        #cover the case of division by zero internally in vector_interp
                        new_vcolors_triangle[k][:]=vector_interp(new_vertices_triangle[0][:],vertices_triangle[i][:],new_vcolors_triangle[0][:],vcolors_triangle[i][:],new_vertices_triangle[k][0],1)
                    else:           
                        new_vcolors_triangle[k][:]=vector_interp(new_vertices_triangle[0][:],vertices_triangle[i][:],new_vcolors_triangle[0][:],vcolors_triangle[i][:],new_vertices_triangle[k][1],2)                      
                    k=k+1                                     
            g_shading(image,new_vertices_triangle,new_vcolors_triangle) 

        elif count== 1:  # in case I have a corner off
            new_vertices_triangle=np.zeros((4,2))
            new_vcolors_triangle=np.zeros((4,3))

            k=0
            for i in range(0,3): #finds the vertex that is inside and places it in the new table with its color and coordinates
                if vertices_triangle[i][option]<lim:
                    new_vertices_triangle[k][:]=vertices_triangle[i][:]
                    new_vcolors_triangle[k][:]=vcolors_triangle[i][:] 
                    k=k+1         

            for i in range (0,3):# finds the coordinates of the new two vertices and the corresponding colors      
                if vertices_triangle[i][option]>=lim:
                    sl=slope(new_vertices_triangle[0][0],new_vertices_triangle[0][1],vertices_triangle[i][0],vertices_triangle[i][1])
                    if sl<0:
                        new_vertices_triangle[2][other]=math.floor(calculate_coordinates(sl,new_vertices_triangle[0][0],new_vertices_triangle[0][1],option,lim))
                    else:
                        new_vertices_triangle[2][other]=math.ceil(calculate_coordinates(sl,new_vertices_triangle[0][0],new_vertices_triangle[0][1],option,lim))
                    new_vertices_triangle[2][option]=lim

                    #for coloring  
                    if new_vertices_triangle[0][1]==vertices_triangle[i][1]:
                        #cover the case of division by zero internally in vector_interp
                        new_vcolors_triangle[2][:]=vector_interp(new_vertices_triangle[0][:],vertices_triangle[i][:],new_vcolors_triangle[0][:],vcolors_triangle[i][:],new_vertices_triangle[2][0],1)
                    else:            
                        new_vcolors_triangle[2][:]=vector_interp(new_vertices_triangle[0][:],vertices_triangle[i][:],new_vcolors_triangle[0][:],vcolors_triangle[i][:],new_vertices_triangle[2][1],2)                           
                    
                    #for next point
                    sl=slope(new_vertices_triangle[1][0],new_vertices_triangle[1][1],vertices_triangle[i][0],vertices_triangle[i][1])
                    if sl<0:
                        new_vertices_triangle[3][other]=math.floor(calculate_coordinates(sl,new_vertices_triangle[1][0],new_vertices_triangle[1][1],option,lim))
                    else:
                        new_vertices_triangle[3][other]=math.ceil(calculate_coordinates(sl,new_vertices_triangle[1][0],new_vertices_triangle[1][1],option,lim))
                    new_vertices_triangle[3][option]=lim

                    #for coloring
                    if new_vertices_triangle[0][1]==vertices_triangle[i][1]:
                        #cover the case of division by zero internally in vector_interp
                        new_vcolors_triangle[3][:]=vector_interp(new_vertices_triangle[0][:],vertices_triangle[i][:],new_vcolors_triangle[0][:],vcolors_triangle[i][:],new_vertices_triangle[3][0],1)
                    else:            
                        new_vcolors_triangle[3][:]=vector_interp(new_vertices_triangle[0][:],vertices_triangle[i][:],new_vcolors_triangle[0][:],vcolors_triangle[i][:],new_vertices_triangle[3][1],2)                                               
            #the 2 new triangles
            triangle1=new_vertices_triangle[[0, 1, 2], :]
            colors1=new_vcolors_triangle[[0, 1, 2], :]
            triangle2=new_vertices_triangle[[1, 2, 3], :] 
            colors2=new_vcolors_triangle[[1, 2, 3], :] 

            if np.array_equal(new_vertices_triangle[2][:], new_vertices_triangle[3][:]):# in case the two triangles coincide
                g_shading(image,triangle1,colors1)
            else:      
                g_shading(image,triangle1,colors1)  
                g_shading(image,triangle2,colors2)     

def calculate_coordinates(slope,x,y,option,known_cor):
    if option==1 : # for x
        return (1 / slope) * (known_cor - y) + x
    elif option==0: # for y
        return slope * (known_cor - x) + y   
    
def count_elements(column, lim): #counts vertices that are out
    count = 0
    for i in range(0,column.shape[0]):
        if column[i] >= lim:
            count += 1
    return count                      