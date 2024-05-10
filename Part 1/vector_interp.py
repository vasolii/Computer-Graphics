import numpy as np
def vector_interp(p1,p2,V1,V2,coord,dim):

    p1 = np.array(p1)
    p2 = np.array(p2)
    V1 = np.array(V1)
    V2 = np.array(V2)
    #periptwsh poy to coord einai x
    if dim==1:
        l=(coord-p1[0])/(p2[0]-p1[0])
    #periptwsh poy to coord einai y    
    if dim==2:
        l=(coord-p1[1])/(p2[1]-p1[1])
    V=V1+(V2-V1)*l   
    return V 
  