import numpy as np
import math
from vector_interp import vector_interp

def g_shading(img, vertices, vcolors):

    #arxikopoihsh pinakwn
    y_min = np.zeros(3)
    y_max = np.zeros(3)
    x_max = np.zeros(3)
    x_min = np.zeros(3)
    slopes = np.zeros(3)

    for k in range(3):
        l = (k + 1) % 3 # kalyptetai etsi h periptwsh k=2 wste h epomenh pleyra na einai h 1
        y_min[k] = min(vertices[k][1], vertices[l][1])
        y_max[k] = max(vertices[k][1], vertices[l][1])
        x_max[k] = max(vertices[k][0], vertices[l][0])
        x_min[k] = min(vertices[k][0], vertices[l][0])
        slopes[k] = slope(vertices[k][0],vertices[k][1],vertices[l][0],vertices[l][1])

    ymin=int(min(y_min)) #tetagmenh ths pio xamhlhs koryfhs
    ymax=int(max(y_max)) #tetagmenh ths pio ypshlhs koryfhs 
    y=ymin #grammh sarwshs

    #objects -edges- x_values
    edgeA=x_values_edges(0,slopes[0],0,0,y_min[0],y_max[0],x_min[0],x_max[0])
    edgeB=x_values_edges(1,slopes[1],0,0,y_min[1],y_max[1],x_min[1],x_max[1])
    edgeC=x_values_edges(2,slopes[2],0,0,y_min[2],y_max[2],x_min[2],x_max[2])
    allEdges=[edgeA, edgeB, edgeC]

    #activeEdges
    active_edges=[]
    for k in range(3):
        if y_min[k]==y:
            active_edges.append(k)
        if y_max[k]==y:
            active_edges.remove(k)  

    #ActiveShmeia
    for edge in allEdges:
        if edge.edge in active_edges:
                if edge.slope<=0: # an h pleyra exei arnhtikh klhsh
                    edge.statement=1
                    edge.xValue=edge.xmax 
                if edge.slope>0: # an h pleyra exei thetikh klhsh
                    edge.statement=1
                    edge.xValue=edge.xmin                                     

    #LOOP GIA DRAW          
    for y in range(ymin, ymax,1):#mexri y max => h oriakes times den xrwmatizontai
        x_values=[]
        #topothethsh tetmhmenwn energwn shmeiwn se enan neo pinaka
        for edge in allEdges:
            if edge.statement==1:
                x_values.append(edge.xValue)  
        x_values.sort()
 
        x_values[0]=math.ceil(x_values[0])
        x_values[1]=math.ceil(x_values[1])
        #arxh sarwshs
        for i in range(1, 2, 2):
            for x in range((x_values[0]),x_values[1]):
                img[y][x]=pixel_color(y,allEdges,active_edges,vertices,vcolors,x) #drawpixel
        #kalesma gia ananewsh energwn shmeiwn
        allEdges,active_edges=active_boundary_points(y,allEdges,active_edges,vertices)    

#Energa shmeia
def active_boundary_points(y,allEdges,active_edges,vertices):
    new_active_edges=[]
    #sthn periptwsh neas energhs akmhs
    for k in range(3):
        for edge in allEdges:
            if edge.ymin==y+1 and k not in active_edges and edge.edge==k:
                new_active_edges.append(k) 
                if vertices[edge.edge][1]==y+1:
                    edge.xValue=vertices[edge.edge][0] #?
                else:
                    l = (k + 1) % 3
                    edge.xValue=vertices[l][0] 
            #sthn periptwsh afaireshs akmhs
            if edge.ymax==y+1 and k in active_edges and edge.edge==k:
                if edge.edge in new_active_edges:
                    new_active_edges.remove(edge.edge)
                active_edges.remove(edge.edge)
                edge.statement=0
            #gia oles tis ypoloipes yparxouses akmes poy den kalyfthkan apo tis parapanw periptwseis
            if edge.edge==k and k in active_edges:
                if edge.slope != float('inf') and edge.slope != 0:
                    edge.xValue=(edge.xValue + (1 / edge.slope))                     
    active_edges.extend(new_active_edges) 
    #diaxeirhsh statement analoga me tis parapanw allages
    for edge in allEdges: 
        if edge.edge in active_edges:
            edge.statement=1
        if edge.edge not in active_edges:
            edge.statement=0    

    return allEdges, active_edges   
        
#Synarthsh eyreyshs klhshs
def slope(x1, y1, x2, y2):
    if x2==x1:
        s=float("inf")
    else:    
        s = (y2-y1)/(x2-x1)
    return s    

#Synarthsh eyreshs xrwmatos kathe pixel
def pixel_color(y,allEdges,active_edges,vertices,vcolors,x):
    V = np.zeros((2,3))
    final_color=np.zeros((1,3))
    xV=np.zeros(2)
    i=0
    for edge in allEdges:
        if edge.edge in active_edges:
            k=edge.edge
            l = (k + 1) % 3
            # eyresh xrwmatos syneytheikwn kata ton aksona y shmeiwn
            if vertices[k][1]==vertices[l][1]:
                #kalyptetai h periptwsh diaireshs me to mhden eswterika ths vector_interp
                V[i][:]=vector_interp(vertices[k][:],vertices[l][:],vcolors[k][:],vcolors[l][:],x,1)
            else:            
                V[i][:]=vector_interp(vertices[k][:],vertices[l][:],vcolors[k][:],vcolors[l][:],y,2)
            xV[i]=math.ceil(edge.xValue)
            i=i+1     
    #eyresh telikoy xrwmatos                        
    final_color=vector_interp((xV[0],y),(xV[1],y),V[0][:],V[1][:],x,1)
    return final_color

class x_values_edges:
    def __init__(self, edge, slope, xValue,statement,ymin,ymax,xmin,xmax):
        self.edge=edge
        self.slope=slope
        self.xValue=xValue
        self.statement=statement
        self.ymin=ymin
        self.ymax=ymax
        self.xmin=xmin
        self.xmax=xmax

