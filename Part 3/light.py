import numpy as np
def light(point, normal, vcolor, cam_pos, ka, kd, ks, n, lpos, lint,lamb):

    num_light=len(lint)
    N=normal

    #Calculate and normalize the vector V
    V=cam_pos-point
    V=V/np.linalg.norm(V)

    #Initialize sum of diffuse and specular component
    sum_I = np.zeros(3)

    #for every light source
    for i in range(num_light):
        #Calculate and normalize the vectors L
        L=lpos[i]-point
        L=L/np.linalg.norm(L)

        # Calculate the reflection vector
        R = 2 * np.dot(N, L) * N - L
        R = R / np.linalg.norm(R)

        # Diffuse component
        diffuse = kd * np.dot(N, L) * lint[i]
        diffuse = np.maximum(diffuse, 0) 

        # Specular component
        specular = ks * (np.dot(R, V) ** n) * lint[i]
        specular = np.maximum(specular, 0) 

        # Sum contributions from this light source
        sum_I += diffuse + specular

    ambient=lamb*ka  
    ambient = np.maximum(ambient, 0)  
    I=ambient+sum_I
    final_color=I*vcolor
    
    # Clip the final color to the range [0, 1]
    final_color = np.clip(final_color, 0, 1)

    return final_color