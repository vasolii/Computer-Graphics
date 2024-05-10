from render_img import render_img
import numpy as np
import matplotlib.pyplot as plt

#fortwsh dedomenwnapo to hw1.npy
data = np.load('hw1.npy', allow_pickle=True)
data_final = data.item()

#topothethsh dedomenwn stoys analogoys pinakes
faces = data_final['faces']
vertices = data_final['vertices']
vcolors = data_final['vcolors']
depth = data_final['depth']

image=render_img(faces,vertices,vcolors,depth,"g")

# Kathorismos onomatos eikonas
filename = 'demo_g.jpg'

# Apothikeysh ths eikonas
plt.imsave(filename, image)