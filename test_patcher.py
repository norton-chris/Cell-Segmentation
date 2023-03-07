from Patcher import Patcher
from PIL import Image
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from tiler import tiler
import numpy as np

img = Image.open("img1.tif")
img = np.array(img)

tiles=tiler(img, 385, 385,2)
reImg=tiles.tile(list(tiles))

fig,ax = plt.subplots(5,4)



ax[0,0].imshow(img,cmap='tab20c')
ax[0,0].set_axis_off()
ax[0,1].imshow(reImg,cmap='tab20c')
ax[0,1].set_axis_off()
c=0
for i in range(1,5):
    for j in range(0,4):
        ax[i,j].imshow(tiles[c],cmap='tab20c')
        ax[i,j].set_axis_off()
        c+=1

plt.show()
