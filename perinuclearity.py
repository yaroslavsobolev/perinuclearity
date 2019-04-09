import skimage
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from shapely.geometry import LinearRing, LineString
from scipy.ndimage import center_of_mass

target_nucleus_file = 'data/MCF7 80-20 24h/11 MCF7 8020 24h exp#2 Black nucleous.tif'
target_periphery_file = 'data/MCF7 80-20 24h/11 MCF7 8020 24h exp#2 Black outside.tif'
nucleus_image = skimage.io.imread(target_nucleus_file)
nucleus_image = skimage.util.invert(nucleus_image)
nucleus_contour = measure.find_contours(nucleus_image[:,:,0], 10)[0]
nucleus_center = center_of_mass(nucleus_image[:,:,0])
nucleus_contour_shape = LinearRing(nucleus_contour)

periphery_image = skimage.io.imread(target_periphery_file)
periphery_contour = measure.find_contours(periphery_image[:,:,0], 10)[0]
periphery_contour_shape = LinearRing(periphery_contour)

target_point = np.array((579, 469))
raylength = 1000
ray1 = LineString(((nucleus_center[0], nucleus_center[0]),
                  nucleus_center + raylength/np.linalg.norm(target_point-nucleus_center)*(target_point-nucleus_center)))
int_point = ray1.intersection(nucleus_contour_shape)
nucl_intersection = np.array((int_point.x, int_point.y))
int_point = ray1.intersection(periphery_contour_shape)
peri_intersection = np.array((int_point.x, int_point.y))
OA = np.linalg.norm(nucl_intersection-nucleus_center)
OT = np.linalg.norm(target_point-nucleus_center)
OB = np.linalg.norm(peri_intersection-nucleus_center)
if OT < OA:
    norm_metrics = -1*(OA-OT)/OA
else:
    norm_metrics = (OT-OA)/(OB-OA)
print(norm_metrics)
# Display the image and plot all contours found
fig, ax = plt.subplots()
dic = skimage.io.imread('data/MCF7 80-20 24h/11 MCF7 8020 24h exp#2 bg400 td l250-1500_RGB.bmp')
ax.imshow(np.transpose(dic, axes=(1,0,2)), alpha=0.6) #

ax.plot(nucleus_contour[:, 0], nucleus_contour[:, 1], linewidth=2)
peri_for_plot = np.vstack([periphery_contour, periphery_contour[0,:]])
# ax.plot(periphery_contour[:, 0], periphery_contour[:, 1], linewidth=2)
ax.plot(peri_for_plot[:, 0], peri_for_plot[:, 1], linewidth=2)

ax.plot(ray1.xy[0], ray1.xy[1])
ax.plot(nucl_intersection[0], nucl_intersection[1], 'o', color='blue')
ax.plot(peri_intersection[0], peri_intersection[1], 'o', color='red')
ax.plot(target_point[0], target_point[1], 'o', color='green', markersize=10)
ax.plot(nucleus_center[0], nucleus_center[1], 'o', color='yellow')
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
fig.savefig('illustration1.png', dpi=400)
plt.show()