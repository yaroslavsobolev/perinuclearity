import skimage
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from shapely.geometry import LinearRing, LineString
from scipy.ndimage import center_of_mass
import tifffile as tiff

def load_nucleus_and_periphery_from_files(target_nucleus_file, target_periphery_file, do_plotting=False):
    nucleus_image = skimage.io.imread(target_nucleus_file)
    nucleus_image = skimage.util.invert(nucleus_image)
    nucleus_contour = measure.find_contours(nucleus_image[:,:,0], 100)[0]
    nucleus_center = center_of_mass(nucleus_image[:,:,0])
    nucleus_contour_shape = LinearRing(nucleus_contour)
    periphery_image = skimage.io.imread(target_periphery_file)
    periphery_contours = measure.find_contours(periphery_image[:,:,0], 100)
    periphery_contour = np.vstack(periphery_contours)
    # for contour in periphery_contours:

    periphery_contour_shape = LinearRing(periphery_contour)
    if do_plotting:
        plt.plot(nucleus_center[0], nucleus_center[1], 'o', color='#2ca02c', alpha=0.7)
        plt.plot(nucleus_contour[:, 0], nucleus_contour[:, 1], linewidth=2, color = '#1f77b4', alpha=0.7)
        peri_for_plot = np.vstack([periphery_contour, periphery_contour[0, :]])
        # ax.plot(periphery_contour[:, 0], periphery_contour[:, 1], linewidth=2)
        plt.plot(peri_for_plot[:, 0], peri_for_plot[:, 1], linewidth=2, color = '#ff7f0e', alpha=0.7)
    return nucleus_center, nucleus_contour_shape, periphery_contour_shape

def get_perinuclearity_for_point(target_point, nucleus_center, nucleus_contour_shape,
                                 periphery_contour_shape, raylength=5000, do_plotting=False):
    ray1 = LineString(((nucleus_center[0], nucleus_center[1]),
                      nucleus_center + raylength/np.linalg.norm(target_point-nucleus_center)*(target_point-nucleus_center)))
    int_point = ray1.intersection(nucleus_contour_shape)
    if int_point.geom_type == 'MultiPoint':
        distances_to_center = np.array([np.linalg.norm(np.array((point.x, point.y))-nucleus_center)
                                        for point in int_point])
        point_furthest_from_center = int_point[np.argmax(distances_to_center)]
        nucl_intersection = np.array((point_furthest_from_center.x,
                                      point_furthest_from_center.y))
    else:
        try:
            nucl_intersection = np.array((int_point.x, int_point.y))
        except AttributeError:
            plt.plot(ray1.xy[0], ray1.xy[1])
            # plt.plot(nucl_intersection[0], nucl_intersection[1], 'o', color='blue')
            # plt.plot(peri_intersection[0], peri_intersection[1], 'o', color='red')
            plt.plot(target_point[0], target_point[1], 'o', color='green', markersize=10)
            plt.show()
            raise AttributeError
    int_point = ray1.intersection(periphery_contour_shape)
    if int_point.geom_type == 'MultiPoint':
        distances_to_center = np.array([np.linalg.norm(np.array((point.x, point.y))-nucleus_center)
                                        for point in int_point])
        point_furthest_from_center = int_point[np.argmax(distances_to_center)]
        peri_intersection = np.array((point_furthest_from_center.x,
                                      point_furthest_from_center.y))
    else:
        # try:
        peri_intersection = np.array((int_point.x, int_point.y))
        # except AttributeError:
        #     plt.plot(ray1.xy[0], ray1.xy[1])
        #     # plt.plot(nucl_intersection[0], nucl_intersection[1], 'o', color='blue')
        #     # plt.plot(peri_intersection[0], peri_intersection[1], 'o', color='red')
        #     plt.plot(target_point[0], target_point[1], 'o', color='green', markersize=10)
        #     plt.show()
        #     raise AttributeError
    OA = np.linalg.norm(nucl_intersection-nucleus_center)
    OT = np.linalg.norm(target_point-nucleus_center)
    OB = np.linalg.norm(peri_intersection-nucleus_center)
    if OT < OA:
        norm_metrics = -1*(OA-OT)/OA
    else:
        norm_metrics = (OT-OA)/(OB-OA)
    if do_plotting:
        plt.plot(ray1.xy[0], ray1.xy[1])
        plt.plot(nucl_intersection[0], nucl_intersection[1], 'o', color='blue')
        plt.plot(peri_intersection[0], peri_intersection[1], 'o', color='red')
        plt.plot(target_point[0], target_point[1], 'o', color='green', markersize=10)
    return norm_metrics

# Display the image and make illustration for perinuclearity
def test_perinuclearity(target_nucleus_file, target_periphery_file, target_dic_file):
    fig, ax = plt.subplots()
    dic = skimage.io.imread(target_dic_file, plugin='pil')
    ax.imshow(np.transpose(dic, axes=(1,0,2)), alpha=0.6) #
    nucleus_center, nucleus_contour_shape, periphery_contour_shape = \
        load_nucleus_and_periphery_from_files(target_nucleus_file, target_periphery_file,
                                              do_plotting=True)
    plt.show()
    target_point = np.array((579, 469))
    P = get_perinuclearity_for_point(target_point, nucleus_center, nucleus_contour_shape,
                                     periphery_contour_shape, do_plotting=True)
    # ax.plot(ray1.xy[0], ray1.xy[1])
    # ax.plot(nucl_intersection[0], nucl_intersection[1], 'o', color='blue')
    # ax.plot(peri_intersection[0], peri_intersection[1], 'o', color='red')
    # ax.plot(target_point[0], target_point[1], 'o', color='green', markersize=10)
    ax.plot(nucleus_center[0], nucleus_center[1], 'o', color='yellow')
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('illustration2.png', dpi=400)
    # plt.show()

def plot_perinuclearity_isolevels():
    f0 = plt.figure(1)
    target_nucleus_file = 'data/MCF7 80-20 24h/11 MCF7 8020 24h exp#2 Black nucleous.tif'
    target_periphery_file = 'data/MCF7 80-20 24h/11 MCF7 8020 24h exp#2 Black outside.tif'
    nucleus_center, nucleus_contour_shape, periphery_contour_shape = \
        load_nucleus_and_periphery_from_files(target_nucleus_file, target_periphery_file,
                                              do_plotting=True)
    xs = np.arange(1,1000,5)
    ys = np.arange(1,1000,5)
    xv, yv = np.meshgrid(xs, ys)
    zs = xv/yv
    for i in range(zs.shape[0]):
        for j in range(zs.shape[1]):
            x = xv[i,j]
            y = yv[i,j]
            zs[i,j]=get_perinuclearity_for_point(np.array((x,y)), nucleus_center, nucleus_contour_shape,
                                                 periphery_contour_shape)
            if zs[i,j] > 1:
                zs[i,j] = 1
    # f3 = plt.figure(3)
    # mask = zs < 1
    plt.contour(xv, yv, zs, levels = np.linspace(-1, 1, 11), cmap='coolwarm')
    plt.xlim(-20, 1000)
    plt.ylim(-20, 1000)
    plt.colorbar()
    f0.savefig('isoperinuclearity_levels.png', dpi=300)
    plt.show()

def compute_perinuclearity_for_data(target_nucleus_file,
                                    target_periphery_file, target_labels_file, do_plotting=False,
                                    bkg_correction=0,
                                    do_show=False):
    f0 = plt.figure(1, figsize=(6,6))
    nucleus_center, nucleus_contour_shape, periphery_contour_shape = \
        load_nucleus_and_periphery_from_files(target_nucleus_file, target_periphery_file,
                                              do_plotting=True)
    # labels_image = skimage.io.imread(target_labels_file, plugin='pil')[:,:,0] #
    # f1 = plt.figure(1)
    labels_image = tiff.imread(target_labels_file)[:,:,0]
    if bkg_correction:
        labels_image[labels_image <= bkg_correction] = bkg_correction
        labels_image = labels_image - bkg_correction
    # print(np.max(labels_image[:,:,0]))
    # print(np.max(labels_image[:, :, 1]))
    # print(np.max(labels_image[:, :, 2]))
    plt.imshow(np.transpose(labels_image, axes=(1, 0)), alpha=0.95, cmap='Greys')  #
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    if do_show:
        plt.show()
    weights = []
    perivalues = []
    perivalues_w = []
    for i in range(labels_image.shape[0]):
        for j in range(labels_image.shape[1]):
            if labels_image[i,j] > 0:
                x = j
                y = i
                P = get_perinuclearity_for_point(np.array((i,j)), nucleus_center, nucleus_contour_shape,
                                                 periphery_contour_shape, do_plotting=do_plotting)
                # if P < 0:
                #     plt.plot(i,j, 'o', color='green')
                perivalues.append(P)
                for k in range(int(round(labels_image[i,j]/50))):
                    perivalues_w.append(P)
                # weights.append(labels_image[i,j])
            # if i>300:
            #     plt.show()
        print(i)
    perivalues = np.array(perivalues)
    perivalues_w = np.array(perivalues_w)
    # weights = np.array(weights)
    np.save(target_labels_file + 'perivalues-w', perivalues_w)
    np.save(target_labels_file + 'perivalues', perivalues)
    perivalues = np.load(target_labels_file + 'perivalues.npy')
    perivalues_w = np.load(target_labels_file + 'perivalues-w.npy')
    # weights = np.load('weights.npy')
    # hist = np.histogram(perivalues, bins=20)#, weights=weights)
    # plt.imshow(labels_image[:,:,0])
    f0.savefig(target_labels_file + '_overlays.png', dpi=400)
    f2 = plt.figure(2)
    plt.hist(perivalues_w, bins=np.linspace(-1,1,100), density=True) #weights=weights,
    # plt.hist(perivalues, bins=200, color='green', density=True, alpha = 0.5)  # weights=weights,
    plt.xlim(-1,1)
    f2.savefig(target_labels_file + '_hist.png', dpi=400)
    f0.clf()
    f2.clf()
    del f0
    del f2

if __name__ == '__main__':
    test_perinuclearity(
            target_nucleus_file='data/MCF7 Control cell/Control/11_C_Black nucleous.tif',
            target_periphery_file='data/MCF7 Control cell/Control/11_C_Black outside.tif',
            target_dic_file="data/MCF7 Control cell/Control/11_MCf7 LT 50nM 30' no NPs bg700 td l250-1700_RGB.tif"
    )