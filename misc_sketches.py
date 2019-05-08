from perinuclearity import *

# labels_image = tiff.imread('data/MCF10A Control test/01_MCF10A red.tif')[:, :, 0]
# print('Reading tiff test')
# compute_perinuclearity_for_data(
#     target_nucleus_file='data/tests/MCF7 80-20 24h/11 MCF7 8020 24h exp#2 Black nucleous.tif',
#     target_periphery_file='data/tests/MCF7 80-20 24h/11 MCF7 8020 24h exp#2 Black outside.tif',
#     target_labels_file='data/tests/16bit_test/11 MCF7 8020 24h exp#2 bg400_RGB_Cy3 dye–labeled IgG antibody_pH 7.2.tif'
# )

# target_labels_file = 'data/MCF7 8020/01_MCF7 8020 red.tif'
# perivalues_w = np.load(target_labels_file + 'perivalues-w.npy')
# # weights = np.load('weights.npy')
# # hist = np.histogram(perivalues, bins=20)#, weights=weights)
# # plt.imshow(labels_image[:,:,0])
# # f0.savefig(target_labels_file + '_overlays.png', dpi=400)
# f2 = plt.figure(2)
# plt.hist(perivalues_w, bins=np.linspace(-1,1,100), density=True) #weights=weights,
# # plt.hist(perivalues, bins=200, color='green', density=True, alpha = 0.5)  # weights=weights,
# plt.xlim(-1,1)
# plt.show()


# f3 = plt.figure(3, figsize=(10,5))
# data = []
# for pos, base_filename in enumerate(base_filename_list[:]):
#     print('Plotting file {0}'.format(base_filename))
#     perivalues_w = np.load(target_folder + '/' + base_filename + 'red.tifperivalues-w.npy')
#     mask = np.logical_and(perivalues_w > 0, perivalues_w < 1)
#     data.append(perivalues_w[mask])
# # kdfactor = (len(data[0]))**(-1./(1+4))
# print('All loaded. Plotting.')
# plt.violinplot(data, range(len(data)), points=60, widths=0.7, showmeans=False,
#               showextrema=False, showmedians=False)
# plt.boxplot(data, positions=range(len(data)), whis='range')
# plt.ylim([0,1])
# plt.ylabel('Perinuclearity')
# plt.show()


# plt.violinplot(data, [1], points=60, widths=0.7, showmeans=False,
#               showextrema=False, showmedians=False)
# plt.boxplot(data, positions=[1], whis='range')
# plt.ylim([0,1])
# plt.ylabel('Perinuclearity')
# f3.savefig(target_folder + '_netboxplot.png')
# plt.show()


# compute_perinuclearity_for_data(
#     target_nucleus_file='data/MCF7 Control cell/Control/11_C_Black nucleous.tif',
#     target_periphery_file='data/MCF7 Control cell/Control/11_C_Black outside.tif',
#     target_labels_file="data/16bit_test/11 MCF7 8020 24h exp#2 bg400_RGB_Cy3 dye–labeled IgG antibody_pH 7.2.tif",
#     do_plotting=True
# )