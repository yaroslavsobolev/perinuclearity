import numpy as np
import matplotlib.pyplot as plt
import spm1d
import glob
from scipy import stats

folder_control = 'data/CCD1058sk Control'
folder_8020 = 'data/CCD1058sk 8020'
# data_8020 = np.load(npy_file_control)
# data_control = np.load(npy_file_8020)

nbins = 100
# target_folder = folder_control
def get_all_hists_from_folder(target_folder):
    filelist = [f for f in glob.glob('./' + target_folder + '/*perivalues-w.npy')]
    all_cells = []
    for f in filelist:
        data = np.load(f)
        all_cells.append(np.histogram(data, bins=nbins)[0])
    return np.array(all_cells)

def get_all_means_from_folder(target_folder):
    filelist = [f for f in glob.glob('./' + target_folder + '/*perivalues-w.npy')]
    all_cells = []
    for f in filelist:
        data = np.load(f)
        data = data[np.logical_not(np.isnan(data))]
        all_cells.append(np.mean(data[np.logical_and(data > 0, data < 1)]))
    return np.array(all_cells)
# f1 = plt.figure(1)
# nbins = 100
# hist1 = np.histogram(data_8020, bins=nbins)
# hist2 = np.histogram(data_control, bins=nbins)
# # print(hist1[0].shape)
# # print(hist1[1][1:].shape)
# plt.plot(0.5*(hist2[1][1:] + hist2[1][:-1]), hist2[0])
#
# f2 = plt.figure(2)
# # control_data = get_all_hists_from_folder(folder_control)
# # work_data = get_all_hists_from_folder(folder_8020)
# noise_frac = 0.1
# ndims = 10
# control_data = []
# for i in range(20):
#     control_data.append(np.sin(np.linspace(0, 10, ndims)) + noise_frac*np.random.randn(ndims))
#     plt.plot(control_data[i], color='blue', alpha=0.2)
# control_data = np.array(control_data)
# test_data = []
# for i in range(20):
#     test_data.append(np.sin(0.0+np.linspace(0, 10, ndims)) + noise_frac*np.random.randn(ndims))
#     plt.plot(test_data[i], color='red', alpha=0.2)
# test_data = np.array(test_data)
#
# T2 = spm1d.stats.hotellings2(control_data, test_data)
# T2i = T2.inference(0.05)
# print(T2i)
# # T2i.plot()
# plt.show()
def test_for_cell(cellname, foldertest=None, foldercontrol=None):
    if not foldertest:
        foldertest = 'data/{0} 8020'.format(cellname)
        foldercontrol = 'data/{0} Control'.format(cellname)
    test = get_all_means_from_folder(foldertest)
    control = get_all_means_from_folder(foldercontrol)
    ks_here = stats.ks_2samp(test,control)[1]
    tt_here = stats.ttest_ind(test,control, equal_var=True)[1]
    mw_here = stats.mannwhitneyu(test,control)[1]
    thresh = 0.005
    if ks_here < thresh or tt_here < thresh or mw_here < thresh:
        result = 'PASS'
    else:
        result = 'FAIL'
    print('CELLNAME:{4}, RES:{5} KS:{0}, TT:{1}, MW:{2}, meandiff={3}'.format(ks_here, tt_here, mw_here,
                                                        np.mean(test)-np.mean(control),
                                                        cellname,
                                                        result))

cellnames = ["SKBR3", "MCF7", "HT1080", "MEF", "Rat2", "CCD1058sk"]
for cellname in cellnames:
    test_for_cell(cellname, foldertest=None, foldercontrol=None)

test_for_cell("MDA231", 'data/MDA231 8020 24h', 'data/MDA231 Control')
test_for_cell("MCF10A", 'data/MCF10A 8020 24h', 'data/MCF10A Control')