from perinuclearity import *

nucname = 'black nucleous.tif'
outsidename = 'black outside.tif'
redname = 'red.tif'


def process_folder(numlist, base_filename_list, target_folder, nucname, outsidename):
    compute_folder = True
    if compute_folder:
        for base_filename in base_filename_list[:]:
            print('Processing file {0}'.format(base_filename))
            target_file = target_folder + '/' + base_filename
            compute_perinuclearity_for_data(
                target_nucleus_file=target_file + nucname,
                target_periphery_file=target_file + outsidename,
                target_labels_file=target_file + redname,
                bkg_correction=0,
                do_show=False
            )

    # f3 = plt.figure(3, figsize=(3,5))
    for pos, base_filename in enumerate(base_filename_list[:]):
        print('Loading file {0}'.format(base_filename))
        perivalues_w = np.load(target_folder + '/' + base_filename + 'red.tifperivalues-w.npy')
        mask = np.logical_and(perivalues_w > 0, perivalues_w < 1)
        if pos == 0:
            data = np.copy(perivalues_w[mask])
        else:
            data = np.concatenate((data, perivalues_w[mask]))

    np.save(target_folder + '_overall_perivalues.npy', data)

# ####### MCF DATASET
numlist = list(range(13))
numlist.remove(7)
numlist.remove(0)
base_filename_list=['{0:02d}_MCF7 8020 '.format(n) for n in numlist]
target_folder = 'data/MCF7 8020'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = list(range(12))
# numlist.remove(7)
numlist.remove(0)
base_filename_list=['{0:02d}_MCF7 '.format(n) for n in numlist]
target_folder = 'data/MCF7 Control'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

#### RAT DATASET
numlist = list(range(11))
# numlist.remove(7)
numlist.remove(0)
base_filename_list=['{0:02d}_Rat2 8020 '.format(n) for n in numlist]
target_folder = 'data/Rat2 8020'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = list(range(14))
# numlist.remove(7)
numlist.remove(8)
numlist.remove(0)
base_filename_list=['{0:02d}_Rat2 '.format(n) for n in numlist]
target_folder = 'data/Rat2 Control'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

# # ####### HT1080 DATASET
numlist = list(range(21))
numlist.remove(3)
numlist.remove(5)
numlist.remove(10)
numlist.remove(12)
numlist.remove(14)
numlist.remove(0)
base_filename_list=['{0}_HT1080 8020 '.format(n) for n in numlist]
target_folder = 'data/HT1080 8020'
nucname = 'td black nucleous.tif'
outsidename = 'td black outside.tif'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = [11, 12, 21, 26, 28, 31]
base_filename_list=['{0}_HT1080 '.format(n) for n in numlist]
target_folder = 'data/HT1080 Control'
nucname = 'td black nucleous.tif'
outsidename = 'td black outside.tif'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

## CCD1058sk
numlist = list(range(14))
numlist.remove(0)
numlist.remove(8)
numlist.remove(10)
numlist.remove(12)
base_filename_list=['{0:02d}_CCD1058sk 8020 '.format(n) for n in numlist]
target_folder = 'data/CCD1058sk 8020'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = list(range(12))
numlist.remove(0)
numlist.remove(10)
base_filename_list=['{0:02d}_CCD1058sk '.format(n) for n in numlist]
target_folder = 'data/CCD1058sk Control'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

##SKBR dataset
numlist = list(range(17))
numlist.remove(0)
numlist.remove(6)
numlist.remove(7)
numlist.remove(10)
numlist.remove(11)
base_filename_list=['{0:02d}_SKBR3 8020 '.format(n) for n in numlist]
target_folder = 'data/SKBR3 8020'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = [3,4,6,7,8,9,10,110,111,12,13]
base_filename_list=['{0:02d}_SKBR3 '.format(n) for n in numlist]
target_folder = 'data/SKBR3 Control'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

## MDA231
numlist = [1,3,4,5,6,7,101,102,11,14,16]
base_filename_list=['{0:02d}_MDA231 '.format(n) for n in numlist]
target_folder = 'data/MDA231 Control'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = [2,3,4,7,9,10,11,12,42,44,46,48,50,51,52]
base_filename_list=['{0:02d}_MDA231 8020 '.format(n) for n in numlist]
target_folder = 'data/MDA231 8020 12h'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = [20,21,22,23,24,30,31,32,33,34,35,37,38,39]
base_filename_list=['{0:02d}_MDA231 8020 '.format(n) for n in numlist]
target_folder = 'data/MDA231 8020 24h'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

## MEF
numlist = [1,2,3,5,6,7,8,9,10,11,12,13]
base_filename_list=['{0:02d}_MEF 8020 '.format(n) for n in numlist]
target_folder = 'data/MEF 8020'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = [1,3,5,6,7,8,9,13,14]
base_filename_list=['{0:02d}_MEF '.format(n) for n in numlist]
target_folder = 'data/MEF Control'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

### MCF10
numlist = [1,2,3,4,5,6,7]
base_filename_list=['{0:02d}_MCF10A '.format(n) for n in numlist]
target_folder = 'data/MCF10A Control'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = [1,2,3,4,5,6]
base_filename_list=['{0:02d}_MCF10A 8020 '.format(n) for n in numlist]
target_folder = 'data/MCF10A 8020 12h'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)

numlist = [12, 13, 14, 15, 16, 171, 172, 19, 20]
base_filename_list = ['{0:02d}_MCF10A 8020 '.format(n) for n in numlist]
target_folder = 'data/MCF10A 8020 24h'
nucname = 'black nucleous.tif'
outsidename = 'black outside.tif'
process_folder(numlist, base_filename_list, target_folder, nucname, outsidename)