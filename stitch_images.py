import numpy as np
import os
import skimage
import glob
import matplotlib.pyplot as plt

margins = 200
def load_and_cut(filename):
    return skimage.io.imread(filename)[margins:-margins, margins:-margins, :]

def stitch_all_from_folder(target_folder, saveto):
    filelist = [f for f in glob.glob('./' + target_folder + '/*overlays.png')]
    numfiles = len(filelist)
    print(numfiles)
    fileid = 0
    rowid = 0
    while fileid < numfiles:
        image = load_and_cut(filelist[fileid])
        for i in range(4):
            fileid += 1
            if fileid < numfiles:
                image2 = load_and_cut(filelist[fileid])
            else:
                print('blank')
                image2 = 255*np.ones_like(load_and_cut(filelist[0]))
            image = np.concatenate((image, image2), axis=1)
        if rowid == 0:
            image3 = np.copy(image)
        else:
            image3 = np.concatenate((image3, image), axis=0)
        rowid += 1
        fileid += 1
    skimage.io.imsave(saveto, image3)

celllines = ["Rat2", "MCF7", "HT1080", "CCD1058sk", "SKBR3", "MEF"]
folders_list = []
for cellname in  celllines:
    folders_list.append('data/{0} Control'.format(cellname))
    folders_list.append('data/{0} 8020'.format(cellname))
folders_list.extend(['data/MDA231 8020 12h', 'data/MDA231 8020 24h',
                     'data/MCF10A 8020 12h', 'data/MCF10A 8020 24h'])
for folder in folders_list:
    print('Stitching folder: ' + folder)
    stitch_all_from_folder(folder, folder+'_summary.png')