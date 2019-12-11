import numpy as np;np.random.seed(1)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def print_stattests(perivalues_w, perivalues_for_test):
    ks_here = stats.ks_2samp(perivalues_w, perivalues_for_test)
    tt_here = stats.ttest_ind(perivalues_w, perivalues_for_test, equal_var=True)
    mw_here = stats.mannwhitneyu(perivalues_w, perivalues_for_test, alternative='two-sided')
    print('P-values: KS={0}, TT={1}, MW={2}'.format(ks_here, tt_here, mw_here))

a = []
feats = []
xs = []
datalist = [a, feats, xs]

def append_data(cellname, npy_file_8020, npy_file_control):
    perivalues_w = np.load(npy_file_control)
    perivalues_for_test = np.copy(perivalues_w)
    if len(datalist[0]) == 0:
        datalist[0] = np.copy(perivalues_w)
        datalist[1] = ["Control"] * len(perivalues_w)
        datalist[2] = [cellname] * len(perivalues_w)
    else:
        datalist[0] = np.concatenate((datalist[0], perivalues_w))
        datalist[1] = datalist[1] + ["Control"] * len(perivalues_w)
        datalist[2] = datalist[2] + [cellname] * len(perivalues_w)

    perivalues_w = np.load(npy_file_8020)
    datalist[0] = np.concatenate((datalist[0], perivalues_w))
    datalist[1] = datalist[1] + ["80:20"] * len(perivalues_w)
    datalist[2] = datalist[2] + [cellname] * len(perivalues_w)

    # print_stattests(perivalues_w, perivalues_for_test)

# celllines = ["Rat2", "MCF7", "HT1080", "CCD1058sk", "SKBR3", "MEF"]
for cellname in ["SKBR3", "MCF7"]:
    append_data(cellname,
                'data/{0} 8020_overall_perivalues.npy'.format(cellname),
                'data/{0} Control_overall_perivalues.npy'.format(cellname))

# append_data("MDA231\n12h", 'data/MDA231 8020 12h_overall_perivalues.npy', 'data/MDA231 Control_overall_perivalues.npy')
append_data("MDA231", 'data/MDA231 8020 24h_overall_perivalues.npy', 'data/MDA231 Control_overall_perivalues.npy')
for cellname in ["HT1080"]:
    append_data(cellname,
                'data/{0} 8020_overall_perivalues.npy'.format(cellname),
                'data/{0} Control_overall_perivalues.npy'.format(cellname))
# append_data("MCF10A\n12h", 'data/MCF10A 8020 12h_overall_perivalues.npy', 'data/MCF10A Control_overall_perivalues.npy')
append_data("MCF10A", 'data/MCF10A 8020 24h_overall_perivalues.npy', 'data/MCF10A Control_overall_perivalues.npy')
for cellname in ["MEF", "Rat2", "CCD1058sk"]:
    append_data(cellname,
                'data/{0} 8020_overall_perivalues.npy'.format(cellname),
                'data/{0} Control_overall_perivalues.npy'.format(cellname))

df = pd.DataFrame({"Perinuclearity" : datalist[0], "Condition" : datalist[1], "cells": datalist[2]})
fig = plt.figure(figsize=(40,5))
fig.add_subplot(1, 2, 1)
print('plotting violin')
ax = sns.violinplot(x='cells', y='Perinuclearity',
              data=df,
              scale='area',
              hue='Condition', split=True,
              palette={"Control": "lightgrey", "80:20": "magenta"},
              inner='quartile')
ax.tick_params(axis='y', direction='in')
plt.legend(bbox_to_anchor=(2, 1), loc='upper right', ncol=1)
plt.tight_layout()
fig.savefig('violins_3.png', dpi=600)
plt.show()