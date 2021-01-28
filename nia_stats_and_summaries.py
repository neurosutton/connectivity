"""
Common analytic tools for non-imaging comparisons.


Author: Brianne Sutton, PhD
Date: Dec 2020
version: 0.1
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def summarize_group_differences(df, group_col, msrs, graph=False):
    msr_dict={msr:['mean','std','count'] for msr in msrs}
    result=df.groupby(group_col).agg(msr_dict).round(2).T.unstack()
    groups = list(set(df[group_col]))
    for msr in msr_dict.keys():
        grp1=df.loc[(df[group_col]==groups[0]),msr].dropna()
        grp2=df.loc[(df[group_col]==groups[1]),msr].dropna()
        result.loc[msr,('stats','pvalue')] = ttest_ind(grp1,grp2)[-1].round(3)
    print(result)

    if graph:
        keep = msrs + [group_col]
        tmp = df[keep]
        sns.pairplot(tmp,hue=group_col, palette='winter')
        plt.xticks(rotation=80)
        plt.show()
    return result
