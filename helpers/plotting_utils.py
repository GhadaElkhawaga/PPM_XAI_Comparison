from numpy import percentile
import pandas as pd
import numpy as np
import os
import pickle


#a function to get specifications of figures to be plotted:
def get_fig_specs(no_figs,perm_flag=None):
    if no_figs >= 7 and perm_flag==False:
        nrows, ncols, figure_size, font_s = 3, 3, (6,4), 8
    elif no_figs >= 7 and perm_flag==True:
        nrows, ncols, figure_size, font_s = 4, 2, (6,6), 8
    elif no_figs > 2 and no_figs <= 6:
        ncols, figure_size, font_s = 2, (8,6), 8
        if no_figs%2 ==0:
            nrows = no_figs//2
        else:
            nrows = no_figs//2 +1
    else:
        nrows, figure_size, font_s = 1, (14,6), 12
        if no_figs == 2:
            ncols = 2
        else:
            ncols = 1
    return [nrows, ncols, figure_size, font_s]


#a functon to computer histogram bins- (lower and upper values are retrieved from the define_range method)
def compute_histogram_bins(lower, upper, desired_bin_size):
    min_boundary = -1.0 * (lower % desired_bin_size - lower)
    max_boundary = upper - upper % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins


#to define the lower and upper values in a certain feature and the percentage of outliers 
def define_range(X_feat, X, feat_idx):
    q25, q75 = percentile(X_feat, 25), percentile(X_feat, 75)
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers 
    outliers = [i for i in X_feat if i < lower or i > upper]
    outliers_indices = [np.where(X_feat==i)[0] for i in X_feat if i < lower or i > upper]
    #remove outliers from the feature column
    X_feat_outliers_removed = [x for x in X_feat if x >= lower and x <= upper]
    #remove outliers from the data array
    X_outliers_removed = X[(X[:, feat_idx]>= lower) &  (X[:, feat_idx]<= upper)]
    range = upper - lower
    if len(outliers) > len(X_feat)*0.05:
        results = [False, 'skip', lower, upper, len(outliers),X_feat_outliers_removed,X_outliers_removed]
    elif range < 0.5: #adopt broader scale to demonstrate the tight range
        results = [True, 'tight', lower, upper, len(outliers),X_feat_outliers_removed,X_outliers_removed]
    else: #adopt moderate scale to demonstrate the wider range
        results = [True, 'wide', lower, upper, len(outliers),X_feat_outliers_removed,X_outliers_removed]
    del outliers_indices, outliers,X_feat_outliers_removed,X_outliers_removed 
    return results

