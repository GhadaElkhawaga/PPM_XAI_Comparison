import pandas as pd #using pandas==0.25
import numpy as np
import sys
import os
import pickle
import io
import warnings
warnings.simplefilter('ignore')
import csv
from alibi.explainers import ALE, plot_ale
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
from helpers.features_retrieval import get_important_features,get_correlated_features
from helpers.plotting_utils import get_fig_specs
from helpers.Data_retrieval import retrieve_artefact,get_corr_files
from plotting.plotting_comparisons import plot_perm, plot_shap_dependence

#Defining Basic parameters
method_name = 'single_agg'
#cls_method = 'logit'

#method_name = 'prefix_index'
gap = 1

if method_name == 'single_agg':
     datasets = ["sepsis1", "sepsis2", "sepsis3",'traffic_fines',"hospital_billing_1","hospital_billing_2", "BPIC2017_O_Accepted", "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"]
else:
     datasets = ["sepsis1", "sepsis2", "sepsis3",'traffic_fines',"BPIC2017_O_Refused"]

ALE_dir = 'ALE_artefacts'
shap_dir = 'shap_artefacts'
Perm_dir = 'Perm_artefacts'
comparisons_dir = 'comparisons_%s_%s' %(method_name,gap)
if not os.path.exists(comparisons_dir):
  os.makedirs(os.path.join(comparisons_dir))
ALE_comparisons = os.path.join(comparisons_dir,'ALE_comparisons')
if not os.path.exists(ALE_comparisons):
  os.makedirs(os.path.join(ALE_comparisons))
perm_comparisons = os.path.join(comparisons_dir,'Perm_comparisons')
if not os.path.exists(perm_comparisons):
  os.makedirs(os.path.join(perm_comparisons))
SHAP_comparisons = os.path.join(comparisons_dir,'SHAP_comparisons')
if not os.path.exists(SHAP_comparisons):
  os.makedirs(os.path.join(SHAP_comparisons))
saved_artefacts = 'model_and_hdf5'
if not os.path.exists(saved_artefacts):
  os.makedirs(os.path.join(saved_artefacts))
EDA_output = 'EDA_output_%s' % (method_name)
saved_exec_logs = 'output_text_files'
if not os.path.exists(saved_exec_logs):
  os.makedirs(os.path.join(saved_exec_logs))
encoded_datasets_dir = 'encoded_datasets_%s' %(method_name)
if not os.path.exists(encoded_datasets_dir):
  os.makedirs(os.path.join(encoded_datasets_dir))
output_file_perm = open(os.path.join(saved_exec_logs,'output_permComparisons_%s_%s.txt'%(method_name,gap)), 'w')
output_file_shap = open(os.path.join(saved_exec_logs,'output_SHAPComparisons_%s_%s.txt'%(method_name,gap)), 'w')
output_file_ale = open(os.path.join(saved_exec_logs,'output_ALEComparisons_%s_%s.txt'%(method_name,gap)), 'w')


def plot_ALE(folder, feat_indices, target_names, frmt_str, flag):
    if flag == 'positive':
        target = target_names[0]
        title_part = 'positiveClass'
    else:
        target = target_names[1]
        title_part = 'negativeClass'
    fig, ax = plt.subplots(1, len(feat_indices), figsize=(8, 4), sharey='row');
    plot_ale(ALE_xgb, features=feat_indices, targets=[target], ax=ax, line_kw={'label': 'XGB'});
    plot_ale(ALE_logit, features=feat_indices, targets=[target], ax=ax, line_kw={'label': 'LR'});
    plt.savefig(os.path.join(folder, 'ALE_comparison_%s_%s.png'
                             %(frmt_str,title_part)), dpi=300, bbox_inches='tight');
    plt.clf()

def compare_ALE(dataset_name, method_name, bkt_size, prfx_len, feat_num, gap):
    """
    comparing models according to ALE values, the algorithm is as follows:
    restore trained models from pickle files (xgboost and logit)
    get the most important feature according to xgboost.feature_importances_ and the weights through logit.coef_
    plot both models' ALE values (for the most important features) from the explainers for both models
    """

    frmt_str = '%s_%s_%s_%s' % (dataset_name, method_name, bkt_size, prfx_len)
    # decide which feature whose ALE effects will be used to compare the models
    plotted_features = get_important_features(saved_artefacts, 1, dataset_name, method_name, bkt_size, prfx_len,
    if isinstance(plotted_features, str):
        return

    # load ALE objects and explainers for both classifiers:
    ALE_xgb = retrieve_artefact(ALE_dir, '.pickle', 'ALE_pred_explainer', 'xgboost', frmt_str)
    ALE_logit = retrieve_artefact(ALE_dir, '.pickle', 'ALE_pred_explainer', 'logit', frmt_str)
    ALE_features, target_names = list(ALE_xgb.feature_names), list(ALE_xgb.target_names)
    # a feature may have high importance (XGB) or coeff(logit), but maybe it wasn't used in ALE computations
    feat_indices = [ALE_features.index(f) for f in plotted_features if f in ALE_features]
    try:
        # compare models (positive class)
        plot_ALE(ALE_comparisons, feat_indices, target_names, frmt_str, 'positive')
    except:
        pass

    try:
        # compare models (negative class)
        plot_ALE(ALE_comparisons, feat_indices, target_names, frmt_str, 'negative')
    except:
        pass


def compare_perm(dataset_name, method_name, bkt_size, prfx_len, feat_num, gap, files):
    """
    this function has multiple purposes:
    (1) getting highly important features according to XGBoost and checking their importance using permutations
    this works as a trial to check flaws in xgboost feature importance based on gain
    (2) getting highly correlated features and checking their permutation importance.
    this acts as a trial to discover how permutations are assigned to correlated features (they are expected to have low importance)
    (3) exploring importances of the first two important features at each classifier
    """
    frmt_str = '%s_%s_%s_%s' % (dataset_name, method_name, bkt_size, prfx_len)
    # list containing the most important feature to each classifier (the first two features)
    xgb_logit_features = get_important_features(saved_artefacts, 2, dataset_name, method_name, bkt_size, prfx_len,
                                                feat_num, gap)
    if not isinstance(xgb_logit_features, str):
        try:
            file_name = 'permutation_importance_%s' % (dataset_name)
            Id_str = '%s_%s_%s' % (method_name, bkt_size, prfx_len)
            perm_logit_df = retrieve_artefact(Perm_dir, '_final.csv', file_name, 'logit', Id_str, feat_num)
            perm_xgboost_df = retrieve_artefact(Perm_dir, '_final.csv', file_name, 'xgboost', Id_str, feat_num)
            # retrieving the permutations of the xgb_logit_features (important features according to each classifier)
            specs_imp = get_fig_specs(len(xgb_logit_features),True)
            plot_perm(perm_comparisons, frmt_str, specs_imp, perm_logit_df, perm_xgboost_df, xgb_logit_features,
                      'Importance')
        except:
            pass
    # get features with high correlation together
    for i in files:
        corr_df = pd.read_csv(i, sep=';')
        corr_df.set_index(corr_df.columns[1:], inplace=True)
        # a dictionary containing highly correlated features:
        corr_with_others = get_correlated_features(corr_df, 0.8, 'others', output_file_perm)

        feat_list, feat_values = [], []
        feat_list.extend(list(corr_with_others.keys()))
        for j in corr_with_others.values():
            feat_values.extend(j)

        feat_list.extend(feat_values)
        feat_list = list(set(feat_list))
        limit = 8  # max number of correlated elements to be plotted
        if len(feat_list) >= limit:
            feat_list = feat_list[0:limit]
        if i == files[1]:  # num_csv
            plot_type = 'CorrNumerical'
        else:
            plot_type = 'CorrCategorical'
        specs_corr = get_fig_specs(len(feat_list),True)
        # plot_perm(perm_comparisons,frmt_str, specs_corr, perm_logit_df, perm_xgboost_df,corr_with_others,'Correlations')
        try:
            plot_perm(perm_comparisons, frmt_str, specs_corr, perm_logit_df, perm_xgboost_df, feat_list, plot_type)
        except:
            pass


def compare_shap(dataset_name, method_name, bkt_size, prfx_len, feat_num, files):
    """
    this function serves two purposes:
    (1) get the highly important features according to xgboost and logit and compare their shap values in the same figure
    features to be compared should have no more than 10% of its shap values = zero
    (2) plotting also dependence plots of features which have high correlation together
    """
    # load shap values files and explainers for both classifiers
    frmt_str = '%s_%s_%s_%s_%s_%s' % (dataset_name, method_name, 'training', bkt_size, prfx_len, feat_num)
    shap_values_xgboost = retrieve_artefact(shap_dir, '.pickle', 'shap_values_xgboost', frmt_str)
    shap_values_logit = retrieve_artefact(shap_dir, '.pickle', 'shap_values_logit', frmt_str)
    encoded_dataset = retrieve_artefact(encoded_datasets_dir, '.csv', 'encoded_training', dataset_name, method_name,
                                        bkt_size, prfx_len, feat_num)
    ffeatures = retrieve_artefact(saved_artefacts, '.pickle', 'ffeatures', 'xgboost', dataset_name, method_name,
                                  bkt_size, prfx_len)
    # get the highest important feature for each classifier
    xgb_logit_features = get_important_features(saved_artefacts, 1, dataset_name, method_name, bkt_size, prfx_len,
                                                feat_num, gap)
    # plot dependence plots of these features
    specs_imp_shap = get_fig_specs(2, False)
    if not isinstance(xgb_logit_features, str):
        for x in xgb_logit_features:
            plot_shap_dependence(SHAP_comparisons, frmt_str, specs_imp_shap, shap_values_xgboost, shap_values_logit, x,
                                 'Importance', encoded_dataset, ffeatures)

    # get the highest correlated features
    compared_features = get_important_features(saved_artefacts, 3, dataset_name, method_name, bkt_size, prfx_len,
                                               feat_num, gap)
    for i in files:
        corr_df = pd.read_csv(i, sep=';')
        corr_df.set_index(corr_df.columns[1:], inplace=True)
        # a dictionary containing highly correlated features:
        corr_with_others = get_correlated_features(corr_df, 0.8, 'others', output_file_shap)
        plotted_features = []
        feat_list = list(corr_with_others.keys())
        for y in compared_features:
            for x in feat_list:
                if x == y:
                    plotted_features.append(x)
                    continue
        limit = 9  # max number of elements to be plotted
        if i == files[1]:  # num_csv
            plot_type = 'CorrNumerical'
        else:
            plot_type = 'CorrCategorical'

        # plot dependence plots of these features
        if len(plotted_features) > 0:
            if len(plotted_features) >= limit:
                plotted_features = plotted_features[0:limit]
            for y in plotted_features:
                plot_shap_dependence(SHAP_comparisons, frmt_str, specs_imp_shap, shap_values_xgboost, shap_values_logit,
                                     y, plot_type, encoded_dataset, ffeatures)


for dataset_name in datasets:
    output_file_perm = open(os.path.join(saved_exec_logs, 'output_permComparisons_%s_%s.txt' % (method_name, gap)), 'a')
    output_file_shap = open(os.path.join(saved_exec_logs, 'output_SHAPComparisons_%s_%s.txt' % (method_name, gap)), 'a')
    output_file_ale = open(os.path.join(saved_exec_logs, 'output_ALEComparisons_%s_%s.txt' % (method_name, gap)), 'a')
    info_df = pd.read_csv(os.path.join(saved_artefacts, 'all_datasets_info.csv'), sep=';')
    # to drop rows containing info about testing datasets:
    info_df = info_df[~info_df.dataset_type.str.contains("testing")]
    info_df.drop(['dataset_type'], inplace=True, axis=1)
    grouped_info = info_df.groupby(['dataset', 'method'])

    for idx, group in grouped_info:
        if idx[0] == dataset_name and idx[1] == method_name:
            for i in range(1, group.shape[0] + 1, gap):
                for row_idx, row in group.iterrows():
                    if i == row['prefix_length']:
                        bkt_size, feat_num, prfx_len = row['bucket_size'], row['feature_num'], row['prefix_length']
                        files = get_corr_files(EDA_output, dataset_name, method_name, 'xgboost', feat_num)
                        compare_ALE(row['dataset'], row['method'], bkt_size, prfx_len, feat_num, gap)
                        compare_perm(row['dataset'], row['method'], bkt_size, prfx_len, feat_num, gap, files)
                        compare_shap(row['dataset'], row['method'], bkt_size, prfx_len, feat_num, files)

    output_file_ale.close()
    output_file_perm.close()
    output_file_shap.close()
