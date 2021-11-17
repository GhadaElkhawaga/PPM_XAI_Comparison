import os
import pandas as pd
import numpy as np
import csv
from helpers.Data_retrieval import get_corr_files, retrieve_datasets_info


preprocessing_method = ['single_agg', 'prefix_index']
datasets = ['sepsis1','sepsis2','sepsis3','traffic_fines', 'BPIC2017_O_Refused','BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled', 'hospital_billing_1', 'hospital_billing_2']
single_list = ['BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled', 'hospital_billing_1', 'hospital_billing_2']
info_dir = 'model_and_hdf5'
for p in preprocessing_method:
    EDA_output = 'EDA_output_%s' %(p)
    if p == 'prefix_index':
        datasets = np.setdiff1d(datasets,single_list)
    info_df = pd.read_csv(os.path.join(info_dir,'all_datasets_info.csv'),sep=';')
    info_df = info_df[(info_df.dataset_type.str.contains("training"))&( info_df.method.str.contains(p))&(info_df.dataset.isin(datasets))]
    grouped_info = info_df.groupby(['dataset','method'])
    for idx, group in grouped_info:
        if (idx[1] == 'single_agg')or (idx[0]=='traffic_fines'):
                    gap = 1
        elif (idx[1] == 'prefix_index') and (idx[0] in datasets):
                    gap = 5
        for row_idx, row in group.iterrows():
                if row['prefix_length'] in range(1, group.shape[0]+1,gap):
                          d, feat_num, prfx_len = idx[0], row['feature_num'], row['prefix_length']
                          corr_dict = {}
                          files = get_corr_files(EDA_output,d, p, 'xgboost', feat_num)
                          for f in files:
                             df = pd.read_csv(f, sep=';')
                             df.set_index(df.columns[0], drop=True, inplace=True)
                             for col in df:
                                #to get a list of tuples of(features highly correlated with the current column, correlation values)
                                corr_l = list(zip(df.index[(abs(df[col])>=0.8)].tolist(), df.loc[(abs(df[col])>=0.8), col].tolist()))
                                # delete the tuples of the current column and the encoded_label
                                corr_without = list(filter(lambda x: x[0] not in ['encoded_label', col],corr_l))
                                corr_dict[col] = corr_without
                          try:
                               with open(os.path.join(EDA_output,'correlations_count_%s_%s_%s_%s.csv' %(d,idx[1],feat_num, prfx_len)), 'w') as csvfile:
                                  for key in corr_dict.keys():
                                      csvfile.write("%s; %s\n" % (key, corr_dict[key]))
                          except IOError:
                                   print("I/O error")
                          correlations_df = pd.read_csv(os.path.join(EDA_output, 'correlations_count_%s_%s_%s_%s.csv' %(d,idx[1],feat_num, prfx_len)),sep=';')

