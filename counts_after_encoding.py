import pandas as pd
import csv
import os
import numpy as np
from helpers.Data_retrieval import retrieve_file
from helpers.DatasetManager import DatasetManager


datasets = ['sepsis1','sepsis2','sepsis3','hospital_billing_1','hospital_billing_2', 'BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled', 'BPIC2017_O_Refused','traffic_fines']
max_len = {'sepsis1':[17], 'sepsis2':[12],'sepsis3':[32],'traffic_fines':[11],'BPIC2017_O_Refused':[17]}
in_dir = 'encoded_datasets'
out_dir = 'datasets_counts'
if not os.path.exists(out_dir):
  os.makedirs(os.path.join(out_dir))
models_dir = 'model_and_hdf5' 
datasets_info_file = os.path.join(models_dir, 'all_datasets_info.csv')
subject_datasets_dict = {"dataset_name":[],"method":[],"bkt_size":[],"prfx_len":[],"feat_num":[]};
info_df = pd.read_csv(datasets_info_file,sep=';')
#to drop rows containing info about training datasets:
training_info_df = info_df[info_df.dataset_type.str.contains("training")]
training_info_df.drop(['dataset_type'], inplace=True, axis=1)
training_grouped_info = training_info_df.groupby(['dataset','method'])
encoded_datasets = []
for idx, group in training_grouped_info:
          for row_idx, row in group.iterrows():
                  d_name, method, bkt_size, prfx_len, feat_num = idx[0], idx[1], row['bucket_size'], row['prefix_length'], row['feature_num']
                  if method == 'prefix_index' and d_name not in ['hospital_billing_1', 'hospital_billing_2', 'BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled']:
                      gap = 5
                      for i in range(1, group.shape[0]+1, gap):
                          if i == prfx_len:
                              f_name = 'encoded_training_%s_%s_%s_%s_%s' %(d_name, method, bkt_size, prfx_len, feat_num)
                              encoded_datasets.append(f_name)
                  elif method == 'single_agg':
                      f_name = 'encoded_training_%s_%s_%s_%s_%s' %(d_name, method, bkt_size, prfx_len, feat_num)
                      encoded_datasets.append(f_name)
                  else:
                      break
for d in datasets:
   dm = DatasetManager(d)
   for encfile in encoded_datasets:
     if d in encfile:
         f = encfile + '.csv'
         df = pd.read_csv(os.path.join(in_dir,f), sep=';')
         counts_df = df.nunique(dropna=False)
         counts_df = counts_df.to_frame()
         counts_df.columns = ['values']
         counts_df.reset_index(level=0, inplace=True)
      
         # get indices of numerical features in the counts_df
         original_num_cols = dm.dynamic_num_cols + dm.static_num_cols
         feat_cat_indices = list(set([i for i, feature_name in enumerate(counts_df['index']) if not (
                          any(feat in feature_name for feat in original_num_cols) and not (
                      any(s in feature_name for s in ['concept:name', 'True', 'False', 'other'])))]))
      
         counts_df['values_count'] = 'Numerical Variable'
         for cat_col_idx in feat_cat_indices:
                  if counts_df.loc[cat_col_idx, 'values'] <= 20:
                      col = counts_df.loc[cat_col_idx, 'index']
                      counts_df.loc[cat_col_idx, 'values_count'] = [df[col].value_counts().to_dict()]
                  else:
                      counts_df.loc[cat_col_idx, 'values_count'] = 'more than 20 cat levels'
      
         for col in df:
                  for x in counts_df['index']:
                      if x == col:
                          counts_df['type'] = df.dtypes[col]
      
         counts_df.to_csv(os.path.join(out_dir, 'counts_file_after_encoding_%s.csv' % (encfile)),
                               sep=';', index=False)

          